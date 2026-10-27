"""
Faithfulness ("does the answer stick to the source?") and answer-quality judges.

Two judges share one interface, judge(question, answer, context_chunks, reference):

  LexicalJudge  offline and deterministic. Splits the answer into sentence
                claims and marks a claim supported when >= 75% of its content
                words occur within two adjacent sentences of the retrieved
                context and every number in it appears in the context. Good at
                catching copied-in material that isn't in the sources and
                changed numbers; it under-credits faithful paraphrase. Its
                error rates are measured by `calibrate()` so the numbers it
                produces can be read with the right amount of salt.

  LLMJudge      claim decomposition + verification by an LLM (Gemini), plus a
                correct/partial/incorrect grade against the reference answer.
                This is the standard RAGAS-style faithfulness metric. It needs
                GEMINI_API_KEY; its token cost is tracked separately from the
                pipeline's.

Both also report abstention: whether the answer declines with the app's
"insufficient evidence" wording, which is right for unanswerable questions and
wrong otherwise.
"""
import json
import random
import re
from typing import Dict, List, Optional, Sequence

from app.core.config import settings
from app.core.telemetry import record_llm_response, trace
from app.retrieval.chunker import split_sentences
from evaluation.dataset import Corpus, QAItem, normalize

ABSTAIN_PATTERNS = re.compile(
    r"insufficient evidence|do not contain sufficient|does not contain (?:enough|sufficient)|"
    r"not (?:mentioned|covered|addressed) in the (?:provided )?documents", re.I)

_STOP = set("""a an the and or but if then else of to in on at by for with from into onto about as is are was were be
been being it its this that these those there here which who whom whose what when where why how not no nor so than too
very can could should would may might must shall will do does did done has have had having i you he she we they them
their our your his her my me us also just only such any all each every some more most other own same both either neither
one per via etc e g ie""".split())
_CITATION = re.compile(r"\*?\[(?:Doc|Source)[^\]]*\]\*?", re.I)
_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_SOURCE_LABEL = re.compile(r"^\S+\.(?:txt|md|pdf|docx?|html?|json|csv|py) \(Page \d+", re.I)


def is_abstention(answer: str) -> bool:
    return bool(ABSTAIN_PATTERNS.search(answer or ""))


def _content_words(text: str) -> set:
    return {w for w in re.findall(r"[a-z0-9_]+", text.lower())
            if w not in _STOP and (len(w) > 2 or w.isdigit())}


def extract_claims(answer: str, min_words: int = 4) -> List[str]:
    """Sentences of the answer that make a checkable statement."""
    text = _CITATION.sub("", answer or "")
    claims = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "⚠️" in line or line.startswith("---"):
            continue
        line = re.sub(r"[*_`]+", "", line).lstrip("-• ").strip()
        if (line.endswith(":") and len(line) < 120) or _SOURCE_LABEL.match(line):  # "file.txt (Page 1, ...):"
            continue
        for _, sentence in split_sentences(line):
            if len(_content_words(sentence)) >= min_words:
                claims.append(sentence)
    return claims


class LexicalJudge:
    name = "lexical"

    def __init__(self, support_threshold: float = 0.75, window: int = 2):
        self.support_threshold = support_threshold
        self.window = window

    def _windows(self, context: str) -> List[set]:
        sents = [_content_words(s) for _, s in split_sentences(context)]
        if not sents:
            return []
        return [set().union(*sents[i:i + self.window]) for i in range(max(1, len(sents) - self.window + 1))]

    def claim_support(self, claim: str, windows: List[set], context_numbers: set) -> float:
        words = _content_words(claim)
        if not words:
            return 1.0
        if not set(_NUMBER.findall(claim)) <= context_numbers:
            return 0.0
        return max((len(words & w) / len(words) for w in windows), default=0.0)

    def faithfulness(self, answer: str, context: str) -> Dict:
        claims = extract_claims(answer)
        if not claims:
            return {"faithfulness": None, "n_claims": 0, "unsupported": []}
        windows = self._windows(context)
        numbers = set(_NUMBER.findall(context))
        unsupported = [c for c in claims if self.claim_support(c, windows, numbers) < self.support_threshold]
        return {
            "faithfulness": 1 - len(unsupported) / len(claims),
            "n_claims": len(claims),
            "unsupported": unsupported,
        }

    def judge(self, question: str, answer: str, context_chunks: Sequence[dict],
              reference: Optional[str]) -> Dict:
        context = "\n\n".join(c["text"] for c in context_chunks)
        out = self.faithfulness(answer, context)
        out["abstained"] = is_abstention(answer)
        if reference:
            ref = _content_words(reference)
            out["reference_token_recall"] = len(ref & _content_words(answer)) / len(ref) if ref else None
        return out


FAITHFULNESS_PROMPT = """You are grading whether an ANSWER is supported by SOURCE passages.
Split the answer into atomic factual claims. For each claim decide whether the SOURCES directly support it.
Ignore citation markers and formatting. A statement that the sources lack information counts as supported.
Return only JSON, no markdown: {{"claims": [{{"claim": "...", "supported": true}}]}}

SOURCES:
{context}

ANSWER:
{answer}
"""

CORRECTNESS_PROMPT = """Compare the ANSWER to the REFERENCE answer for the QUESTION.
"correct" = contains the reference's key facts and nothing contradicting them; "partial" = some key facts;
"incorrect" = missing or wrong. Return only JSON, no markdown: {{"verdict": "correct", "reason": "..."}}

QUESTION: {question}
REFERENCE: {reference}
ANSWER: {answer}
"""


def _parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        text = text[4:] if text.startswith("json") else text
    return json.loads(text)


class LLMJudge:
    name = "llm"

    def __init__(self, model: Optional[str] = None, client=None):
        self.model = model or settings.LLM_MODEL
        if client is None:
            if not settings.has_llm_key:
                raise RuntimeError("LLM judge needs GEMINI_API_KEY (or the api_key_here file).")
            from google import genai
            client = genai.Client(api_key=settings.GEMINI_API_KEY)
        self.client = client
        self.usage = []  # judge token usage, kept apart from the pipeline's cost

    def _ask(self, purpose: str, prompt: str) -> dict:
        with trace() as t:
            response = self.client.models.generate_content(model=self.model, contents=prompt)
            text = response.text or ""
            record_llm_response(purpose, self.model, response, prompt, text)
        self.usage.append(t.summary())
        return _parse_json(text)

    def faithfulness(self, answer: str, context: str) -> Dict:
        try:
            claims = self._ask("judge_faithfulness",
                               FAITHFULNESS_PROMPT.format(context=context, answer=answer)).get("claims", [])
        except Exception as e:  # malformed JSON or API error: record, don't crash the run
            return {"faithfulness": None, "n_claims": 0, "unsupported": [], "error": str(e)}
        if not claims:
            return {"faithfulness": None, "n_claims": 0, "unsupported": []}
        unsupported = [c.get("claim", "") for c in claims if not c.get("supported")]
        return {"faithfulness": 1 - len(unsupported) / len(claims), "n_claims": len(claims),
                "unsupported": unsupported}

    def judge(self, question: str, answer: str, context_chunks: Sequence[dict],
              reference: Optional[str]) -> Dict:
        context = "\n\n".join(f"[{i + 1}] {c['text']}" for i, c in enumerate(context_chunks))
        out = self.faithfulness(answer, context)
        out["abstained"] = is_abstention(answer)
        if reference:
            try:
                verdict = self._ask("judge_correctness", CORRECTNESS_PROMPT.format(
                    question=question, reference=reference, answer=answer)).get("verdict")
                out["correctness"] = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}.get(verdict)
            except Exception as e:
                out["correctness"] = None
                out["correctness_error"] = str(e)
        return out


def get_judge(name: str):
    if name == "lexical":
        return LexicalJudge()
    if name == "llm":
        return LLMJudge()
    raise ValueError(f"Unknown judge {name!r}")


# ---------------------------------------------------------------------------
# Judge calibration: run the judge on answers whose faithfulness we know.
# ---------------------------------------------------------------------------

def _context_window(doc_text: str, span: str, radius: int = 3000) -> str:
    """~1.5k tokens around the evidence, roughly what top-5 retrieval feeds the LLM."""
    pattern = r"\s+".join(re.escape(w) for w in span.split())
    match = re.search(pattern, doc_text, re.IGNORECASE)
    centre = match.start() if match else 0
    return doc_text[max(0, centre - radius): centre + radius]


def calibrate(judge, items: List[QAItem], corpus: Corpus, seed: int = 0) -> Dict:
    """
    Builds four kinds of answers with known labels from the golden set:
      faithful_extractive   the gold evidence itself          (should score 1.0)
      faithful_abstractive  the hand-written reference answer (should score ~1.0)
      injected              evidence + one sentence copied from a different PEP;
                            the injected sentence should be flagged
      number_swap           evidence with every digit changed (should be flagged)
    """
    rng = random.Random(seed)
    answerable = [i for i in items if i.answerable]
    foreign_pool = {
        name: [s for _, s in split_sentences(doc["raw_text"]) if len(_content_words(s)) >= 8 and not s.startswith("#")]
        for name, doc in corpus.docs.items()
    }
    extractive, abstractive, injected_hits, swap_hits = [], [], [], []
    for item in answerable:
        ev = item.evidence[0]
        context = _context_window(corpus.docs[ev.doc]["raw_text"], ev.text)
        # Newline-joined so separate spans stay separate claims.
        evidence_answer = "\n".join(e.text for e in item.evidence if e.doc == ev.doc)
        extractive.append(judge.faithfulness(evidence_answer, context)["faithfulness"])
        if item.reference_answer:
            abstractive.append(judge.faithfulness(item.reference_answer, context)["faithfulness"])

        other_docs = [d for d in foreign_pool if d != ev.doc]
        foreign = rng.choice(foreign_pool[rng.choice(other_docs)])
        result = judge.faithfulness(evidence_answer + "\n" + foreign, context)
        flagged = any(normalize(foreign)[:40] in normalize(u) or normalize(u)[:40] in normalize(foreign)
                      for u in result["unsupported"])
        injected_hits.append(float(flagged))

        if _NUMBER.search(evidence_answer):
            swapped = re.sub(r"\d", lambda m: str((int(m.group()) + 3) % 10), evidence_answer)
            r = judge.faithfulness(swapped, context)
            swap_hits.append(float(bool(r["unsupported"])))

    def avg(xs):
        xs = [x for x in xs if x is not None]
        return round(sum(xs) / len(xs), 3) if xs else None

    return {
        "judge": judge.name,
        "faithful_extractive_score": avg(extractive),
        "faithful_abstractive_score": avg(abstractive),
        "injected_claim_detection_rate": avg(injected_hits),
        "number_swap_detection_rate": avg(swap_hits),
        "n": {"extractive": len(extractive), "abstractive": len(abstractive),
              "injected": len(injected_hits), "number_swap": len(swap_hits)},
    }
