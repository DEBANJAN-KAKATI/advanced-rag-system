import math
import re
from collections import Counter, defaultdict
from typing import Dict, List

STOPWORDS = frozenset("""
a about above after again against all am an and any are as at be because been before being below between both
but by can could did do does doing down during each few for from further had has have having he her here hers
herself him himself his how i if in into is it its itself just me more most my myself no nor not now of off on
once only or other our ours ourselves out over own same she should so some such than that the their theirs them
themselves then there these they this those through to too under until up very was we were what when where which
while who whom why will with would you your yours yourself yourselves tell show give explain describe please
document documents doc file says say mention mentions
""".split())

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.'][a-z0-9]+)*")


def _stem(token: str) -> str:
    """Very light stemming so plurals and simple verb forms match ("risks" ~ "risk")."""
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def tokenize(text: str) -> List[str]:
    tokens = []
    for raw in _TOKEN_RE.findall((text or "").lower()):
        token = raw.strip(".'")
        if not token or token in STOPWORDS:
            continue
        if len(token) == 1 and not token.isdigit():
            continue
        tokens.append(_stem(token))
    return tokens


class BM25Index:
    """
    Okapi BM25 over an inverted index. Uses the non-negative IDF variant
    log(1 + (N - df + 0.5) / (df + 0.5)), so terms in small corpora still score.
    """

    def __init__(self, documents: List[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_count = len(documents)
        self.doc_lengths: List[int] = []
        self.postings: Dict[str, List[tuple]] = defaultdict(list)
        for idx, text in enumerate(documents):
            counts = Counter(tokenize(text))
            self.doc_lengths.append(sum(counts.values()))
            for term, tf in counts.items():
                self.postings[term].append((idx, tf))
        self.avg_len = (sum(self.doc_lengths) / self.doc_count) if self.doc_count else 0.0

    def idf(self, term: str) -> float:
        df = len(self.postings.get(term, ()))
        return math.log(1.0 + (self.doc_count - df + 0.5) / (df + 0.5))

    def scores(self, query: str) -> Dict[int, float]:
        """Returns {doc_index: score} for documents containing at least one query term."""
        result: Dict[int, float] = defaultdict(float)
        if not self.doc_count:
            return result
        for term in set(tokenize(query)):
            postings = self.postings.get(term)
            if not postings:
                continue
            idf = self.idf(term)
            for idx, tf in postings:
                norm = self.k1 * (1 - self.b + self.b * self.doc_lengths[idx] / (self.avg_len or 1.0))
                result[idx] += idf * tf * (self.k1 + 1) / (tf + norm)
        return result


def term_coverage(query: str, text: str) -> float:
    """Fraction of distinct query terms present in text (0-1)."""
    q_terms = set(tokenize(query))
    if not q_terms:
        return 0.0
    return len(q_terms & set(tokenize(text))) / len(q_terms)
