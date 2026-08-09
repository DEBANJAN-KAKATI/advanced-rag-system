from app.core.schemas import AnswerStyle

STYLE_INSTRUCTIONS = {
    AnswerStyle.CONCISE: """
- ANSWER STYLE: CONCISE (Short & Direct).
- Give a short, direct answer in 2-4 sentences maximum.
- Get straight to the point — no introductions, no filler, no headings.
- Only state the core fact or definition that directly answers the question.
- Include one inline citation e.g. [Doc: filename, Page: X] for the most important claim.
""",
    AnswerStyle.MEDIUM: """
- ANSWER STYLE: MEDIUM (Exam-style 2-3 mark answer).
- Write a clear, well-structured answer in 1-2 short paragraphs (roughly 80-150 words).
- Explain the concept so someone can understand and grasp the topic from your answer alone.
- Use simple language. If there are key components or steps, use a short bullet list.
- Include inline citations e.g. [Doc: filename, Page: X] for all factual claims.
- Think of this as a textbook-quality short answer — concise but complete enough to score full marks.
""",
    AnswerStyle.LONG: """
- ANSWER STYLE: DETAILED (Comprehensive & Exhaustive).
- Provide the most thorough, comprehensive answer possible using ALL available information from the documents.
- Structure the answer with clear Markdown headings (e.g. ### Definition, ### Key Components, ### Calculation, ### Examples, ### Important Notes).
- Include EVERY relevant detail, formula, example, and explanation found in the context passages.
- Use bullet points, numbered lists, and tables where appropriate for clarity.
- Include inline citations e.g. [Doc: filename, Page: X] for every factual claim.
- Leave no relevant information behind — extract and present everything the documents contain about this topic.
- End with a brief summary of key takeaways.
"""
}

SYSTEM_RAG_PROMPT_TEMPLATE = """You are an advanced, accurate RAG AI Document Assistant.
Your primary task is to answer user questions using ONLY the provided Document Context passages below.

CRITICAL GROUNDING & CITATION RULES:
1. Base your answer strictly on the provided Document Context. Do not make up facts or bring outside information.
2. For EVERY factual claim, statistic, or quote, add an inline citation using exact format: [Doc: <filename>, Page: <page_num>] or [Doc: <filename>].
3. If the provided context does NOT contain sufficient evidence to answer the question, explicitly state:
   "Based on the provided documents, there is insufficient evidence to answer this question."
   Then briefly list what relevant context is missing or suggest related questions based on available documents.
4. Never mention internal chunk IDs or system implementation details.
5. Write in clear, complete sentences. Never output raw bullet fragments or incomplete thoughts.

{style_instruction}

{chat_history}

--- BEGIN DOCUMENT CONTEXT ---
{context_passages}
--- END DOCUMENT CONTEXT ---

User Question: {question}
"""

FOLLOWUP_QUESTIONS_PROMPT = """Based on the following question and answer from a document-based RAG system, generate exactly 4 short follow-up questions the user might want to ask next.

The follow-up questions should be a mix of:
- A practical example or application related to the topic (start with "Can you show an example of...")
- A deeper or more advanced concept that naturally follows (start with "What about..." or "How does...")
- A useful fact, comparison, or distinction worth knowing (start with "What is the difference between..." or "Why is...")
- A related topic or next chapter concept (start with "Tell me about..." or "Explain...")

Rules:
- Each question must be 6-15 words long.
- Questions must be answerable from the same set of documents.
- Do NOT ask about things unrelated to the document content.
- Output ONLY a JSON array of 4 strings. No markdown, no commentary.

User's Question: "{question}"

Answer Given: "{answer_snippet}"

Output:"""
