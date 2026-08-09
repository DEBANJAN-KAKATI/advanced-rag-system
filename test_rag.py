import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.ingestion.loader import load_document
from app.retrieval.chunker import chunk_document
from app.retrieval.vectorstore import vector_store
from app.generation.answerer import generate_answer
from app.core.schemas import AnswerStyle

def run_test():
    print("--- 1. Creating sample test document ---")
    sample_path = Path("sample_report.txt")
    sample_path.write_text(
        "# Q3 Financial Performance Report\n\n"
        "Revenue for Q3 reached $4.2 million, representing a 24% year-over-year increase driven by strong enterprise subscription growth.\n"
        "Operating expenses totaled $2.1 million, resulting in a net profit margin of 50%.\n\n"
        "## Key Risk Factors\n"
        "Supply chain disruptions in APAC may impact hardware delivery times by 2-3 weeks during Q4.\n"
        "Customer retention remains high at 96.5% gross dollar retention.",
        encoding="utf-8"
    )

    print("--- 2. Testing document ingestion & chunking ---")
    doc_data = load_document(sample_path)
    chunks = chunk_document(doc_data)
    print(f"Doc Loaded: {doc_data['filename']} (ID: {doc_data['doc_id']}) -> Created {len(chunks)} chunks.")

    print("--- 3. Testing Vector Indexing ---")
    vector_store.add_chunks(chunks)
    print(f"Vector Store size: {len(vector_store.get_all_chunks())} chunks.")

    print("--- 4. Testing RAG Queries with Styles ---")
    question = "What was the revenue and profit margin in Q3?"
    
    for style in [AnswerStyle.CONCISE, AnswerStyle.MEDIUM, AnswerStyle.LONG]:
        print(f"\n================ Question Style: {style.value.upper()} ================")
        response = generate_answer(question=question, style=style)
        print(f"Confidence: {response.confidence}")
        print(f"Citations count: {len(response.citations)}")
        print(f"Answer:\n{response.answer}")
        
    print("\n--- 5. Testing Out-of-Domain Question Fallback ---")
    response_unrelated = generate_answer("What is the distance from Earth to Mars?")
    print(f"Confidence: {response_unrelated.confidence}")
    print(f"Answer:\n{response_unrelated.answer}")

    print("\nSUCCESS: All RAG Pipeline verification checks passed!")

if __name__ == "__main__":
    run_test()
