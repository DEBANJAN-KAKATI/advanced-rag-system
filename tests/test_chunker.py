from app.retrieval.chunker import chunk_document, chunk_search_text, detect_heading


def _sentences(start, end):
    return [f"Sentence {i} describes the quarterly revenue in some detail." for i in range(start, end)]


def test_long_unbroken_page_is_split_within_size(make_doc):
    # pypdf output often has no blank lines: one "paragraph" per page.
    doc = make_doc(text="\n".join(_sentences(0, 80)))
    chunks = chunk_document(doc, chunk_size=600, chunk_overlap=100)
    assert len(chunks) > 5
    assert all(c["char_count"] <= 600 for c in chunks)


def test_chunks_end_on_sentence_boundaries_and_overlap(make_doc):
    chunks = chunk_document(make_doc(text=" ".join(_sentences(0, 40))), chunk_size=500, chunk_overlap=120)
    for c in chunks:
        assert c["text"].endswith(".")
    # The start of each chunk repeats the tail of the previous one.
    first_sentence_of_second = chunks[1]["text"].split(".")[0]
    assert first_sentence_of_second in chunks[0]["text"]


def test_sections_tracked_and_carried_across_pages(make_doc):
    doc = make_doc(pages=[
        {"page": 1, "text": "# Annual Report\n\n" + " ".join(_sentences(0, 4)) + "\n\n## Risks\n\n"
                            + " ".join(_sentences(4, 6))},
        {"page": 2, "text": " ".join(_sentences(6, 8))},
        {"page": 3, "text": "## Outlook\n\nGrowth expected."},
    ])
    chunks = chunk_document(doc, chunk_size=600, chunk_overlap=0)
    sections = [c["section"] for c in chunks]
    assert "Annual Report > Risks" in sections
    risks = next(c for c in chunks if c["section"] == "Annual Report > Risks")
    assert risks["page"] == 1 and risks["page_end"] == 2
    assert chunks[-1]["section"] == "Annual Report > Outlook"
    assert chunks[-1]["text"].startswith("## Outlook")


def test_huge_word_sequence_is_split_by_words(make_doc):
    chunks = chunk_document(make_doc(text=" ".join(["token"] * 2000)), chunk_size=300, chunk_overlap=0)
    assert all(c["char_count"] <= 300 for c in chunks)
    assert sum(c["text"].count("token") for c in chunks) == 2000


def test_heading_detection():
    assert detect_heading("## Key Risks") == (2, "Key Risks")
    assert detect_heading("KEY RISK FACTORS") == (2, "KEY RISK FACTORS")
    assert detect_heading("This is a normal sentence.") is None
    assert detect_heading("USA.") is None


def test_search_text_includes_document_and_section(make_doc):
    chunk = chunk_document(make_doc(text="## Revenue\n\nIt grew 24%."))[0]
    assert chunk_search_text(chunk).startswith("report.txt | Revenue")


def test_tiny_sections_merge_instead_of_creating_fragments(make_doc):
    doc = make_doc(text="# Title\n\nShort intro.\n\n## Details\n\nShort details.")
    chunks = chunk_document(doc, chunk_size=1200, chunk_overlap=0)
    assert len(chunks) == 1 and "## Details" in chunks[0]["text"]
