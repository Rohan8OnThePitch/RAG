from documents import process_docx
doc_path = "r.docx"
chunks = process_docx(doc_path)
for chunk in chunks:
    print(f"Section: {chunk['section']}")
    print(f"Subsection: {chunk['sub_section']}")
    print(f"Content: {chunk['text'][:100]}...")  # First 100 chars
    print("---")