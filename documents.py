from docx import Document
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
import os

def process_docx(file_path):
    """Processes a .docx file and extracts structured text chunks."""
    try:
        doc = Document(file_path)
        sections = []
        current_section = "General"
        current_subsection = None
        buffer = []

        def is_heading(paragraph):
            """Determines if a paragraph is a heading."""
            if paragraph.style.name.startswith('Heading'):
                return True
            if re.match(r'^\d+\.(\d+\.)*\s', paragraph.text.strip()):
                return True
            if paragraph.text.isupper() and len(paragraph.text.strip()) > 3:
                return True
            header_keywords = ['chapter', 'section', 'part', 'introduction', 'conclusion', 
                               'summary', 'overview', 'appendix']
            if any(paragraph.text.lower().startswith(keyword) for keyword in header_keywords):
                return True
            try:
                if paragraph.runs and paragraph.runs[0].bold:
                    return True
            except:
                pass
            return False

        def get_heading_level(paragraph):
            """Determines the heading level based on styles and numbering."""
            if paragraph.style.name.startswith('Heading'):
                try:
                    return int(paragraph.style.name.replace('Heading ', ''))
                except:
                    return 1
            text = paragraph.text.strip()
            if re.match(r'^\d+\.\s', text): return 1
            if re.match(r'^\d+\.\d+\.\s', text): return 2
            if re.match(r'^\d+\.\d+\.\d+\.\s', text): return 3
            return 1

        def flush_buffer():
            """Saves collected text to sections before moving to a new section."""
            if buffer:
                sections.append({"text": "\n".join(buffer), "section": current_section, "sub_section": current_subsection})
                buffer.clear()

        for paragraph in doc.paragraphs:
            text = paragraph.text.strip()
            if not text:
                continue

            if is_heading(paragraph):
                flush_buffer()
                level = get_heading_level(paragraph)
                if level == 1:
                    current_section = text
                    current_subsection = None
                elif level == 2:
                    current_subsection = text
                else:
                    current_subsection = f"{current_subsection} - {text}" if current_subsection else text
            else:
                buffer.append(text)

        flush_buffer()
        return sections

    except Exception as e:
        print(f"Error processing .docx file: {e}")
        return []

def process_txt(file_path):
    """Reads a .txt file and treats it as a single section."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read().strip()
        return [{"text": text, "section": "General", "sub_section": None}]
    except Exception as e:
        print(f"Error processing .txt file: {e}")
        return []

def process_pdf(file_path):
    """Extracts text from a PDF file."""
    try:
        doc = fitz.open(file_path)
        text = "\n".join([page.get_text("text") for page in doc])
        return [{"text": text, "section": "General", "sub_section": None}]
    except Exception as e:
        print(f"Error processing .pdf file: {e}")
        return []

def process_document(file_path, chunk_size=1000, chunk_overlap=100):
    """Detects file type and processes accordingly, returning structured chunks."""
    ext = os.path.splitext(file_path)[-1].lower()
    
    if ext == ".docx":
        sections = process_docx(file_path)
    elif ext == ".txt":
        sections = process_txt(file_path)
    elif ext == ".pdf":
        sections = process_pdf(file_path)
    else:
        print("Unsupported file format.")
        return []

    # Split text into chunks while preserving metadata
    structured_chunks = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    for section in sections:
        if not section["text"].strip():
            continue

        chunks = text_splitter.split_text(section["text"])
        for chunk in chunks:
            structured_chunks.append({
                "text": chunk,
                "section": section["section"],
                "sub_section": section["sub_section"]
            })

    return structured_chunks
