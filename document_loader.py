from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

def load_and_chunk_documents(file_path, chunk_size=1000, chunk_overlap=100):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File '{file_path}' not found. Please check the file path.")

    if file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith('.docx'):
        loader = Docx2txtLoader(file_path)
    else:
        loader = TextLoader(file_path)

    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(documents)
    return chunks

# Example usage
if __name__ == '__main__':
    file_path = 'Tetst.docx'  # Ensure the file path and name are correct
    chunks = load_and_chunk_documents(file_path)
    print(chunks)
