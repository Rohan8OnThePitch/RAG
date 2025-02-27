import uuid
import logging
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance
from sentence_transformers import SentenceTransformer
from qdrant_client.models import PointStruct

# Initialize Qdrant client and model
qdrant_client = QdrantClient(host="localhost", port=6333)
model = SentenceTransformer("all-MiniLM-L6-v2")

# Set up logging
logging.basicConfig(level=logging.INFO)

COLLECTION_NAME = "document_chunks"

def create_collection_if_not_exists(collection_name):
    """
    Creates a Qdrant collection if it doesn't already exist.
    """
    try:
        collections_response = qdrant_client.get_collections()
        existing_collections = [col.name for col in collections_response.collections]

        if collection_name not in existing_collections:
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=384,  # Embedding dimension of all-MiniLM-L6-v2
                    distance=Distance.COSINE  # Cosine distance for similarity search
                )
            )
            logging.info(f"Collection '{collection_name}' created.")
        else:
            logging.info(f"Collection '{collection_name}' already exists.")
    except Exception as e:
        logging.error(f"Error creating collection '{collection_name}': {e}")
        raise

def index_document(collection_name, document_id, chunks, batch_size=50):
    """
    Indexes document chunks into the specified Qdrant collection.
    """
    try:
        logging.info(f"Starting indexing for document: {document_id}")
        create_collection_if_not_exists(collection_name)

        if not chunks:
            logging.warning("No chunks provided for indexing.")
            return {"status": "error", "message": "No chunks found"}

        logging.info(f"Indexing {len(chunks)} chunks.")

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            logging.info(f"Processing batch {i//batch_size + 1}, size: {len(batch)}")

            # Get embeddings only for text content in chunks
            embeddings = model.encode([chunk.page_content for chunk in batch], convert_to_tensor=False).tolist()

            points = []
            for idx, (chunk, embedding) in enumerate(zip(batch, embeddings)):
                chunk_id = str(uuid.uuid4())
                payload = {
                    "document_id": document_id,
                    "text": chunk.page_content,
                    "metadata": chunk.metadata,
                    "chunk_index": i + idx
                }
                points.append(
                    PointStruct(
                        id=chunk_id,
                        vector=embedding,
                        payload=payload
                    )
                )

            qdrant_client.upsert(
                collection_name=collection_name,
                points=points,
                wait=True
            )

            collection_info = qdrant_client.get_collection(collection_name)
            logging.info(f"After batch {i//batch_size + 1}: Collection has {collection_info.points_count} points")

        return {"status": "success", "chunks": len(chunks)}

    except Exception as e:
        logging.error(f"Error indexing document '{document_id}': {e}")
        return {"status": "error", "message": str(e)}

def query_qdrant(collection_name, query_text, top_k=5):
    """
    Queries the Qdrant collection and returns relevant results.
    """
    try:
        query_vector = model.encode([query_text], convert_to_tensor=False)[0].tolist()
        search_results = qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=top_k,
            score_threshold=0.3,
            #with_payload=True
        )

        results = [{"score": hit.score, "text": hit.payload['text'], "metadata": hit.payload.get("metadata", {})} for hit in search_results]
        logging.info(f"Query returned {len(results)} results.")
        return results

    except Exception as e:
        logging.error(f"Error querying collection '{collection_name}': {e}")
        return []

# Example usage
if __name__ == "__main__":
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

    # Load and chunk example document
    file_path = 'Tetst.docx'
    chunks = load_and_chunk_documents(file_path)
    print(f"Loaded {len(chunks)} chunks.")

    # Indexing example
    #index_result = index_document(COLLECTION_NAME, "test_doc_1", chunks)
    #print("Indexing Result:", index_result)

    # Querying example
    query_result = query_qdrant(COLLECTION_NAME, "Sky")
    print("Query Result:", query_result)
