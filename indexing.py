import uuid
import logging
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance
from sentence_transformers import SentenceTransformer
from qdrant_client.models import PointStruct, VectorParams, Distance

# Initialize Qdrant client and model
qdrant_client = QdrantClient(host="localhost", port=6333)
model = SentenceTransformer("all-MiniLM-L6-v2")

# Set up logging
logging.basicConfig(level=logging.INFO)

def create_collection_if_not_exists(collection_name):
    """
    Creates a Qdrant collection if it doesn't already exist.
    
    Args:
        collection_name (str): Name of the collection.
    """
    try:
        # Check if the collection exists
        collections_response = qdrant_client.get_collections()
        existing_collections = [col.name for col in collections_response.collections]

        if collection_name not in existing_collections:
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=384,  # Ensure this matches your embedding dimensions
                    distance=Distance.COSINE  # Use cosine distance for similarity
                )
            )
            logging.info(f"Collection '{collection_name}' created.")
        else:
            logging.info(f"Collection '{collection_name}' already exists.")
    except Exception as e:
        logging.error(f"Error creating collection '{collection_name}': {e}")
        raise
    """
def chunk_text(text, chunk_size=1000, chunk_overlap=100):
    """
   # Splits text into chunks using LangChain's text splitter.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    chunks = text_splitter.split_text(text)
    return [{"text": chunk} for chunk in chunks]
"""
def index_document(collection_name, document_id, chunks, batch_size=50):
    try:
        print(f"Starting indexing for document: {document_id}")

        create_collection_if_not_exists(collection_name)

        if not chunks:
            logging.warning("No chunks provided for indexing.")
            return {"status": "error", "message": "No chunks found"}

        print(f"Using {len(chunks)} preprocessed chunks from documents.py")

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            print(f"Processing batch {i//batch_size + 1}, size: {len(batch)}")

            # Get embeddings only for text
            embeddings = model.encode([c["text"] for c in batch], convert_to_tensor=False).tolist()
            print(f"Created embeddings for batch. First embedding size: {len(embeddings[0])}")

            points = []
            for idx, (chunk, embedding) in enumerate(zip(batch, embeddings)):
                chunk_id = str(uuid.uuid4())
                payload = {
                    "document_id": document_id,
                    "text": chunk["text"],
                    "section": chunk.get("section", "General"),
                    "sub_section": chunk.get("sub_section"),
                    "chunk_index": i + idx
                }
                points.append(
                    PointStruct(
                    id=chunk_id,
                    vector=embedding,
                    payload=payload
                    ))

            qdrant_client.upsert(
                collection_name=collection_name,
                points=points,
                wait=True
            )

            collection_info = qdrant_client.get_collection(collection_name)
            print(f"After batch {i//batch_size + 1}: Collection has {collection_info.points_count} points")

        return {"status": "success", "chunks": len(chunks)}

    except Exception as e:
        logging.error(f"Error indexing document '{document_id}': {e}")
        print(f"Detailed indexing error: {str(e)}")
        return {"status": "error", "message": str(e)}
