import os
import logging
import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer
import qdrant_helper

load_dotenv()

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Validate HF_TOKEN
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise EnvironmentError("HF_TOKEN environment variable is not set")

# Load Sentence Transformer for Query Embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Specify the model name
model_name = "Intel/dynamic_tinybert"

# Load the tokenizer associated with the specified model
tokenizer = AutoTokenizer.from_pretrained(model_name, padding=True, truncation=True, max_length=512)

# Define a question-answering pipeline using the model and tokenizer
question_answerer = pipeline(
    "question-answering", 
    model=model_name, 
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1,  # Use GPU if available
    return_tensors='pt'
)

def generate_answer(query, context, max_length=256, temperature=1.0):
    """
    Generate an answer for a query based on the provided context using the LLM.
    
    Args:
        query (str): The user's question
        context (str): The context text to use for answering
        
    Returns:
        str: The generated answer
    """
    logger.info("Generating answer using TinyBERT")
    if not context.strip():
        logger.warning("Empty context provided to generate_answer")
        return "No information found in the database."

    try:
        result = question_answerer(
            question=query, 
            context=context,
            max_length=max_length,
            temperature=temperature
        )
        return result["answer"]
    except Exception as e:
        logger.error(f"Error generating answer: {str(e)}")
        return f"Error generating answer: {str(e)}"


def answer_query(user_query, top_k=5):
    """
    End-to-end function to answer a user query using RAG.
    
    Args:
        user_query (str): User's question
        top_k (int): Maximum number of chunks to retrieve
        
    Returns:
        dict: Dictionary containing the answer and relevant chunks
    """
    try:
        logger.info(f"🔍 Processing query: {user_query}")
        
        # Use the query_qdrant function to get relevant contexts
        context_items = qdrant_helper.query_qdrant(
            collection_name=qdrant_helper.COLLECTION_NAME,
            query_text=user_query, 
            top_k=top_k
        )
        
        logger.debug(f"Retrieved {len(context_items)} contexts from Qdrant")

        if not context_items:
            logger.warning("No contexts found")
            return {
                "answer": "No relevant information found in the database.",
                "chunks": []
            }

        # Extract text from the retrieved dictionaries
        contexts = [item["text"] for item in context_items]
        
        # Combine text from all results
        combined_context = " ".join(contexts)
        if not combined_context.strip():
            logger.warning("No valid context after combining results")
            return {"answer": "No relevant information found.", "chunks": []}

        # Generate answer from combined context
        generated_answer = generate_answer(user_query, combined_context)
        logger.info("Answer generated successfully")

        return {"answer": generated_answer, "chunks": context_items}

    except Exception as e:
        logger.error(f"Error during query: {str(e)}", exc_info=True)
        return {"answer": f"An error occurred: {str(e)}", "chunks": []}


if __name__ == "__main__":
    query = "What kind of support can AI-driven chatbots provide for mental health?"
    response = answer_query(query)
    print(response.get("answer", "No answer found"))
