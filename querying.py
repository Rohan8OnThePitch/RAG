import os
import torch
import logging
import re
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

# Configure Logging
logging.basicConfig(level=logging.INFO)

# Load Qdrant Configuration from Environment
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))

# Initialize Qdrant Client
qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

# Load Sentence Transformer for Query Embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load GPT-2 from Hugging Face
GPT2_MODEL_NAME = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(GPT2_MODEL_NAME)
gpt2_model = AutoModelForCausalLM.from_pretrained(
    GPT2_MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)

def validate_context_relevance(context, query, threshold=0.1):
    """
    Validates if the context is relevant to the query using semantic similarity.
    Returns True if context is relevant, False otherwise.
    """
    if not context.strip():
        return False
        
    context_chunks = [context[i:i+512] for i in range(0, len(context), 512)]
    
    query_embedding = embedding_model.encode(query)
    max_similarity = 0
    
    for chunk in context_chunks:
        if chunk.strip():
            chunk_embedding = embedding_model.encode(chunk)
            similarity = torch.nn.functional.cosine_similarity(
                torch.tensor(query_embedding).unsqueeze(0),
                torch.tensor(chunk_embedding).unsqueeze(0)
            ).item()
            max_similarity = max(max_similarity, similarity)
    
    return max_similarity >= threshold

def extract_answer_from_response(response):
    """
    Extracts and validates the answer from the model's response.
    Returns None if the answer doesn't meet quality criteria.
    """
    answer_parts = response.split("Answer:")
    answer = answer_parts[-1].strip() if len(answer_parts) > 1 else response.strip()
    
    if len(answer) < 2 or len(answer) > 500:
        return None
        
    hallucination_patterns = [
        r"I don't have enough information",
        r"I cannot provide",
        r"no information available",
        r"insufficient data"
    ]
    
    if any(re.search(pattern, answer, re.IGNORECASE) for pattern in hallucination_patterns):
        return None
        
    return answer

def generate_answer(query, context):
    """Generates a concise, focused response using GPT-2 based on the retrieved context."""
    print("HEllo")
    if not context.strip():
        return "No information found in the database."

    prompt = f"""
    Context: {context}
    Question: {query}
    Answer: Let me answer based on the context provided."""

    inputs = tokenizer(prompt, return_tensors="pt").to(gpt2_model.device)
    
    outputs = gpt2_model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.3,
        top_p=0.95,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
        do_sample=True
    )
    
   # print(f"Raw Output Tokens: {outputs}")  # Debug: Print the raw output tokens
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    #print(f"Decoded Answer: {response}")  # Debug: Print the decoded answer before cleaning
    
    response = re.sub(r"[\w\s]*\(\s*[^)]+\s*\)", "", response)
    print(f"Final Answer: {response}")  # Debug: Print the final cleaned answer
    
    answer = extract_answer_from_response(response)
    return answer.strip() if answer else "No answer generated"

def query_documents(collection_name, user_query, top_k=5, score_threshold=0.3):
    """Queries Qdrant, retrieves matching documents, and generates an answer using GPT-2."""
    print("hello")
    try:
        logging.info(f"🔍 Processing query: {user_query}")

        query_vector = embedding_model.encode(user_query).tolist()

        search_results = qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=top_k,
            score_threshold=score_threshold,
            with_payload=True
        )

        if not search_results:
            return {
                "answer": "No relevant information found in the database.",
                "chunks": []
            }

        filtered_results = [
            {
                "id": res.id,
                "score": res.score,
                "text": res.payload.get("text", ""),
                "section": res.payload.get("section", ""),
                "sub_section": res.payload.get("sub_section", "")
            }
            for res in search_results if "text" in res.payload
        ]
        '''if filtered_results:
            for i, result in enumerate(filtered_results):
                print(f"\nResult {i + 1} (Score: {result['score']:.3f}):")
                print(f"ID: {result['id']}")
                print(f"Text: {result['text']}")
                print(f"Document ID: {result['document_id']}")
                print(f"Chunk Index: {result['chunk_index']}")
                print(f"Full Payload: {result['payload']}")  # Print the entire payload
        else:
            print("No relevant results found.")'''

        filtered_results.sort(key=lambda x: (x["section"], x["sub_section"], -x["score"]))
        
        # Print the relevant payload for each result
       
        
        combined_context = " ".join([result["text"] for result in filtered_results[:top_k]])
        if not combined_context.strip():
            return {"answer": "No relevant information found.", "chunks": []}

        generated_answer = generate_answer(user_query, combined_context)

        return {"answer": generated_answer, "chunks": filtered_results[:top_k]}
    except Exception as e:
        logging.error(f"Error during query: {e}")
        return {"error": str(e)}
