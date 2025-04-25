import numpy as np
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from openai import OpenAI
import ast


load_dotenv()
openai=OpenAI()

def load_retriever(index_path = "data", embedding_model=None):
    
    if embedding_model is None:
        embedding_model = OpenAIEmbeddings()

    faiss_store = FAISS.load_local(
        index_path,
        embeddings=embedding_model,
        allow_dangerous_deserialization=True
    )
    return faiss_store


def cosine_similarity(a, b):

    a_np, b_np = np.array(a), np.array(b)
    if np.linalg.norm(a_np) == 0 or np.linalg.norm(b_np) == 0:
        return 0.0
    return float(np.dot(a_np, b_np) / (np.linalg.norm(a_np) * np.linalg.norm(b_np)))


def get_relevant_chunks(query, chat_history= "", k= 10, index_path= "data"):
    """
    Retrieves chunks based on content_id, cosine similarity on content_headline, or vector similarity.
    Priority:
      1) Exact content_id
      2) Cosine similarity between query embedding and content_headline embeddings
      3) Default FAISS vector similarity search.
    """
    faiss_store = load_retriever(index_path)
    embed_model = faiss_store.embedding_function


    SYSTEM_PROMPT = """\
    You are an ID extractor. Given the user’s query plus chat history, find **all** numeric identifiers—whether \
    they appear as id, [id], "id", or just standalone numbers—and return them **exactly** as **strings** \
    in a Python list literal—nothing else, and **do not** wrap that list in quotes.

    **Output format:**  
    - A Python list of string literals: ["id1", "id2", ...]  
    - If no IDs are found, return an empty list: []

    **Few-shot examples:**

    User: What are the IDs for articles 1.6636959 and id 42 in the system?  
    Output: ["1.6636959", "42"]

    User: Please fetch [id]: 100, "id": 200, and also 3.14 somewhere else.  
    Output: ["100", "200", "3.14"]

    User: There is no identifier here.  
    Output: []

    User: Mixed content id123 but also 456 and [id]:789.  
    Output: ["123", "456", "789"]
"""

    response= openai.chat.completions.create(
        model='gpt-4o',
        messages=[{"role":"system", "content":SYSTEM_PROMPT},{"role":"user", "content":query+ "\n" +chat_history}])
    
    id_list=response.choices[0].message.content
    raw = id_list.strip()        

    if (raw.startswith("'") and raw.endswith("'")) or (raw.startswith('"') and raw.endswith('"')):
        raw = raw[1:-1]

    try:
        ids = ast.literal_eval(raw)
    except Exception:
        ids = []

    if id_list is not None:
        try:
            exact_docs = []
            for id in ids:
                exact_docs.extend([doc for doc in faiss_store.docstore._dict.values() if doc.metadata.get("content_id") == id])
            if exact_docs:
                return exact_docs[:]
        except ValueError:
            pass

    query_embedding = embed_model.embed_query(query)
    headline_scores = []
    
    docs = list(faiss_store.docstore._dict.values())
    headlines = [doc.metadata.get("content_headline", "") for doc in docs]
    headline_embeddings = embed_model.embed_documents(headlines)
    
    # Compute similarity
    for doc, head_emb in zip(docs, headline_embeddings):
        if not doc.metadata.get("content_headline"):
            continue
        score = cosine_similarity(query_embedding, head_emb)
        headline_scores.append((score, doc))
        
    # Sort and take top k 
    headline_scores.sort(key=lambda x: x[0], reverse=True)
    top_headline = [doc for score, doc in headline_scores if score > 0.83]
    if top_headline:
        return top_headline[:k]

    return faiss_store.similarity_search(query, k=k)


# if __name__ == "__main__":
   
#     queries = [
#         "can you bring the article which headline is : Newfoundland Growlers hit the ice in preparation for 1st pandemic hockey season"
#     ]
#     for q in queries:
#         print(f"\n>> Query: {q}")
#         docs = get_relevant_chunks(q, k=5)
#         for doc in docs:
#             print("--- Chunk ---")
#             print(doc.page_content.replace("\n", " "))
#             print("Metadata:", doc.metadata)
#             print()
