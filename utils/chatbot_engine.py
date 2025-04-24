import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory
from openai import OpenAI

from utils.retriever import get_relevant_chunks
from utils.crawler_guidelines import crawler
from utils.process_data import chunker, build_faiss_index


#data processing
if not os.path.exists('data/guidelines.json'):
    crawler()
    chunker()
    build_faiss_index()


load_dotenv()
openai = OpenAI()


POLICY_QA_PROMPT = PromptTemplate(
    input_variables=["question", "context", "chat_history"],
    template="""
    You are CBC's expert editorial assistant chatbot. You will receive questions from the user and must respond \
    clearly and informatively, using markdown formatting when helpful to improve clarity and readability.\


    Most questions will fall into one of the following categories:

    - Inquiries about CBC’s editorial policies, where you must use the provided/retrieved editorial guidelines to answer the query.
    - Requests to retrieve an article by a specific content ID or headline or one part of article (a few sentences)
    - Requests to generate an SEO-optimized headline or a social media summary (e.g., for Twitter) based on a given article.

    Your responses should always adopt the tone and writing style most appropriate for the user’s query.

    To strengthen your answers, you must cite internal documents when relevant.

    As you access to `content_id` and `url` as metadata in the retrieved content and you have to use them to cite using the following format.\
    - For article references (retrieved data coming from article), use the format: [source: article id content_id] 
    - For editorial guideline references (retrieved data coming from guideline), use the format: [source: url]  
    
    Do not fabricate citations — only cite sources when you have access to the actual content ID or url as metadata. \
    exact id and url. If not available, omit the citation/sources.\

    If the user asks something like **"What is the article [id]?"**, answer the question and then follow up by asking:
    **"Would you like to see the full content of the article?"**
    If the user says yes, then provide the complete body of the article.

    Use the editorial guidelines, retrieved article content, and chat history to generate accurate, well-grounded, and context-aware responses.

    Editorial guidelines or Article information:
    {context}

    Chat history:
    {chat_history}

    Question: {question}
    """
)

chat_history = ChatMessageHistory()

def clear_history():
    chat_history.messages.clear()
    
def chat(query, placeholder=None, model = "gpt-4o"):

    history_str = "\n".join([f"{m.type.capitalize()}: {m.content}" for m in chat_history.messages])
    response = ""
    
    chunks = get_relevant_chunks(query, chat_history=history_str, k=20)
    
    prompt = POLICY_QA_PROMPT.format_prompt(
        question=query,
        context=chunks,
        chat_history=history_str
    ).to_string()
    
    stream = openai.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": prompt}],
        stream=True
    )
    
    for chunk in stream:
        token = chunk.choices[0].delta.content or ''
        response += token
        yield token
        
        if placeholder:
            placeholder.text(response)  
    
    
    chat_history.add_user_message(query)
    chat_history.add_ai_message(response)
    
    return response
    

# if __name__ == "__main__":
#     print("CBC Editorial Chatbot — type 'quit' to exit")
#     while True:
#         user = input("\nYou: ")
#         if user.lower() in ("exit", "quit"):
#             break
#         response = chat(user)
#         print("Bot:",)
