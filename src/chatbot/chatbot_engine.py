import os
from dotenv import load_dotenv
from langchain_community.chat_message_histories import ChatMessageHistory
from openai import OpenAI

from src.chatbot.retriever import get_relevant_chunks
from src.data_processing.crawler_guidelines import crawler
from src.data_processing.process_data import chunker, build_faiss_index
from src.chatbot.prompts_config import (
    ARTICLE_PROMPT,
    HEADLINE_PROMPT,
    SUMMARY_PROMPT,
    SOCIAL_MEDIA_PROMPT,
    POLICY_QA_PROMPT
)

#data processing
if not os.path.exists('data/guidelines.json'):
    crawler()
    chunker()
    build_faiss_index()


load_dotenv()
openai = OpenAI()


def find_user_intent(query, chat_history):
    
    prompt = f"""
    You are a CBC editorial assistant chatbot. Your task is to identify the user's intent based on their query. \
    The user may ask about editorial policies, request an article by content ID or headline, or ask for an SEO-optimized \
    headline or social media summary based on a given article. Your response should be a single word indicating the intent: \
    The exact intents are as follows and you should return the exact string as it is:
    - "article" for requests to retrieve an article/ news by content ID or headline
    - "headline" for requests to generate an SEO-optimized headline
    - "summary" for requests to summarize an article
    - "social_media" for requests to generate a social media summary
    - "guideline" for requests that require editorial guidelines and policies
    - "greet" for general greetings or small talk
    """
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": prompt}, {"role": "user", "content": query}])
    id_list=response.choices[0].message.content
    print("User intent:", id_list)
    return id_list  


chat_history = ChatMessageHistory()

def clear_history():
    chat_history.messages.clear()
    
def chat(query, placeholder=None, model = "gpt-4o"):

    history_str = "\n".join([f"{m.type.capitalize()}: {m.content}" for m in chat_history.messages])
    response = ""
    
    intent = find_user_intent(query, chat_history=history_str)
    
    chunks = get_relevant_chunks(query, chat_history=history_str, k=20)
    if intent == "article":
        prompt = ARTICLE_PROMPT
    elif intent == "headline":
        prompt = HEADLINE_PROMPT
    elif intent == "summary":
        prompt = SUMMARY_PROMPT
    elif intent == "social_media":
        prompt = SOCIAL_MEDIA_PROMPT
    elif intent == "guideline":
        prompt = POLICY_QA_PROMPT
    else:
        prompt = POLICY_QA_PROMPT          

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
