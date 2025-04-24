import streamlit as st
from utils.chatbot_engine import chat, clear_history



st.set_page_config(page_title="CBC Editorial Assistant Chatbot", layout="wide")

st.title("CBC Editorial Assistant Chatbot")
st.markdown("Ask me about CBC editorial policies, get SEO-optimized headlines, or summaries of articles.")


# Initialize session state for chat history
if "history" not in st.session_state:
    st.session_state.history = []
if "input_processed" not in st.session_state:
    st.session_state.input_processed = False  
if "user_input" not in st.session_state:
    st.session_state.user_input = ""
                    
if st.session_state.input_processed:
    st.session_state.user_input = ""
    st.session_state.input_processed = False
         

with st.sidebar:

    if st.button("Clear Chat History"):
        st.session_state.history = []
        clear_history()
        st.session_state.input_processed = False
        st.rerun()
              
    model = st.selectbox(
        "Choose a model:",
        options=["gpt-4o", "gpt-4"],  
        index=0 
    )
           
st.subheader("Chat History")
for entry in st.session_state.history:
    st.markdown(f"**You:** {entry['query']}")
    st.markdown(f"**Bot:** {entry['response']}")
    st.markdown("---")

query = st.text_input("Your question:", key="user_input")
if st.button("Submit", key="send"): 
    if query.strip() and not st.session_state.input_processed:
        with st.spinner("Generating response..."):
            response_placeholder = st.empty()
            response = ""

            for token in chat(query, placeholder=response_placeholder, model=model):
                response += token             
        st.session_state.history.append({"query": query, "response": response})
        st.session_state.input_processed = True
        st.rerun()


