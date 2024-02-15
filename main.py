##streamlit code 

import streamlit as st
from langchain.chains import ChatVectorDBChain
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings

OPENAI_API_KEY = st.secrets.OPENAI_API_KEY

st.title("Healthcare Assistant")
st.subheader(
    "Ask any healthcare related questions",
)

db = None

chat_history = []

llm = ChatOpenAI(
        openai_api_key = OPENAI_API_KEY,
        model_name='gpt-3.5-turbo',
        temperature=0.0
    )

def load_faiss_embeddings(path):
    global db
    embeddings = OpenAIEmbeddings(
        openai_api_key=OPENAI_API_KEY
        )
    db = FAISS.load_local('db_faiss', embeddings)

    print("db loaded")

def query_faiss(query):
    global db
    ans = db.similarity_search(query)
    print(ans)
    # return ans.page_content

def query_from_doc(text):
    global chat_history
    qa = ChatVectorDBChain.from_llm(llm, db)
    prompt_template = "The user will ask you with problems related to healthcare, you are given information and using that answer the users queries"
    result = qa({"question":prompt_template+ text, "chat_history": chat_history})
    print(result["answer"])
    chat_history = [(text, result["answer"])]
    # return result["answer"]


if 1:

    load_faiss_embeddings("db_faiss")

    if "messages" not in st.session_state.keys(): # Initialize the chat messages history
        st.session_state.messages = [{"role": "assistant", "content": "Ask me anything about healthcare"}]

    if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

    for message in st.session_state.messages: # Display the prior chat messages
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # If last message is not from assistant, generate a new response
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                msg = query_from_doc(prompt)
                st.write(msg)
                message = {"role": "assistant", "content": msg}
                st.session_state.messages.append(message) 
