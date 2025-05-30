import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

load_dotenv()

# Load vectorstore
vectorstore = FAISS.load_local(
    "vectorstore/faiss_index",
    GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    ),
    allow_dangerous_deserialization=True
)
retriever = vectorstore.as_retriever()

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

st.title("GreenLead Chatbot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("Ask a question about your documents:")

if st.button("Send") and user_input:
    with st.spinner("Thinking..."):
        result = qa_chain({"query": user_input})
        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("Bot", result["result"]))

for speaker, message in st.session_state.chat_history:
    st.markdown(f"**{speaker}:** {message}")