import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
import streamlit.components.v1 as components
import pinecone
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
# Load environment variables
load_dotenv()


pinecone_api = os.getenv('pinecone_api')
os.environ["PINECONE_API_KEY"] = pinecone_api  # ✅ Add this line
pc = Pinecone(api_key=pinecone_api)
google_api_key = os.getenv('GOOGLE_API_KEY')

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=google_api_key
)

# Load vectorstore from Pinecone index
vectorstore = PineconeVectorStore.from_existing_index(
    index_name="greenlead-global",
    embedding=embeddings
)
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)
# Load model
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.2,
    # google_api_key=os.getenv("GOOGLE_API_KEY")
    google_api_key = google_api_key
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# Streamlit Title
st.title("GreenLead Chatbot")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Build the chat HTML with styles and icons
chat_html = """
<style>
    #chat-box {
        height: 300px;
        overflow-y: auto;
        padding: 10px;
        border: 1px solid #ddd;
        border-radius: 8px;
        background-color: #f9f9f9;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .message {
        max-width: 70%;
        margin-bottom: 12px;
        padding: 10px 15px;
        border-radius: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        line-height: 1.4;
        font-size: 14px;
        word-wrap: break-word;
        white-space: pre-wrap;
        display: flex;
        align-items: center;
    }
    .user-msg {
        background-color: #e8f0fe;
        color: #1a73e8;
        margin-left: auto;
        border-bottom-right-radius: 2px;
        justify-content: flex-end;
    }
    .bot-msg {
        background-color: #d7f2d8;
        color: #34a853;
        margin-right: auto;
        border-bottom-left-radius: 2px;
        justify-content: flex-start;
    }
    .icon {
        margin: 0 8px;
        font-size: 18px;
        user-select: none;
    }
</style>

<div id='chat-box'>
"""

for speaker, message in st.session_state.chat_history:
    if speaker == "You":
        chat_html += f"""
        <div class='message user-msg'>
            <span>{message}</span>
            <span class='icon' title='User'>👤</span>
        </div>"""
    else:
        chat_html += f"""
        <div class='message bot-msg'>
            <span class='icon' title='Bot'>🤖</span>
            <span>{message}</span>
        </div>"""

chat_html += """
</div>
<script>
    var chatBox = document.getElementById('chat-box');
    chatBox.scrollTop = chatBox.scrollHeight;
</script>
"""

# Render chat container
components.html(chat_html, height=320)

# Chat input form
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("Type your message...", key="input")
    input_prompt = "You are a friendly chatbot that is related to a organization Green Lead. They work for better environment. You need to answer to the following question as friendly as possible and in details: " + user_input

    submit = st.form_submit_button("Send")

if submit and user_input:
    with st.spinner("Thinking..."):
        result = qa_chain({"query": input_prompt})
        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("Bot", result['result']))

        # Write retrieved documents to a .txt file
        source_docs = result.get("source_documents", [])
        with open("retrieved_documents.txt", "a", encoding="utf-8") as f:
            f.write(f"\n--- Query: {user_input} ---\n")
            for i, doc in enumerate(source_docs):
                f.write(f"\nDocument {i+1}:\n{doc.page_content}\n")
    st.rerun()  # Refresh immediately to show update
