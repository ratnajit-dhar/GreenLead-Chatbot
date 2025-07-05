from fastapi import FastAPI, Request
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
import pinecone
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from fastapi.middleware.cors import CORSMiddleware
# Load environment variables
load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, restrict this to your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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


class QueryRequest(BaseModel):
    query: str

@app.post("/chat")
async def chat_endpoint(data: QueryRequest):
    input_prompt = (
        "You are a friendly chatbot that is related to an organization called Green Lead. "
        "They work for better environment. You need to answer the following question as friendly as possible and in details: "
        + data.query
    )

    result = qa_chain({"query": input_prompt})
    answer = result["result"]

    return {
        "answer": answer,
    }