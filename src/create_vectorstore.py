from PyPDF2 import PdfReader
import base64
import os
import logging
from dotenv import load_dotenv
load_dotenv()

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_core.documents import Document

import pinecone
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore


pinecone_api = os.getenv('pinecone_api')
os.environ["PINECONE_API_KEY"] = pinecone_api  # ✅ Add this line
pc = Pinecone(api_key=pinecone_api)


log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logger = logging.getLogger('create_vectorstore')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'create_vectorstore.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def get_pdf_text(pdfs):
    try:
        text = ''
        for pdf in pdfs:
            if not os.path.exists(pdf):
                logger.error(f"PDF File not found: {pdf}")
                raise FileNotFoundError(f"PDF file not found: {pdf}")
            loader = PyPDFLoader(pdf)
            pdf_text = loader.load()
            if not pdf_text:
                logger.warning(f"No Text extracted from {pdf}")
            else:
                text += ' '.join([doc.page_content for doc in pdf_text])
                logger.debug(f"Extracted text from {pdf}. Length: {len(pdf_text)} characters.")
                        
        logger.debug(f"Extracted text from {len(pdfs)} PDFs.")
        return text
    except FileNotFoundError as e:
        logger.error(f"PDF file not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Error extracting text from PDFs: {e}")
        raise

def get_web_text(webpages):
    try:
        text = ''
        for webpage in webpages:
            if not webpage.startswith('http'):
                logger.error(f"Invalid URL: {webpage}")
                raise ValueError(f"Invalid URL: {webpage}")
            web_loader = WebBaseLoader(webpage)
            web_text = web_loader.load()
            if not web_text:
                logger.warning(f"No Text extracted from {webpage}")
            else:
                text = ' '.join([doc.page_content for doc in web_text])
                logger.debug(f"Extracted text from {webpage}. Length: {len(web_text)} characters")
        logger.debug(f"Extracted text from {len(webpages)} webpages.")
        return text
    except ValueError as e:
        logger.error(f"Invalid URL: {e}")
        raise 
    except Exception as e:
        logger.error(f"Error extracting text from webpages: {e}")
        raise
    
                

    

def get_text_chunks(text):
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=256)
        chunks = text_splitter.split_text(text)
        chunks = list(dict.fromkeys(chunks))
        logger.debug(f"Split text into {len(chunks)} chunks.")
        if not chunks:
            logger.warning("No text chunks were created. Check the input text.")
            raise ValueError("No text chunks created from the input text.")
        return chunks
    except Exception as e:
        logger.error(f"Error splitting text into chunks: {e}")
        raise

def get_vectorstore(text_chunks):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model = 'models/embedding-001',
            google_api_key= os.getenv('GOOGLE_API_KEY')
        )

        index_name = 'greenlead-global'
        if index_name not in pc.list_indexes().names():
            pc.create_index(
                name = index_name,
                dimension=768,
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
        
        vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)
        batch_size = 100
        for i in range(0, len(text_chunks), batch_size):
            batch = text_chunks[i:i+batch_size]
            documents =[Document(page_content=chunk) for chunk in batch]
            vectorstore.add_documents(documents)
        return vectorstore
    except Exception as e:
        logger.error(f"Error creating vectorstore: {e}")
        raise

path = '../pdfs'
pdfs  = [os.path.join(path, pdf) for pdf in os.listdir(path) if pdf.endswith('.pdf')]
texts = get_pdf_text(pdfs)
webpage = ['https://greenleadglobal.org/','https://greenleadglobal.org/impact/',"https://greenleadglobal.org/women-for-climate-w4c/",
    "https://greenleadglobal.org/clan/",
    "https://greenleadglobal.org/gcl/",
    "https://greenleadglobal.org/education/",
    "https://greenleadglobal.org/stories-of-climate-action/",
    "https://greenleadglobal.org/gl-at-cop/",
    "https://greenleadglobal.org/h-map/",
    "https://greenleadglobal.org/re-energize-bangladesh/",
    "https://greenleadglobal.org/climate-conversation/",
    "https://greenleadglobal.org/unplastic/",
    "https://greenleadglobal.org/cs/",
    "https://greenleadglobal.org/team/"]
web_loader = get_web_text(webpage)
texts += web_loader
with open('docs checking.txt', 'w', encoding='utf-8') as f:
    f.write(texts)
text_chunks = get_text_chunks(texts)
vectorstore = get_vectorstore(text_chunks)