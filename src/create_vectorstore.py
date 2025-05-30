from PyPDF2 import PdfReader
import base64
import os
import logging
from dotenv import load_dotenv
load_dotenv()

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

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
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        logger.debug(f"Extracted text from {len(pdfs)} PDFs.")
        return text
    except FileNotFoundError as e:
        logger.error(f"PDF file not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Error extracting text from PDFs: {e}")
        raise
    

def get_text_chunks(text):
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_text(text)
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
        vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        logger.debug("Created FAISS vectorstore from text chunks.")
        return vectorstore
    except Exception as e:
        logger.error(f"Error creating vectorstore: {e}")
        raise

path = 'pdfs'
pdfs  = [os.path.join(path, pdf) for pdf in os.listdir(path) if pdf.endswith('.pdf')]
texts = get_pdf_text(pdfs)
text_chunks = get_text_chunks(texts)
db = get_vectorstore(text_chunks)
db.save_local('vectorstore/faiss_index')