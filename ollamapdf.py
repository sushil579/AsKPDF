from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
import ollama
from utils import load_pdf

class OllamaPDF:

    def __init__(self, model='llama2'):
        self.model = model
    
    def embed_documents(self, pages):
        embeddings = OllamaEmbeddings()
        self.vector_db = FAISS.from_documents(pages, embeddings)
    
    def similarity_search(self, question, k=2):
        self.question = question
        docs = self.vector_db.similarity_search(question, k=k)
        return docs
        # content = question + docs[0].page_content
        # return content
    
    def chat(self, messages):
        response = ollama.chat(model= self.model, messages=messages)
        return response['message']['content']
