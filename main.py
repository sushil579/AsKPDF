from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
from pathlib import Path


load_dotenv(".env")


def split(path):
    path = Path(path)
    loader = PyPDFLoader(path)
    documents = loader.load()
    return documents

documents = split("GeneralBiology.pdf")