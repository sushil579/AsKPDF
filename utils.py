from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
from pathlib import Path


load_dotenv(".env")


def add_system_message(message , content):
    message.append({"role": "system", "content": content})
    return message

def add_user_message(message , content):
    message.append({"role": "user", "content": content})
    return message

def make_prompt(data , question):
    prompt = f"data - {data} , question : {question} "
    return prompt


def make_init_message(data , question):
    messages = []
    messages = add_system_message(messages , "You will be given data and the question ,provide answer based on the data")
    prompt = make_prompt(data , question)
    messages = add_user_message(messages , prompt)

    return messages
        

def pdf_loader(path):
    path = Path(path)
    loader = PyPDFLoader(path)
    documents = loader.load()
    return documents
