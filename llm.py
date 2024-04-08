import os

import ollama
from dotenv import load_dotenv
from langchain_community.embeddings import OllamaEmbeddings, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from openai import OpenAI


class openAIPDF:
    def __init__(self, model="gpt-3.5-turbo", api_key=None):
        if not api_key:
            load_dotenv(".env")
            api_key = os.environ["OPENAI_API_KEY"]

        self.client = OpenAI(api_key=api_key)
        self.model = model

    def embed_documents(self, documents):
        embeddings = OpenAIEmbeddings()
        self.vector_db = FAISS.from_documents(documents, embeddings)

    def similarity_search(self, query, k):
        self.query = query
        retriever = self.vector_db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": 0.6, "k": k},
        )
        res_docs = retriever.get_relevant_documents(query)
        return res_docs

    def chat(self, messages):
        completion = self.client.chat.completions.create(
            model=self.model, messages=messages
        )

        return completion.choices[0].message.content


class OllamaPDF:
    def __init__(self, model="llama2"):
        self.model = model

    def embed_documents(self, pages):
        embeddings = OllamaEmbeddings()
        self.vector_db = FAISS.from_documents(pages, embeddings)

    def similarity_search(self, query, k=2):
        self.question = query
        docs = self.vector_db.similarity_search(query, k=k)
        return docs

    def chat(self, messages):
        response = ollama.chat(model=self.model, messages=messages)
        return response["message"]["content"]
