from openai import OpenAI
from dotenv import load_dotenv
import os
from langchain_community.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from utils import make_init_message

message = []

class openAIGPT():

    def __init__(self ,  model="gpt-3.5-turbo"):
        load_dotenv(".env")
        api_key = os.environ["OPENAI_API_KEY"] 
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def embed_documents(self , documents):
        embeddings = OpenAIEmbeddings()
        self.vector_db = FAISS.from_documents(documents, embeddings)

    def similarity_search(self , query):
        self.query = query
        retriever = self.vector_db.as_retriever(search_type="similarity_score_threshold" , search_kwargs={"score_threshold": 0.5 , "k":2})
        res_docs = retriever.get_relevant_documents(query)
        return res_docs
   
    def completion(self, res_docs ):
        messages = make_init_message(res_docs , self.query)
        completion = self.client.chat.completions.create(
        model=self.model,
        messages=messages
        )

        return completion.choices[0].message