from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv


from pathlib import Path


load_dotenv(".env")


embeddings = OpenAIEmbeddings()


path = Path("GeneralBiology.pdf")
loader = PyPDFLoader(path)
documents = loader.load()


db = FAISS.from_documents(documents, embeddings)


query = "what does scientific method deals with ?"

retriever = db.as_retriever(search_type="similarity_score_threshold" , search_kwargs={"score_threshold": 0.5 , "k":2})
docs = retriever.get_relevant_documents(query)

for i in docs:
    print(docs)




