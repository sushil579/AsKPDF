from llm import openAIPDF
from utils import get_data_and_source, make_init_message, pdf_loader

import streamlit
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import os
from langchain_openai import OpenAI
from langchain.chains import RetrievalQA, RetrievalQAWithSourcesChain
import tempfile
from langchain_community.document_loaders import PyPDFLoader


streamlit.title("ChatPDF")

streamlit.write("Welcome to ChatPDF! Please upload the pdf,enter your query below:")


#give options to select the model
streamlit.write("Please select the model you want to use:")

#select the model and give options
model = streamlit.selectbox("Select Model", ["gpt-3.5-turbo-0125", "Ollama"])
if model == "gpt-3.5-turbo-0125":
    api_key = streamlit.text_input("openai api key")

#create a widget to upload the pdf
streamlit.write("Please upload the pdf:")
pdf = streamlit.file_uploader("Upload PDF")
#query = input("Enter question here")
#pdf_path = r"C:\Users\Daku\Desktop\ChatPDF\GeneralBiology.pdf"
#model = "gpt-3.5-turbo"

if pdf is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf.read())
        tmp_path = tmp.name
    #output = pdf_query(pdf,query)
    pages = pdf_loader(tmp_path)

    gpt = openAIPDF(model)
    gpt.embed_documents(pages)
    query = streamlit.text_input("Query")

    os.environ["OPENAI_API_KEY"] = api_key

    if streamlit.button('Send'):
        res_docs = gpt.similarity_search(query, k=2)
        data_content, source = get_data_and_source(res_docs)
        messages = make_init_message(data_content, query)
        output = gpt.chat(messages)
        streamlit.write(output)
else:
    streamlit.write("Please upload a PDF")
