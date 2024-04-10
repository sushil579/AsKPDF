import tempfile

import streamlit

from llm import openAIPDF
from utils import get_data_and_source, make_init_message, pdf_loader

streamlit.title("AskPDF")

streamlit.write("Welcome to AsKPDF! Please upload the pdf,enter your query below:")

with streamlit.sidebar:
    streamlit.write("Please select the model you want to use:")

    model = streamlit.selectbox("Select Model", ["gpt-3.5-turbo-0125", "Ollama"])

    if model == "gpt-3.5-turbo-0125":
        api_key = streamlit.text_input("openai api key", type='password')
        if not api_key:
            streamlit.warning("Please enter the API key.")
        else:
            llm = openAIPDF(model, api_key)
    elif model == "Ollama":
        llm = OllamaPDF()    
    
    pdf = streamlit.file_uploader("Upload PDF")

query = streamlit.text_input("Query")

if pdf is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf.read())
        tmp_path = tmp.name

    pages = pdf_loader(tmp_path)

    llm.embed_documents(pages)

    if streamlit.button("Send"):
        res_docs = llm.similarity_search(query, k=2)
        data_content, metadata = get_data_and_source(res_docs)
        sources = [m["page"] for m in metadata]
        messages = make_init_message(data_content, query)
        output = llm.chat(messages)
        streamlit.write(output)
        streamlit.write("\n\n Sourced from pages")
        streamlit.write(sources)
else:
    streamlit.write("Please upload a PDF to chat")
