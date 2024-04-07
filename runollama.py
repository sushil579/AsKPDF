from ollamapdf import OllamaPDF
from utils import make_prompt, add_system_message, add_user_message, add_assistant_message, make_init_message, load_pdf

model = "llama2"
pages = load_pdf("test.pdf")
question = input("Enter your question: ")

GPTobj = OllamaPDF(model)
GPTobj.embed_documents(pages)

docs = GPTobj.similarity_search(question, k=2)
full_docs = [doc.page_content for doc in docs]
messages = make_init_message(full_docs, question)
print(messages)

respoonse = GPTobj.chat(messages)
print(respoonse)

