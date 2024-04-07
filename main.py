

from llm import openAIGPT
from utils import make_init_message, pdf_loader


pdf_path = r"C:\Users\Daku\Desktop\chatPDF\ChatPDF\GeneralBiology.pdf"
pages = pdf_loader(pdf_path)

gpt = openAIGPT()
gpt.embed_documents(pages)

query = "Nature of science"
res_docs = gpt.similarity_search(query)
messages = make_init_message(res_docs, query)


output = gpt.completion(res_docs)
