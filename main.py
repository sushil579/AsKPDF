

from llm import openAIGPT
from utils import pdf_loader


pdf_path = "GeneralBiology.pdf"
pages = pdf_loader(pdf_path)

gpt = openAIGPT()
gpt.embed_documents(pages)

res_docs = gpt.similarity_search("Nature of science")
output = gpt.completion(res_docs)