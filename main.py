

from llm import openAIGPT
from utils import get_data_and_source, make_init_message, pdf_loader


query = "Nature of science"
pdf_path = r"C:\Users\Daku\Desktop\ChatPDF\GeneralBiology.pdf"
model = "gpt-3.5-turbo"

pages = pdf_loader(pdf_path)

gpt = openAIGPT(model)
gpt.embed_documents(pages)

res_docs = gpt.similarity_search(query)
data_content, source = get_data_and_source(res_docs)

messages = make_init_message(data_content, query)


output = gpt.completion(messages)
print(output)
