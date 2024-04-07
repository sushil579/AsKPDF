

from llm import openAIPDF
from utils import get_data_and_source, make_init_message, pdf_loader


query = input("Enter question here")
pdf_path = r"C:\Users\Daku\Desktop\ChatPDF\GeneralBiology.pdf"
model = "gpt-3.5-turbo"

pages = pdf_loader(pdf_path)

gpt = openAIPDF(model)
gpt.embed_documents(pages)

res_docs = gpt.similarity_search(query, k=2)


data_content, source = get_data_and_source(res_docs)

messages = make_init_message(data_content, query)

output = gpt.chat(messages)

print(output)
