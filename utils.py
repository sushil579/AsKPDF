from langchain_community.document_loaders import PyPDFLoader

def add_system_message(message , content):
    message.append({"role": "system", "content": content})
    return message

def add_user_message(message , content):
    message.append({"role": "user", "content": content})
    return message

def make_prompt(data , question):
    prompt = f"data - {data} , question : {question} "
    return prompt

def add_assistant_message(message , content):
    message.append({"role": "assistant", "content": content})
    return message

def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()
    return pages

def make_init_message(data , question):
    messages = []
    messages = add_system_message(messages , "You will be given data and the question ,provide answer based on the data")
    prompt = make_prompt(data , question)
    messages = add_user_message(messages , prompt)

    return messages