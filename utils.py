def add_system_message(message , content):
    message.append({"role": "system", "content": content})
    return message

def add_user_message(message , content):
    message.append({"role": "user", "content": content})
    return message

def make_prompt(data , question):
    prompt = f"data - {data} , question : {question} "