from openai import OpenAI
from dotenv import load_dotenv
import os

from utils import add_system_message, add_user_message, make_prompt

message = []

class openAIGPT():

    def __init__(self , model="gpt-3.5-turbo"):
        load_dotenv(".env")
        api_key = os.environ["OPENAI_API_KEY"] 
        self.client = OpenAI(api_key=api_key)
        self.model = model

   
    def completion(self, data ,question):
        messages = []
        messages = add_system_message(messages , "You will be given data and the question ,provide answer based on the data")
        prompt = make_prompt(data , question)
        messages = add_user_message(messages , prompt)
        
        completion = self.client.chat.completions.create(
        model=self.model,
        messages=messages
        )

        return completion.choices[0].message