import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv('.env')

api_key=os.getenv('OPENAI_API_KEY')
model="gpt-4o-mini"
genai=ChatOpenAI(model=model,temperature=0.5,api_key=api_key)

res=genai.invoke("what is a tree?")
print(res)