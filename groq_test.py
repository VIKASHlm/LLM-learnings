import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv('.env')

api_key=os.getenv("GROQ_API_KEY")
model="llama-3.3-70b-versatile"

genai=ChatGroq(model=model,temperature=0.5,api_key=api_key)

res=genai.invoke("what is a tree? answer in two lines")
print(res.content)
print(res.response_metadata['token_usage'])