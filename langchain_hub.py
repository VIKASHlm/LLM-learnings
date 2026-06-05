from langchain import hub
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langsmith import Client
load_dotenv('env')
from pprint import pprint
import os
client=Client()
prompt = client.pull_prompt("hardkothari/prompt-maker",dangerously_pull_public_prompt=True)
model=ChatGroq(model="llama-3.3-70b-versatile",api_key=os.getenv("GROQ_API_KEY"),temperature=0.1)
chain=prompt|model|StrOutputParser()

prompt_template=chain.invoke({"lazy_prompt":"love, romance, lust ...","task":"create a shakespear style 5 line story"})

res=model.invoke(prompt_template)
print(res.content)
