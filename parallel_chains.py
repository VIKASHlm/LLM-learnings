from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import os
load_dotenv(".env")

polite=ChatPromptTemplate([
                  ("system","you are a soft hearted police. answer this politely"),
                  ("user","{topic}"),
                  ])
angry=ChatPromptTemplate([
                  ("system","you are a hot headed police. answer this more angrily and rudely"),
                  ("user","{topic}"),
                  ])
model=ChatGroq(model="llama-3.3-70b-versatile",api_key=os.getenv("GROQ_API_KEY"),temperature=0.5)

polite_chain=polite|model|StrOutputParser()
angry_chain=angry|model|StrOutputParser()

map_chain=RunnableParallel(first_gen=polite_chain,second_gen=angry_chain)
topic=" i lost my dog please file a complaint"

res=map_chain.invoke({"topic":topic})
print(res)