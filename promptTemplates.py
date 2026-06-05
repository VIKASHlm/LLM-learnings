from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import os
load_dotenv(".env")

prompt_template=ChatPromptTemplate([
                  ("system","you are a translator specialist. strictly dont change the meaning"),
                  ("user","translate this sentence:{input} into {target_language}"),
                  ])
model=ChatGroq(model="llama-3.3-70b-versatile",api_key=os.getenv("GROQ_API_KEY"),temperature=0.5)

chain=prompt_template|model|StrOutputParser()

res=chain.invoke({"input":"i go to school in an bicycle with a handbag in my shoulders","target_language":"french"})
print(res)