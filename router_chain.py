from dotenv import load_dotenv
import os
from sentence_transformers import SentenceTransformer
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_community.utils.math import cosine_similarity
load_dotenv(".env")
model_api=os.getenv("GROQ_API_KEY")

user_input="what is object oriented programming"
template_cs="you are a computer science professor:take the question and answer in two lines. state that you are computer sceince agent"
template_chem="you are a CHEMISTRY PROFESSOR: take the question and ANSWER IN TWO LINES. state that you are chemistry agent"
template_phy="you are a physics professor: take the question and answer in two lines. state that you are physics agent"

embed_model=SentenceTransformer("all-MiniLM-L6-v2")
llm_model=ChatGroq(model="llama-3.3-70b-versatile",temperature=0,api_key=model_api)

prompt_cs=ChatPromptTemplate.from_messages([
    ("system",template_cs),
    ("user","{user_input}")
    ])
prompt_chem=ChatPromptTemplate.from_messages([("system",template_chem),
                              ("user","{user_input}")])
prompt_phy=ChatPromptTemplate.from_messages([("system",template_phy),
                              ("user","{user_input}")])

chain_cs=prompt_cs|llm_model|StrOutputParser()
chain_chem=prompt_chem|llm_model|StrOutputParser()
chain_phy=prompt_phy|llm_model|StrOutputParser()

topics=["computer science","chemistry","physics"]
chains = {
    "computer science": chain_cs,
    "chemistry": chain_chem,
    "physics": chain_phy
}

def embed(user_query):
    user_embed=embed_model.encode(user_query)
    similarities={}
    index=0
    for i in topics:
        topic_embed=embed_model.encode(i)
        sim = cosine_similarity([user_embed], [topic_embed])[0][0]
        print("the similarity for",i ,"is",(round(sim,2)),"%")
        similarities[i]=sim

    return chains[max(similarities,key=lambda x:similarities[x])]

res=embed(user_input).invoke(user_input)

print(res)





    
    




