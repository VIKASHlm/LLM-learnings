from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from rich.markdown import Markdown
from rich.console import Console
from dotenv import load_dotenv
load_dotenv('.env')
console=Console()

llm=ChatGroq(model="llama-3.3-70b-versatile",temperature=0.7)

store={}

def get_session_id(session_id: str)->BaseChatMessageHistory:
    if session_id not in store:
        store[session_id]=InMemoryChatMessageHistory()
    return store[session_id]

initial_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a creative storyteller. Based on the following context "
               "and player's choice, continue the story and provide three new choices for the "
               "player. keep the story extremely short and concise. Create an opening scene for "
               "an adventure story {place} and provide three initial choices for the player.")
])

context_chain=initial_prompt |llm
config={"configurable":{"session_id":"03"}}
llm_with_message_history=RunnableWithMessageHistory(context_chain,get_session_history=get_session_id)

context=llm_with_message_history.invoke({"place":"a dark forest"},config=config)

console.print(Markdown(context.content))

def process_players_choice(choice):
    response = llm_with_message_history.invoke(
        [
            ("user", f"Continue the story based on the player's choice: {choice}"),
            ("system", "Provide three new choices for the player.")
        ],
        config=config
    )
    return response

while True:
    player_choice=input("enter the choice,if finisher type quit: ")
    if player_choice.lower()=="quit":
        break
    context=process_players_choice(player_choice)
    console.print(Markdown(context.content))

console.print(Markdown(context.content))
    