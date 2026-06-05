import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv('.env')
from groq import Groq
import base64 #binary to text converston scheme 

model="meta-llama/llama-4-scout-17b-16e-instruct"
api_key=os.getenv("GROQ_API_KEY")
img_path="sample_image.png"

prompt="what is details shown in the image? answer in two lines"

def encode_image(img_path):
    with open(img_path,"rb") as tfile:
        return base64.b64encode(tfile.read()).decode('utf-8')

base64_image=encode_image(img_path)


client = Groq()

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}",
                    },
                },
            ],
        }
    ],
    model=model,
)

print(chat_completion.choices[0].message.content)