import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")


response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello, what is your name?"}]
)

print(response["choices"][0]["message"]["content"].strip())
