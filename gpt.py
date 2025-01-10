import openai

openai.api_key = "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"

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello, what is your name?"}]
)

print(response["choices"][0]["message"]["content"].strip())
