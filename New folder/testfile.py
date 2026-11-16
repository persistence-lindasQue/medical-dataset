import litellm

r = litellm.completion(
    model="cohere_chat/c4ai-aya-expanse-32b",
    messages=[{"role": "user", "content": "Say goodbye in French."}],
)
print(r.choices[0].message.content)
