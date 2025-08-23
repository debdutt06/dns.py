import os
from dotenv import load_dotenv
from openai import OpenAI

# Load API key from .env
load_dotenv()
client = OpenAI()

SYSTEM_PROMPT = "You are a helpful assistant. Keep answers short and clear."

def main():
    print("OpenAI Agent (type 'exit' to quit)")
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    while True:
        user = input("You: ").strip()
        if user.lower() in ("exit", "quit"):
            break
        messages.append({"role": "user", "content": user})

        resp = client.chat.completions.create(
            model="gpt-4o-mini",  # fast + cheap model
            messages=messages,
            temperature=0.3,
        )
        reply = resp.choices[0].message.content
        print("Agent:", reply)
        messages.append({"role": "assistant", "content": reply})

if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Missing OPENAI_API_KEY in .env")
    main()
