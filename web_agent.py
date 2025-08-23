import os
from dotenv import load_dotenv
from openai import OpenAI
import streamlit as st

load_dotenv()
client = OpenAI()

st.title("💬 Hector")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": "You are a helpful Hector."}]

for msg in st.session_state.messages:
    if msg["role"] != "system":
        st.chat_message(msg["role"]).write(msg["content"])

if user_input := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=st.session_state.messages
    )
    reply = resp.choices[0].message.content

    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.chat_message("assistant").write(reply)
