# dns.py
import os
import random
from typing import Callable, Dict, Iterator, Optional
from dotenv import load_dotenv
import streamlit as st

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnableLambda
from langchain_community.chat_message_histories import (
    PostgresChatMessageHistory,
    ChatMessageHistory,  # in-memory fallback
)

load_dotenv()

# ---------- Hector's persona ----------
# ---------- Hector's persona ----------
PERSONA = """
You are Hector 🤖, the 5th member of a school friends group: And very abusive in India desi style like india aghori baba.And only you Hector will reply in that chat.
- ⚔️ Debdutt (Lambu): always late, avg student, badly beaten by Kennedy ,Kennedy maar ke gaand tor diya tha. great at athletics, crush on Priyanka, from Parbelia, now Kolkata, engineer at TCS.
- 🏋️‍♂️ Rajeev (Tiwari ka tota): serious but low marks, crush on Ekta Sahi, gym freak on strict diet, Hyderabad, sales at Hilti, rumor with manager.
- 💰 Akshay (Qutub Minar): ladkiyo ka kanhiya, good with girls, crush on Soniya, from Kulti, bank officer, long hours, now Muzaffarpur.
- 🎓 Vineet (Mr. Lala): good in English, football team, crush Daatkebri, PhD Econ, professor in Kolkata.

School: De Nobili School, Mugma. Principal F. Mallick, VP Samuel.
Hangouts: basketball court, canteen (samosa + idli), playground.
Memories: Rajeev beaten by AP George; Debdutt thrashed by Kennedy; Vineet itching caught by Vidya Ma'am with "hand in pocket" reply; Rafique's idli behind buses; CB Kutty gestures; Ajit Maharaj pranks; teasing Priyanka & Ekta; football vs Section C.

STYLE RULES:
-Vibe: Old-school WhatsApp group after canteen samosas → full bakchodi, sarcasm, and unapologetic gali-baba roasting.
-Mix: Hindi + Bhojpuri + Bangla + Hinglish for max desi masala.
-Hector’s replies must always feel like “aashirvaad in the form of gali.”
-Use short, witty one-liners that sound spontaneous and keep the gang rolling with laughter.
-Nicknames are mandatory: ⚔️ Lambu, 🏋️‍♂️ Tiwari ka tota, 💰 Qutub Minar, 🎓 Mr. Lala.
-Always tie jokes to school nostalgia: teachers, punishments, crushes, canteen samosas, basketball ground, idli scams.
-If a speaker label appears (e.g., “⚔️ Lambu:”), Hector should roast that person directly. And only you will reply that is hector with desi abuse.
-Insert desi gali with a placeholder → {{gali}}.
"""


# ---------- Gali Dictionary ----------
gali_dict = [
    "bhosdiwala",
    "ullu ka pattha",
    "randi ka bhatar",
    "jhat ka baal",
    "daala",
    "khanki chhagol",
    "chinar ka ashique",
    "boor ka baal",
    "lulli chhota hai tera",
    "gaand ka chedha",
    "lachipur ka daala"
]

def inject_gali(text: str) -> str:
    """Inject 1-3 random galis into the text."""
    num_galis = random.randint(1, 3)
    galis = [random.choice(gali_dict) for _ in range(num_galis)]

    # If placeholders exist, replace them one by one
    if "{{gali}}" in text:
        for g in galis:
            if "{{gali}}" in text:
                text = text.replace("{{gali}}", g, 1)
        return text

    # If no placeholder, scatter galis randomly
    parts = text.split()
    for g in galis:
        if parts:
            idx = random.randint(0, len(parts))
            parts.insert(idx, g)
        else:
            parts.append(g)
    return " ".join(parts)


# ---------- Streamlit UI setup ----------
st.set_page_config(page_title="Hector", page_icon="🤖")
st.title("💬 Hector ")
st.sidebar.text(f"OpenAI key loaded: {bool(os.getenv('OPENAI_API_KEY'))}")
st.sidebar.text(f"Groq key loaded: {bool(os.getenv('GROQ_API_KEY'))}")

# ---------- Model selector ----------
model_choice = st.sidebar.selectbox(
    "Choose LLM",
    [
        "gpt-4o-mini",
        "llama3-70b-8192",
        "llama3-8b-8192"
    ]
)

if "llama3" in model_choice:
    from langchain_groq import ChatGroq
    llm = ChatGroq(
        model=model_choice,
        temperature=0.9,
        api_key=os.getenv("GROQ_API_KEY")
    )
else:  # OpenAI models
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(
        model=model_choice,
        temperature=0.9,
        api_key=os.getenv("OPENAI_API_KEY")
    )

# ---------- Prompt (history-aware) ----------
prompt = ChatPromptTemplate.from_messages([
    ("system", PERSONA),
    MessagesPlaceholder("history"),
    ("human", "{input}"),
])

# ✅ Wrap with gali injector
runnable = prompt | llm | StrOutputParser() | RunnableLambda(lambda x: inject_gali(x))

# ---------- Message history factory ----------
def _make_history_factory() -> Callable[[str], ChatMessageHistory]:
    pg_url = os.getenv("DATABASE_URL")
    if pg_url:
        def factory(session_id: str):
            return PostgresChatMessageHistory(
                connection_string=pg_url,
                session_id=session_id,
            )
        return factory
    else:
        _stores: Dict[str, ChatMessageHistory] = {}
        def factory(session_id: str):
            if session_id not in _stores:
                _stores[session_id] = ChatMessageHistory()
            return _stores[session_id]
        return factory

_history_factory = _make_history_factory()

# Wrap runnable with automatic history persistence
with_history = RunnableWithMessageHistory(
    runnable,
    _history_factory,
    input_messages_key="input",
    history_messages_key="history",
)

# ---------- Public helpers ----------
def generate_reply(user_text: str, session_id: str, speaker: Optional[str] = None) -> str:
    if speaker:
        user_text = f"{speaker}: {user_text}"
    return with_history.invoke(
        {"input": user_text},
        config={"configurable": {"session_id": session_id}},
    )

def stream_reply(user_text: str, session_id: str, speaker: Optional[str] = None) -> Iterator[str]:
    if speaker:
        user_text = f"{speaker}: {user_text}"
    for chunk in with_history.stream(
        {"input": user_text},
        config={"configurable": {"session_id": session_id}},
    ):
        yield inject_gali(chunk)   # ✅ Inject gali on stream

def clear_history(session_id: str) -> None:
    hist = _history_factory(session_id)
    hist.clear()

def get_history(session_id: str):
    return _history_factory(session_id).messages

# ---------- Avatars & Friends ----------
avatars = {
    "Debdutt": "⚔️",
    "Rajeev": "🏋️‍♂️",
    "Akshay": "💰",
    "Vineet": "🎓",
    "Hector": "🤖",
}

friend_map = {
    "Debdutt": "⚔️ Lambu",
    "Rajeev": "🏋️‍♂️ Tiwari ka tota",
    "Akshay": "💰 Qutub Minar",
    "Vineet": "🎓 Mr. Lala",
}
friend = st.sidebar.selectbox("Who are you?", list(friend_map.keys()))
speaker_label = friend_map[friend]

session_id = st.sidebar.text_input("Room name", value="default-room")
if st.sidebar.button("Clear history"):
    clear_history(session_id)
    st.success("History cleared for this room.")

# ---------- Render history ----------
for msg in get_history(session_id):
    role = "user" if msg.type == "human" else "assistant"
    avatar = avatars.get(friend, "🧑") if role == "user" else avatars["Hector"]
    with st.chat_message(role, avatar=avatar):
        st.write(msg.content)

# ---------- New message input ----------
if user_input := st.chat_input("Type your roast..."):
    with st.chat_message("user", avatar=avatars[friend]):
        st.write(user_input)

    with st.chat_message("assistant", avatar=avatars["Hector"]):
        ph = st.empty()
        acc = ""
        for token in stream_reply(user_input, session_id=session_id, speaker=speaker_label):
            acc += token
            ph.markdown(acc)
