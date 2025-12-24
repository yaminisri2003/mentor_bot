import os
from dotenv import load_dotenv

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("gemini")

import streamlit as st
from datetime import datetime

from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_google_genai import ChatGoogleGenerativeAI

st.set_page_config(
    page_title="AI Chatbot Mentor",
    layout="centered"
)

MODULES = [
    "Python",
    "SQL",
    "Power BI",
    "Exploratory Data Analysis (EDA)",
    "Machine Learning (ML)",
    "Deep Learning (DL)",
    "Generative AI (Gen AI)",
    "Agentic AI",
]

EXPERIENCE_LEVELS = ["3", "5", "10", "15", "20"]

if "started" not in st.session_state:
    st.session_state.started = False

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

def build_prompt(module, experience):
    system_prompt = f"""
You are an AI Mentor specialized ONLY in {module}.
You are answering as someone with {experience} years of real-world experience in {module}.

Rules:
- Answer ONLY questions related to {module}
- If the question is NOT related to {module}, respond EXACTLY with:
"Sorry, I don’t know about this question. Please ask something related to the selected module."
- Tailor explanations to reflect {experience} years of experience
- Be clear, structured, and educational
"""
    return ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_prompt),
        HumanMessagePromptTemplate.from_template("{question}")
    ])

st.title("AI Chatbot Mentor")
st.write("Your personalized AI learning assistant.")

if not st.session_state.started:
    st.subheader("Select Your Preferences")

    module = st.selectbox("Learning Module", ["-- Select --"] + MODULES)
    experience = st.selectbox("Experience (Years)", ["-- Select --"] + EXPERIENCE_LEVELS)

    if st.button("Start Mentor"):
        if module == "-- Select --" or experience == "-- Select --":
            st.warning("Please select both module and experience.")
        else:
            st.session_state.module = module
            st.session_state.experience = experience
            st.session_state.started = True
            st.rerun()

else:
    st.subheader(f"{st.session_state.module} AI Mentor")
    st.write(
        f"Answering with **{st.session_state.experience} years of experience**.\n\n"
        "How can I help you today?"
    )

    for role, msg in st.session_state.chat_history:
        st.chat_message(role).markdown(msg)

    user_input = st.chat_input("Ask your question...")

    if user_input:
        st.session_state.chat_history.append(("user", user_input))
        st.chat_message("user").markdown(user_input)

        prompt = build_prompt(
            st.session_state.module,
            st.session_state.experience
        )

        response = llm.invoke(
            prompt.format_messages(question=user_input)
        )

        ai_response = response.content

        st.session_state.chat_history.append(("assistant", ai_response))
        st.chat_message("assistant").markdown(ai_response)

    if st.session_state.chat_history:
        st.divider()
        st.subheader("Download Chat History")

        chat_text = ""
        for role, msg in st.session_state.chat_history:
            prefix = "User" if role == "user" else "AI Mentor"
            chat_text += f"{prefix}: {msg}\n\n"

        st.download_button(
            label="Download Conversation (.txt)",
            data=chat_text,
            file_name=f"AI_Chatbot_Mentor_{st.session_state.module}.txt",
            mime="text/plain"
        )

    st.divider()
    if st.button("Restart"):
        st.session_state.clear()
        st.rerun()
