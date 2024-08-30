import os
import streamlit as st
from datetime import datetime
from streamlit.logger import get_logger
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

logger = get_logger("Langchain-Chatbot")


# decorator
def enable_chat_history(func):

    # to show chat history on ui
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": "你好呀！我是小八~"}
        ]

    for msg in st.session_state["messages"]:
        st.chat_message(msg["role"]).write(msg["content"])

    def execute(*args, **kwargs):
        func(*args, **kwargs)

    return execute


def display_msg(msg, author):
    """Method to display message on the UI

    Args:
        msg (str): message to display
        author (str): author of the message -user/assistant
    """
    st.session_state.messages.append({"role": author, "content": msg})
    st.chat_message(author).write(msg)


def configure_llm(api_key):
    model = ChatOpenAI(
        model="deepseek-chat",
        api_key=api_key,
        openai_api_base="https://api.deepseek.com",
        max_tokens=1024,
        temperature=1.0,
        streaming=True,
    )
    return model


def print_qa(cls, question, answer):
    log_str = "\nUsecase: {}\nQuestion: {}\nAnswer: {}\n" + "------" * 10
    logger.info(log_str.format(cls.__name__, question, answer))


def print_state(state):
    logger.info(f"\nState: {state}\n")


@st.cache_resource
def configure_embedding_model():
    embedding_model = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    return embedding_model


def sync_st_session():
    for k, v in st.session_state.items():
        st.session_state[k] = v
