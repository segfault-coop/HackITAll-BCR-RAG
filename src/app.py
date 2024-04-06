import os
import tempfile
import streamlit as st
from streamlit_chat import message
from streamlit_option_menu import option_menu
from rag import LLamaChatPDF

# st.set_page_config(page_title="Gheorghe")
# page_bg="""
# <style>
# [data-testid = "stAppViewContainer"]{
# background-color : #3364FF;
# text-color : #FFFFFF;
# }
# <style>
# """
# st.markdown(page_bg, unsafe_allow_html=True)

def HRista():
    st.text("Esti o femeie independenta si plina de bani")

def Teknic():
    st.text("Esti un mascul alpha")

def display_messages():
    st.subheader("Chat")
    for i, (msg, is_user) in enumerate(st.session_state["messages"]):
        message(msg, is_user=is_user, key=str(i))
    st.session_state["thinking_spinner"] = st.empty()


def process_input():
    if st.session_state["user_input"] and len(st.session_state["user_input"].strip()) > 0:
        user_text = st.session_state["user_input"].strip()
        with st.session_state["thinking_spinner"], st.spinner(f"Thinking"):
            agent_text = st.session_state["assistant"].ask(user_text)

        st.session_state["messages"].append((user_text, True))
        st.session_state["messages"].append((agent_text, False))


def read_and_save_file():
    st.session_state["assistant"].clear()
    st.session_state["messages"] = []
    st.session_state["user_input"] = ""

    for file in st.session_state["file_uploader"]:
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf.write(file.getbuffer())
            file_path = tf.name

        with st.session_state["ingestion_spinner"], st.spinner(f"Ingesting {file.name}"):
            st.session_state["assistant"].ingest(file_path)
        os.remove(file_path)


def page():

    st.header("Gheorghe") 
    
    with st.sidebar:
        selected = option_menu(
            menu_title = "Gheorghe",
            options = ["Gicu", "Teknic", "HRista"]
        )
        toggle_switch = st.checkbox("Do you have rights?")

    
        if toggle_switch:
            st.write("Admin")

        else:
            st.write("User")

    if selected == "Gicu":
        st.title(f"Welcome to {selected}")
        if len(st.session_state) == 0:
            st.session_state["messages"] = []
            st.session_state["assistant"] = LLamaChatPDF()
        if toggle_switch:
            st.subheader("Upload a document")
            st.file_uploader(
                "Upload document",
                type=["pdf"],
                key="file_uploader",
                on_change=read_and_save_file,
                label_visibility="collapsed",
                accept_multiple_files=True,
                )

    if selected == "Teknic":
        st.title(f"Welcome to {selected}")
        if len(st.session_state) == 0:
            st.session_state["messages"] = []
            st.session_state["assistant"] = LLamaChatPDF()        
        if toggle_switch:
            st.subheader("Upload a document")
            st.file_uploader(
                "Upload document",
                type=["pdf"],
                key="file_uploader",
                on_change=read_and_save_file,
                label_visibility="collapsed",
                accept_multiple_files=True,
                )
            Teknic()
        
    if selected == "HRista":
        st.title(f"Welcome to {selected}")
        if len(st.session_state) == 0:
            st.session_state["messages"] = []
            st.session_state["assistant"] = LLamaChatPDF()
        if toggle_switch:
            st.subheader("Upload a document")
            st.file_uploader(
                "Upload document",
                type=["pdf"],
                key="file_uploader",
                on_change=read_and_save_file,
                label_visibility="collapsed",
                accept_multiple_files=True,
                )
            HRista()

 

    st.session_state["ingestion_spinner"] = st.empty()

    display_messages()
    st.text_input("Message", key="user_input", on_change=process_input)
        


if __name__ == "__main__":
    page()