"""Python file to serve as the frontend"""
import streamlit as st
from streamlit_chat import message
import os

from langchain.chains import ConversationChain
from langchain.llms import OpenAI

os.environ['OPENAI_API_KEY']=st.text_input("your openai token here")

def load_chain():
    """Logic for loading the chain you want to use should go here."""
    llm = OpenAI(temperature=0)
    chain = ConversationChain(llm=llm)
    return chain

chain = load_chain()

# From here down is all the StreamLit UI.
st.set_page_config(page_title="ChatTube", page_icon=":robot:")
st.header("ChatTube")


if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []

video_url = st.text_input("your YouTube url here")
st.video(video_url)

from langchain.document_loaders import YoutubeLoader

loader = YoutubeLoader.from_youtube_url(video_url, add_video_info=True)

st.button("load video", on_click=loader.load)


def get_text():
    input_text = st.text_input("You: ", "こんにちは！", key="input")
    return input_text


user_input = get_text()

if user_input:
    output = chain.run(input=user_input)

    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if st.session_state["generated"]:

    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
