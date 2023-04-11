"""Python file to serve as the frontend"""
import streamlit as st
from streamlit_chat import message
import os

from langchain.chains import ConversationChain
from langchain.llms import OpenAI


st.set_page_config(page_title="ChatTube", page_icon=":robot:")
st.header("ChatTube")

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []

authorization = st.checkbox("authorize with OpenAI API key")

if authorization:
    api_token = st.text_input('OpenAI API Token',type="password")
    submit_button = st.button('Submit')
else:
    pass

if submit_button:
    if api_token:
        os.environ['OPENAI_API_KEY'] = api_token
        st.write('API token set successfully.')
    else:
        st.write('Please input a valid API token.')
else:
    st.write('Waiting for API token input...')

def load_chain():
    """Logic for loading the chain you want to use should go here."""
    llm = OpenAI(temperature=0)
    chain = ConversationChain(llm=llm)
    return chain

if os.environ['OPENAI_API_KEY']!="":
    try:
        chain = load_chain()
    except Exception as e:
        st.write("error loading data: " + str(e))
else:
    st.write("waiting for api token input...")


video_url = st.text_input("your YouTube url here")
if video_url:
    st.video(video_url)
else:
    pass

load_button = st.button('load')

from langchain.document_loaders import YoutubeLoader

if load_button:
    try:
        loader = YoutubeLoader.from_youtube_channel(video_url, add_video_info=False,language="en")
        loader.load()
    except Exception as e:
        st.write("error loading the video: "+ str(e))
else:
    st.write("waiting for Youtube video to be loaded")


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
