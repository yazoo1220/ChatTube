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


api_token = st.text_input('OpenAI API　Token',type="password")
submit_button = st.button('Submit')

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

from llama_index import download_loader, GPTSimpleVectorIndex

video_url = st.text_input("your YouTube url here")
if video_url:
    st.video(video_url)
    YoutubeTranscriptReader = download_loader("YoutubeTranscriptReader")
    loader = YoutubeTranscriptReader()
    documents = loader.load_data(ytlinks=[video_url])    
else:
    pass

load_button = st.button('load')

from langchain.document_loaders import YoutubeLoader

if load_button:
    try:
        index = GPTSimpleVectorIndex.from_documents(documents)
    except Exception as e:
        st.write("error loading the video: "+ str(e))
else:
    st.write("waiting for Youtube video to be loaded")


def get_text():
    input_text = st.text_input("You: ", "こんにちは！", key="input")
    return input_text


user_input = get_text()

if user_input and index:
    output = index.query(user_input)

    st.session_state.past.append(user_input)
    st.session_state.generated.append(output.response)

if st.session_state["generated"]:
    message(st.session_state["generated"], key=str(i))
    message(st.session_state["past"], is_user=True, key=str(i) + "_user")
