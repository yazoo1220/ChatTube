"""Python file to serve as the frontend"""
import streamlit as st
from streamlit_chat import message
import os

from langchain.chains import ConversationChain
from langchain.llms import OpenAI


st.set_page_config(page_title="ChatTube", page_icon=":robot:")
st.header("▶️ ChatTube")
if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []


api_token = os.environ['OPENAI_API_KEY']　# st.text_input('OpenAI API Token',type="password")
# submit_button = st.button('authorize')

if submit_button:
    if api_token:
        os.environ['OPENAI_API_KEY'] = api_token
       # st.write('authorized.')
    else:
        st.write('Please input a valid API token.')
else:
    st.write('Waiting for API token...')

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
    st.write("waiting for api token...")

from llama_index import download_loader, GPTSimpleVectorIndex

video_url = st.text_input("YouTube URL 🔗")
if video_url:
    st.video(video_url)
    YoutubeTranscriptReader = download_loader("YoutubeTranscriptReader")
    loader = YoutubeTranscriptReader()
    documents = loader.load_data(ytlinks=[video_url])    
else:
    st.video('https://youtu.be/L_Guz73e6fw')
    YoutubeTranscriptReader = download_loader("YoutubeTranscriptReader")
    loader = YoutubeTranscriptReader()
    documents = loader.load_data(ytlinks=['https://youtu.be/L_Guz73e6fw'])  
def get_text():
    input_text = st.text_input("You: ", "この動画の要点を3つまとめてください。回答は日本語でお願いします。", key="input")
    return input_text


user_input = get_text()
load_button = st.button('ask')

from langchain.document_loaders import YoutubeLoader
index = ""
if load_button:
    try:
        index = GPTSimpleVectorIndex.from_documents(documents)
    except Exception as e:
        st.write("error loading the video: "+ str(e))
else:
    st.write("waiting for Youtube video to be loaded")


if index == "":
    pass
else:
    with st.spinner('typing...'):
        output = index.query(user_input)
        st.session_state.past.append(user_input)
        st.session_state.generated.append(output.response)


if st.session_state["generated"]:

    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        try:
            message(st.session_state["generated"][i], key=str(i))
            message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
        except:
            pass
