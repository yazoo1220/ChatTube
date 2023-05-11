"""Python file to serve as the frontend"""
import streamlit as st
from streamlit_chat import message
import os

from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma


st.set_page_config(page_title="ChatTube", page_icon=":robot:")
st.header("▶️ ChatTube")

is_gpt4 = st.checkbox('Enable GPT4',help="With this it might get slower")

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []


api_token = os.environ['OPENAI_API_KEY'] # st.text_input('OpenAI API Token',type="password")

def get_chat_history(inputs) -> str:
    res = []
    for human, ai in inputs:
        res.append(f"Human:{human}\nAI:{ai}")
    return "\n".join(res)

def load_chain(documents):
    """Logic for loading the chain you want to use should go here."""
    if is_gpt4:
        model = "gpt-4"
    else:
        model = "gpt-3.5-turbo"
    llm = ChatOpenAI(temperature=0.9, model_name=model, streaming=True, verbose=True)
    text_splitter = CharacterTextSplitter(chunk_size=3000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    db = Chroma.from_documents(docs, embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 1})
    chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever,get_chat_history=get_chat_history)
    return chain

if os.environ['OPENAI_API_KEY']!="":
    try:
        chain = load_chain()
    except Exception as e:
        st.write("error loading data: " + str(e))
else:
    st.write("waiting for api token...")

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
qa = ""
if load_button:
    try:
        qa = load_chain(documents)
    except Exception as e:
        st.write("error loading the video: "+ str(e))
else:
    st.write("waiting for Youtube video to be loaded")


if qa == "":
    pass
else:
    with st.spinner('typing...'):
        chat_history = []
        result = qa({"question": prefix + user_input, "chat_history": chat_history})
        st.session_state.past.append(user_input)
        st.session_state.generated.append(output.response)


if st.session_state["generated"]:

    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        try:
            message(st.session_state["generated"][i], key=str(i))
            message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
        except:
            pass
