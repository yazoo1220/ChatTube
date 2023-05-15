"""Python file to serve as the frontend"""
import streamlit as st
from streamlit_chat import message
import os

from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS


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

from langchain.memory import ConversationBufferMemory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

def load_chain(documents):
    """Logic for loading the chain you want to use should go here."""
    if is_gpt4:
        model = "gpt-4"
    else:
        model = "gpt-3.5-turbo"
    llm = ChatOpenAI(temperature=0.9, model_name=model, streaming=True, verbose=True)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap  = 20, length_function = len)
    docs = text_splitter.split_documents(documents)
    st.write(len(docs))
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(docs, embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 1})
    chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever,max_tokens_limit=4096,memory=memory)
    return chain


from langchain.document_loaders import YoutubeLoader

video_url = st.text_input("YouTube URL 🔗","https://youtu.be/L_Guz73e6fw")
if video_url:
    st.video(video_url)
else:
    pass

    
def get_text():
    input_text = st.text_input("You: ", "この人の言いたいことを要約してください。", key="input")
    return input_text

with st.form(key='ask'):
    user_input = get_text()
    ask_button = st.form_submit_button('ask')

if ask_button:
    with st.spinner('typing...'):
        chat_history = []
        loader = YoutubeLoader.from_youtube_url(video_url, add_video_info=True)  
        documents = loader.load()
        qa = load_chain(documents)
        prefix = 'あなたは優秀なアシスタントです。'# 'you are a very helpful explainer of videos. The attached is a transcript of a YouTube video and your task is to answer question. if you dont have a good answer based on the video, please say you do not know. yo your answer should be the same as i use after this sentence.  '
        result = qa({"question": prefix + user_input, "chat_history": chat_history})
        st.session_state.past.append(user_input)
        st.session_state.generated.append(result['answer'])


if st.session_state["generated"]:

    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        try:
            message(st.session_state["generated"][i], key=str(i))
            message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
        except:
            pass
