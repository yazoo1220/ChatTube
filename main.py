"""Python file to serve as the frontend"""
import streamlit as st
from streamlit_chat import message
import os

# show header and the authorization
st.set_page_config(page_title="ChatTube", page_icon=":robot:")
st.header("ChatTube")

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []


api_token = st.text_input('OpenAI API Token',type="password")
submit_button = st.button('authorize')

if submit_button:
    if api_token:
        os.environ['OPENAI_API_KEY'] = api_token
        st.write('API token set successfully.')
    else:
        st.write('Please input a valid API token.')
else:
    st.write('Waiting for API token input...')


# set up langchain
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate, LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)


def load_chain():
    """Logic for loading the chain you want to use should go here."""
    chat = ChatOpenAI(temperature=0)

    template="you are a helpful assistant who can explain or summarize what the people are talking about. you should always reply by the language the user is using"
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template="{input}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

    chain = LLMChain(llm=chat, prompt=chat_prompt)
    return chain

if os.environ['OPENAI_API_KEY']!="":
    try:
        chain = load_chain()
    except Exception as e:
        st.write("error loading data: " + str(e))
else:
    st.write("waiting for api token input...")

from llama_index import (download_loader,
    GPTKeywordTableIndex,
    LLMPredictor,
    ServiceContext
)


# create documents for llamaindex
video_url = st.text_input("YouTube URL")
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
    input_text = st.text_input("You: ", "この動画の要点を3つまとめてください。回答は日本語でお願いします。", key="text")
    return input_text


# interact with user and build index
user_input = get_text()
load_button = st.button('ask')


from langchain.llms import OpenAI

index = ""
if load_button:
    try:
        llm_predictor = LLMPredictor(llm= OpenAI(temperature=0.5))
        service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)
        index = GPTKeywordTableIndex.from_documents(documents, service_context=service_context)

    except Exception as e:
        st.write("error loading the video: "+ str(e))
else:
    st.write("ask me anything ;)")


if index == "":
    pass
else:
    with st.spinner('waiting for the answer...'):
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