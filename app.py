import streamlit as st
import openai
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import os
from dotenv import load_dotenv

load_dotenv()

## Langsmith Tracking
os.environ['LANGCHAIN_API_KEY']=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACKING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]="Q&A Chatbot with OPENAI"

## Define Prompt Template
prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assitant. Please respond to the user queries"),
        ("user","Question:{question}")
    ]
)

def generate_respone(question,api_key,llm,temperature,max_tokens):
    ## emperature controls the randomness of an LLM’s output — lower means precise, higher means more creative.
    ## Need to interact with OPENAI models
    openai.api_key=api_key
    llm=ChatOpenAI(model=llm)
    output_parser=StrOutputParser()
    chain=prompt|llm|output_parser
    answer=chain.invoke({'question':question})
    return answer

##Title of the app
st.title("Enchanced Q&A Chatbot with OpenAI")

## Sidebar for settings
st.sidebar.title("Settings")
api_key=st.sidebar.text_input("Enter the OPENAI API Key", type="password")

## Dropdown to select OPENAI LLM models
llm = st.sidebar.selectbox(
    "Select an OPENAI model",
    ("gpt-4.1", "o4-mini", "o3")
)

## Slider to set temp and max tokens
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)

## Main interface for user input
st.write("Ask a Question")
user_input=st.text_input("You:")

if user_input:
    response=generate_respone(user_input, api_key,llm,temperature, max_tokens)
    st.write(response)
else:
    st.write("Please provide the query!")
