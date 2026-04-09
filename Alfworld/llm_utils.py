import os
import openai
from langchain_community.chat_models import ChatOpenAI, ChatOllama
from langchain.schema import (
    HumanMessage,
)

from langchain_core.messages import HumanMessage
import random
import time

# Load environment variables from .env manually
env_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
if os.path.exists(env_file):
    with open(env_file) as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                try:
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value
                except ValueError:
                    continue

openai.api_key = OPENAI_API_KEY = ''
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY', '')
XAI_API_KEY = os.environ.get('XAI_API_KEY', '')

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    HarmBlockThreshold,
    HarmCategory,
)

from langchain.globals import set_llm_cache
from langchain.cache import SQLiteCache
set_llm_cache(SQLiteCache(database_path=".langchain.db"))

def google_llm(prompt, model_name='gemini-1.5-flash', temperature=0.0, top_p=1.0, max_tokens=4096, stop=[]):
    google_api_key = GOOGLE_API_KEY
    step = 0
    while step < 10:
        step += 1
        try:
            llm = ChatGoogleGenerativeAI(temperature=temperature,
                                         model=model_name, 
                                         top_p=top_p, 
                                         max_output_tokens=max_tokens, 
                                         google_api_key=google_api_key,
                                        )
            
            response = llm.invoke(prompt, stop=stop).content
            return response
        except Exception as e:
            wait_time = 2 ** step
            print(f"LLM Error ({step}/10): {e}. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)

import os
os.environ["REPLICATE_API_TOKEN"] = ''
import replicate
def meta_llm(prompt, model_name= "meta/meta-llama-3-70b", temperature=0.0, top_p=1.0, max_tokens=4096, stop=[]):
    input = {
        "top_p": top_p,
        "prompt": prompt,
        "min_tokens":max_tokens,
        "temperature": temperature,
        "presence_penalty": 1.15,

    }
    response = ''
    for event in replicate.stream(
        model_name,
        input=input
    ):
        response += str(event)
    return response

from langchain_community.callbacks import get_openai_callback
def openai_llm(prompt, model_name='gpt-4o', temperature=0.0, top_p=1.0, max_tokens=4096,stop=[]): # stop=["\n"]
    llm = ChatOpenAI(temperature=temperature,
                max_tokens=max_tokens,
                model_name=model_name,
                openai_api_key=OPENAI_API_KEY,
                model_kwargs={"stop": stop})
    with get_openai_callback() as cb:
        response = llm.invoke(prompt).content
        # print(cb)
    # request = llm([HumanMessage(content=prompt)]).content
    return response, cb.total_cost

def grok_llm(prompt, model_name='grok-3-mini', temperature=0.0, top_p=1.0, max_tokens=4096, stop=[]):
    """Grok (xAI) via OpenAI-compatible API. Free tier available at console.x.ai"""
    llm = ChatOpenAI(
        temperature=temperature,
        max_tokens=max_tokens,
        model_name=model_name,
        openai_api_key=XAI_API_KEY,
        openai_api_base='https://api.x.ai/v1',
        model_kwargs={"stop": stop} if stop else {},
    )
    with get_openai_callback() as cb:
        response = llm.invoke(prompt).content
    return response, cb.total_cost
