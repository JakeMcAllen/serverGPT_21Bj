import os
import signal 
import sys 

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceHubEmbeddings
from langchain.embeddings import HuggingFaceHubEmbeddings
from langchain.llms import HuggingFaceHub




GEMEINI_API_KEY = "AIzaSyCFhiseeAtrOTwRc3Ib0-h4vqFha7KLgPM"
HUGGINFACE_TOKEN = "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"


os.environ['HUGGINGFACEHUB_API_TOKEN'] =  HUGGINFACE_TOKEN
os.environ['HF_TOKEN']  =  HUGGINFACE_TOKEN


def signal_handler(sign, frame):
    print('\nThank for using gemini. ')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def get_relevant_context_from_db(query):
    context = ""
    embedding_function = HuggingFaceHubEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory="./chroma_db_nccn", embedding_function=embedding_function)
    search_results = vector_db.similarity_search(query, k=6)
    for result in search_results:
        context += result.page_content + "\n"
    return context

while True:
    print("What would you like to ask")
    query = input(">>> ")
    context = get_relevant_context_from_db(query)
    print(f"content: {context}")