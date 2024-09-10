from nltk.corpus import wordnet as wn
import nltk

import requests 

import google.generativeai as genai

from rag import get_relevant_context_from_db, generate_prompt





nltk.download('wordnet')


# API_URL = "https://api-inference.huggingface.co/models/openai-community/gpt2"
# API_key = "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"
GEMINI_API_KEY = "AIzaSyCFhiseeAtrOTwRc3Ib0-h4vqFha7KLgPM"



def Agent(query, keyWords):
        # WordNet
        keyWords = [wn.synsets(word)[0].name().split(".")[0] if wn.synsets(word) != [] else '' for word, qnt in keyWords.items()]
        keyWords.remove('')
        contextList = ", ".join(keyWords) 

        # RAG
        docs = get_relevant_context_from_db(query)

        # Build response
        question = generate_prompt(query, docs, contextList)
        print(f"\n\nQuestion: {question}\n\n")

        # Question to Model
        response = get_gemini_response(question)
        # response = get_model_response(question)

        # Register data to DB
        """TODO"""

        # Return
        return response


def get_model_response(msg: str):
    pass


def get_gemini_response(prompt):
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(model_name='gemini-pro')
    answere = model.generate_content(prompt)
    return answere.text


query = "How work blockchain ?"
kw = {"blockchain": 1, "work": 2}

resp = Agent(query, kw)
print(f"\nresp: {resp}\n")


# print( get_gemini_response("Introduce yourself please.") )

# def get_llama_HF_response(msg: str, min_length=256, max_length=500, print_response=False, eliminate_input=True) -> str:
#     payload = { "inputs": msg, "top_k": max_length, "min_length": min_length, max_length:"max_length", "temperature": 0.8, "max_time": 40, "do_sample": True }
#     headers = {"Authorization": f"Bearer {API_key}"} 
#     errorCount = 0
    
#     while True:
#         try:
#             response = requests.post(API_URL, headers=headers, json=payload)
            
#             if response.status_code != 200: raise Exception(f"response:{response} and status: {response.json()}")
#             if print_response: print(response.json(), end="\n\n")
#             return response.json()[0]["generated_text"][len(msg) + 1:] if eliminate_input else response.json()[0]["generated_text"]
            
#         except Exception as e:
#             print(f"Error: {e}" if errorCount == 0 else "." if errorCount%100000 == 0 else "", end="")
#             errorCount += 1
# get_gemini_RLHF_response(msg="Hello wold")