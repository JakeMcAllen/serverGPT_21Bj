from nltk.corpus import wordnet as wn
import nltk

import requests 

import google.generativeai as genai

from rag import get_relevant_context_from_db, generate_prompt



nltk.download('wordnet')
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






if __name__ == "__main__":
    query = "How work blockchain ?"
    kw = {"blockchain": 1, "work": 2}

    resp = Agent(query, kw)
    print(f"\nresp: {resp}\n")