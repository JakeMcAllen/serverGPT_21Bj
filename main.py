from flask import Flask
from ai.Agent import Agent

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/agent")
def AgentCall(query, keyWords):
    print(query)
    print(keyWords)
    return "true"

    # return Agent(query, keyWords)