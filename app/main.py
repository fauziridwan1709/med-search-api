from flask import Flask, jsonify, request
from flask_cors import CORS
from search import BSBI_instance
import os

basedir = os.path.abspath(os.path.dirname(__file__))
data_file = os.path.join(basedir, 'collection/')

WORDS = []
with open(data_file, "r") as file:
    for line in file.readlines():
        WORDS.append(line.rstrip())
 
app = Flask(__name__)
CORS(app)
 
@app.route("/")
def home_view():
        return "<h1>Welcome to good med search service</h1>"

@app.route("/search")
def search():
        args_dict = request.args.to_dict()
        query = args_dict.get("q")
        tfidf = BSBI_instance.retrieve_tfidf(query, k = 10)
        result = []
        for (score, doc) in tfidf:
                result.append(doc)
        response = jsonify(result)
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response