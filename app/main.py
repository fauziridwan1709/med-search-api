from flask import Flask, jsonify, request
from search import BSBI_instance
 
app = Flask(__name__)
 
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
        return jsonify(result)