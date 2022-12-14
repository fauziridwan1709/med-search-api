from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
from search import BSBI_instance
 
app = Flask(__name__)
CORS(app, support_credentials=True)
 
@app.route("/")
@cross_origin(supports_credentials=True)
def home_view():
        return "<h1>Welcome to good med search service</h1>"

@app.route("/search")
@cross_origin(supports_credentials=True)
def search():
        args_dict = request.args.to_dict()
        query = args_dict.get("q")
        tfidf = BSBI_instance.retrieve_tfidf(query, k = 10)
        result = []
        for (score, doc) in tfidf:
                result.append(doc)
        return jsonify(result)