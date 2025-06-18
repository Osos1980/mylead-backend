import os
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return "MyLEAD Flask Backend is running."

@app.route('/ask', methods=['POST'])
def ask():
    user_query = request.json.get('query')
    if not user_query:
        return jsonify({"response": "Please provide a query."}), 400

    # 🔄 Static test response instead of Gemini
    return jsonify({"response": "This is a test response from the backend. Gemini is temporarily disabled."})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=10000, debug=True)
