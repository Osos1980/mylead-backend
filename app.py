import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import google.generativeai as genai
from pathlib import Path

load_dotenv(dotenv_path=Path('.') / '.env')

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set.")

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel(model_name="chat-bison")

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

    try:
        response = model.generate_content(user_query)
        return jsonify({"response": response.text})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"response": "MyLEAD is currently unavailable. Please try again later."})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=10000, debug=True)
