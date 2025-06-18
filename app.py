import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import google.generativeai as genai
from pathlib import Path

# Load environment variables
load_dotenv(dotenv_path=Path('.') / '.env')

# Configure Gemini API
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set.")

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-pro')

# File for knowledge base
KNOWLEDGE_BASE_FILE = 'knowledge_base.txt'

# Flask setup
app = Flask(__name__)
CORS(app)

# Load knowledge base
def load_knowledge_base(file_path):
    knowledge = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            entries = f.read().split('---')
            for entry in entries:
                entry = entry.strip()
                if not entry:
                    continue
                topic, details = "", ""
                for line in entry.split('\n'):
                    if line.startswith("Topic:"):
                        topic = line.replace("Topic:", "").strip()
                    elif line.startswith("Details:"):
                        details = line.replace("Details:", "").strip()
                if topic and details:
                    knowledge.append({"topic": topic, "details": details})
    except FileNotFoundError:
        print(f"Knowledge base file '{file_path}' not found.")
    return knowledge

knowledge_base = load_knowledge_base(KNOWLEDGE_BASE_FILE)
print(f"MyLEAD loaded {len(knowledge_base)} knowledge entries.")

# Match user queries
def retrieve_info(query, kb, top_n=3):
    query_lower = query.lower()
    relevant_info = []
    for entry in kb:
        if any(word in entry['topic'].lower() for word in query_lower.split()) or \
           any(word in entry['details'].lower() for word in query_lower.split()):
            relevant_info.append(entry['details'])
    return list(dict.fromkeys(relevant_info))[:top_n]

# Routes
@app.route('/')
def index():
    return "MyLEAD Flask Backend is running."

@app.route('/ask', methods=['POST'])
def ask():
    user_query = request.json.get('query')
    if not user_query:
        return jsonify({"response": "Please provide a query."}), 400

    context_info = retrieve_info(user_query, knowledge_base)
    prompt_parts = [
        "You are MyLEAD, the official tech support AI for LEAD Public Schools.",
        "Respond clearly, helpfully, and refer to LEAD-specific systems.",
        "If unsure, advise users to email support@technologylab.com.",
    ]

    if context_info:
        prompt_parts.append("\n--- Context from Knowledge Base ---")
        for i, info in enumerate(context_info):
            prompt_parts.append(f"Context {i+1}: {info}")
        prompt_parts.append("--- End Context ---")

    prompt_parts.append(f"\nUser Query: {user_query}\nAnswer:")

    try:
        response = model.generate_content("\n".join(prompt_parts))
        return jsonify({"response": response.text})
    except Exception as e:
        import traceback
        traceback.print_exc()  # ✅ Print full error in Render logs
        print("Gemini API Error:", e)
        return jsonify({"response": "MyLEAD is currently unavailable. Please try again later."})

# ✅ Render-compatible run configuration
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=10000, debug=True)
