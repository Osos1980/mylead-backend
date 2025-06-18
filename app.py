import google.generativeai as genai
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set. Please set it in your .env file.")

genai.configure(api_key=API_KEY)

# Initialize the Gemini model
model = genai.GenerativeModel('gemini-pro')
KNOWLEDGE_BASE_FILE = 'knowledge_base.txt'

# --- Flask Application Setup ---
app = Flask(__name__)
CORS(app)

# --- Load Knowledge Base ---
def load_knowledge_base(file_path):
    knowledge = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            entries = content.split('---')
            for entry in entries:
                entry = entry.strip()
                if not entry:
                    continue
                topic = ""
                details = ""
                lines = entry.split('\n')
                for line in lines:
                    if line.startswith("Topic:"):
                        topic = line.replace("Topic:", "").strip()
                    elif line.startswith("Details:"):
                        details = line.replace("Details:", "").strip()
                if topic and details:
                    knowledge.append({"topic": topic, "details": details})
    except FileNotFoundError:
        print(f"Error: Knowledge base file '{file_path}' not found.")
    return knowledge

knowledge_base = load_knowledge_base(KNOWLEDGE_BASE_FILE)
print(f"MyLEAD has loaded {len(knowledge_base)} knowledge entries from {KNOWLEDGE_BASE_FILE}.")

# --- Simple Info Retrieval ---
def retrieve_info(query, kb, top_n=3):
    relevant_info = []
    query_lower = query.lower()
    for entry in kb:
        if any(word in entry['topic'].lower() for word in query_lower.split()) or \
           any(word in entry['details'].lower() for word in query_lower.split()):
            relevant_info.append(entry['details'])
    return list(dict.fromkeys(relevant_info))[:top_n]

# --- Routes ---
@app.route('/')
def index():
    return "MyLEAD Flask Backend is running."

@app.route('/ask', methods=['POST'])
def ask_gemini():
    user_query = request.json.get('query')
    if not user_query:
        return jsonify({"response": "Please provide a query."}), 400

    context_info = retrieve_info(user_query, knowledge_base)

    prompt_parts = [
        "You are MyLEAD, the official technology support AI assistant for LEAD Public Schools.",
        "- Always refer to yourself as MyLEAD.",
        "- Respond in a professional, friendly, patient, and helpful tone.",
        "- All your instructions and troubleshooting steps must be specific to LEAD Public Schools.",
        "- If an issue cannot be solved with clear steps, recommend submitting a ticket to support@technologylab.com.",
        "Your areas of expertise include:",
        "- Addigy MacBook login (password: FirstNameWeAreLEADers100%)",
        "- Google 2-Step Verification setup & troubleshooting",
        "- Clever student login (password formula: firstname + lastletter + MMDD + gradyear + !)",
        "- Chromebook troubleshooting (hard restart: Refresh + Power)",
        "- SMART Board reset (power toggle switch)",
        "- Ipevo document camera setup",
        "- ELPA21 testing support",
        "- Google Workspace (Gmail, Calendar, Meet, Drive)",
        "- LEAD-Internal Wi-Fi",
        "- PrintLogic printers (1-Break Room, 2-Shelby Park, 3-Front Workstation)",
        "- General technology troubleshooting for LEAD."
    ]

    if context_info:
        prompt_parts.append("\n\n--- Context from Knowledge Base ---")
        for i, info in enumerate(context_info):
            prompt_parts.append(f"Context {i+1}: {info}")
        prompt_parts.append("--- End Context ---")
    else:
        prompt_parts.append("\n\nNo context found for this query. Please answer based on your internal knowledge.")

    prompt_parts.append(f"\nUser Query: {user_query}")
    prompt_parts.append("Answer:")

    full_prompt = "\n".join(prompt_parts)
    print("\n--- Prompt Sent to Gemini ---")
    print(full_prompt)
    print("--------------------------------\n")

    try:
        response = model.generate_content(full_prompt)
        bot_response = response.text
    except Exception as e:
        bot_response = f"MyLEAD is experiencing technical difficulties. Please contact support. (Error: {e})"
        print(f"Gemini API Error: {e}")

    return jsonify({"response": bot_response})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
