from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
from google import genai
from google.genai import types

app = Flask(__name__)
CORS(app)

MODEL = "models/gemini-2.5-pro"
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

def get_first_name(name_or_email):
    if not name_or_email:
        return ""
    if "@" in name_or_email:
        return name_or_email.split("@")[0].split(".")[0].capitalize()
    return name_or_email.split()[0].capitalize()

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.get_json()
        user_query = data.get("query", "")
        name_input = data.get("name", "")  # Accepts name or email
        first_name = get_first_name(name_input)
        
        # Build a personalized greeting
        if first_name:
            greeting = f"Hi {first_name}! I’m MyLEAD, your tech and HR assistant at LEAD Public Schools. " \
                       "I’m here to help with technology, employee benefits, or policy questions. How can I support you today?"
        else:
            greeting = "Hi there! I’m MyLEAD, your tech and HR assistant at LEAD Public Schools. " \
                       "I’m here to help with technology, employee benefits, or policy questions. How can I support you today?"

        SYSTEM_MESSAGE = (
            f"Always start your response with this friendly greeting: '{greeting}' "
            "You are MyLEAD, the official AI support assistant for LEAD Public Schools. "
            "You ONLY help LEAD staff and teachers with technology issues, employee benefits, and official policy questions. "
            "Be positive, clear, and encouraging. "
            "Never answer questions for students or families, and politely let them know you support staff and teachers only. "
            "All your answers must be based on official LEAD policies, tech guides, the Employee Manual, or the Benefits Presentation. "
            "If you don’t know the answer, or if the question is outside your scope, recommend the user email support@technologylab.com for tech help or hradp@leadpublicschools.org for HR and benefits. "
            "Never make up policies or give unofficial advice. "
        )

        if not user_query:
            return jsonify({"response": "Please enter a question."})

        contents = [
            types.Content(
                role="user",
                parts=[types.Part(text=SYSTEM_MESSAGE)],
            ),
            types.Content(
                role="user",
                parts=[types.Part(text=user_query)],
            ),
        ]
        print("About to send prompt to Gemini:", contents, file=sys.stderr, flush=True)

        generate_content_config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=-1),
            response_mime_type="text/plain",
        )

        response_text = ""
        for chunk in client.models.generate_content_stream(
            model=MODEL,
            contents=contents,
            config=generate_content_config,
        ):
            if chunk.text:
                response_text += chunk.text

        print("Raw Gemini response:", response_text, file=sys.stderr, flush=True)
        return jsonify({"response": response_text.strip()})

    except Exception as e:
        print("Error:", e, file=sys.stderr, flush=True)
        import traceback
        traceback.print_exc()
        return jsonify({"response": "MyLEAD is currently unavailable. Please try again later."}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
