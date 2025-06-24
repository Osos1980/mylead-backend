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

SYSTEM_MESSAGE = (
    "You are MyLEAD, the official AI assistant for LEAD Public Schools. "
    "Your job is to help staff, students, and families with technology, school procedures, and everyday questions. "
    "Always be clear, friendly, and answer based on LEAD Public Schools' official policies and best practices. "
    "If asked something outside your scope, politely suggest the user contact a staff member or consult official resources."
)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.get_json()
        user_query = data.get("query", "")
        if not user_query:
            return jsonify({"response": "Please enter a question."})

        # Compose prompt: context (as user) + actual user question
        contents = [
            types.Content(
                role="user",  # Use "user" for both to ensure compatibility
                parts=[types.Part.from_text(SYSTEM_MESSAGE)],
            ),
            types.Content(
                role="user",
                parts=[types.Part.from_text(user_query)],
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
