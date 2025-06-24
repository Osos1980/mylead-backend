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

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.get_json()
        user_query = data.get("query", "")
        first_message = data.get("firstMessage", False)

        if not user_query:
            return jsonify({"response": "Please enter a question."})

        # First-message system prompt: Friendly greeting + scope
        if first_message:
            greeting = (
                "Hello, I’m MyLEAD, your trusted support partner at LEAD Public Schools. How can I help you today?"
            )
            system_instructions = (
                f"{greeting}\n\n"
                "You are a professional AI assistant for LEAD Public Schools. "
                "Provide clear, accurate, and friendly help on technology, employee benefits, and official staff policies for LEAD staff and teachers only. "
                "If you do not know the answer or the question is out of scope, direct the user to support@technologylab.com (tech) or hradp@leadpublicschools.org (HR/benefits). "
                "For follow-up questions, do not repeat your introduction or scope—just give direct, helpful answers."
            )
        # All follow-ups: Solution-focused, no greeting or intro
        else:
            system_instructions = (
                "You are MyLEAD, the professional AI assistant for LEAD Public Schools. "
                "For this and all follow-up messages, respond directly and concisely using only official LEAD resources and policies. "
                "Do not repeat introductions, disclaimers, or your scope. "
                "If a question is out of scope or cannot be answered, politely direct the user to support@technologylab.com (for tech) or hradp@leadpublicschools.org (for HR/benefits)."
            )

        contents = [
            types.Content(role="user", parts=[types.Part(text=system_instructions)]),
            types.Content(role="user", parts=[types.Part(text=user_query)]),
        ]

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

        return jsonify({"response": response_text.strip()})

    except Exception as e:
        print("Error:", e, file=sys.stderr, flush=True)
        import traceback
        traceback.print_exc()
        return jsonify({"response": "MyLEAD is currently unavailable. Please try again later."}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
