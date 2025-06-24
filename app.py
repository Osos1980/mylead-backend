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

        # Greeting for the first message ONLY
        if first_message:
            greeting = "Hello, I’m MyLEAD, your trusted support partner at LEAD Public Schools. How can I help you today?"
            system_instructions = (
                f"{greeting}\n\n"
                "You are a professional AI assistant for LEAD Public Schools. "
                "You provide clear, accurate, and friendly help for technology issues, employee benefits, and official staff policies. "
                "Support only LEAD staff and teachers. If you do not know the answer, direct the user to support@technologylab.com (tech) or hradp@leadpublicschools.org (HR/benefits). "
                "Never answer questions for students or families. Always be concise, solution-focused, and reference official LEAD resources."
            )
        else:
            system_instructions = (
                "You are MyLEAD, the professional AI support assistant for LEAD Public Schools. "
                "Provide clear, concise, and accurate help with technology, employee benefits, and official staff policies. "
                "Support only LEAD staff and teachers. If a question is outside your scope, direct users to support@technologylab.com (tech) or hradp@leadpublicschools.org (HR/benefits). "
                "Never answer questions for students or families. Be solution-focused and always reference official LEAD resources and policies."
            )

        if not user_query:
            return jsonify({"response": "Please enter a question."})

        contents = [
            types.Content(
                role="user",
                parts=[types.Part(text=system_instructions)],
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
