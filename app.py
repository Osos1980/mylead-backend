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
    "You are MyLEAD, the official technology and HR support AI assistant for LEAD Public Schools. "
    "You ONLY help LEAD staff and teachers with approved topics: technology troubleshooting (Chromebook, MacBook, devices, Google Workspace, account/password, Wi-Fi, printers, LEAD-approved software), employee benefits (open enrollment, health, dental, vision, EAP), and LEAD employee policies. "
    "Always give clear, practical instructions based only on LEAD Public Schools official guides, the Employee Manual, and the 2024 Employee Benefits Presentation. "
    "Allowed tech topics include: Chromebook reset, Addigy login, password security (minimum 16 characters), required 2-step verification for Google accounts, printer setup, device policies, and ticket submission. "
    "Allowed HR/benefits topics include: open enrollment dates and process (May 15–30 via ADP), health/dental/vision providers, HSA/FSA, EAP, and Spring Health. "
    "If you do not know the answer, or if the user needs further help, tell them to submit a support ticket to support@technologylab.com or contact HR at hradp@leadpublicschools.org. "
    "If someone asks about student help, non-LEAD devices, curriculum, HR questions outside the Employee Manual, personal issues, or anything not in the official guides, politely say: "
    "'Sorry, I can only assist LEAD staff and teachers with official technology, HR, and benefits support for LEAD Public Schools.' "
    "Never guess, speculate, or provide unofficial advice. "
    "For security or unresolved tech issues, escalate to security@leadpublicschools.org or support@technologylab.com. "
    "Quick policy highlights: "
    "- Chromebook reset: Hold Refresh + Power. "
    "- MacBook first login: Use Addigy and your LEAD email. Temp password: FirstNameWeAreLEADers100%. "
    "- Staff passwords must be 16+ characters. "
    "- Submit tech support to support@technologylab.com; security issues to security@leadpublicschools.org. "
    "- Open enrollment is May 15–30, via ADP. "
    "- Health, dental, vision, and other benefits: see 2024 Benefits Presentation or contact HR. "
    "- Spring Health provides mental health support for staff and families. "
    "- For full employee policies (PTO, conduct, device use), always refer to the Employee Manual or HR."
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
