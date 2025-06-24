from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import google.generativeai as genai

app = Flask(__name__)
CORS(app)

# Try to get the API key from environment
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY or GEMINI_API_KEY environment variable not set")
genai.configure(api_key=GOOGLE_API_KEY)

MODEL = "gemini-pro"  # Use 'gemini-pro' for google-generativeai

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/ask", methods=["POST"])
def ask():
    import traceback
    try:
        data = request.get_json()
        user_query = data.get("query", "")
        if not user_query:
            return jsonify({"response": "Please enter a question."})

        print("Got user query:", user_query)
        model = genai.GenerativeModel(MODEL)
        response = model.generate_content(user_query)
        print("Gemini API raw response:", response)
        answer = response.text.strip() if hasattr(response, "text") else str(response)
        print("Gemini extracted answer:", answer)
        return jsonify({"response": answer})
    except Exception as e:
        print("Error:", e)
        traceback.print_exc()
        return jsonify({"response": "MyLEAD is currently unavailable. Please try again later."}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
