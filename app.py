from flask import Flask, request, jsonify
from flask_cors import CORS
import traceback
from vertexai.preview.generative_models import GenerativeModel

app = Flask(__name__)
CORS(app)

MODEL = "gemini-1.0-pro"

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
        model = GenerativeModel(MODEL)
        chat = model.start_chat()
        response = chat.send_message(user_query)
        answer = response.text if hasattr(response, "text") else str(response)
        print("Gemini Vertex AI response:", answer)
        return jsonify({"response": answer})
    except Exception as e:
        print("Error:", e)
        traceback.print_exc()
        return jsonify({"response": "MyLEAD is currently unavailable. Please try again later."}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
