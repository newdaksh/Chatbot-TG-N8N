import os
from flask import Flask, request, jsonify
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

PORT = int(os.getenv("PORT", 8001))
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "llama3-8b-8192")

if not GROQ_API_KEY:
    raise EnvironmentError("Set GROQ_API_KEY env var")

client = Groq(api_key=GROQ_API_KEY)

app = Flask(__name__)

@app.route("/", methods=["GET"])
def health():
    return {"status": "ok"}, 200

@app.route("/chat", methods=["POST"])
def chat():
    body = request.get_json(force=True, silent=True) or {}
    message = body.get("message")
    history = body.get("history", [])

    if not message:
        return {"error": "Field 'message' required"}, 400

    messages = [{
        "role": "system",
        "content": (
            "You are a helpful assistant. Answer the user's question in 40-50 words maximum. "
            "Be concise and do not exceed this limit to save tokens."
        )
    }]
    for turn in history:
        messages.append({
            "role": turn.get("role", "user"),
            "content": turn.get("content", "")
        })
    messages.append({"role": "user", "content": message})

    # Non-streaming call (safer on Render)
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.7,
        max_completion_tokens=512,
        stream=False
    )

    # Convert resp to dict if it's a pydantic/custom object
    if hasattr(resp, "model_dump"):
        resp_dict = resp.model_dump()
    elif hasattr(resp, "to_dict"):
        resp_dict = resp.to_dict()
    else:
        resp_dict = resp

    # Try to extract assistant content
    try:
        content = resp_dict["choices"][0]["message"]["content"]
    except Exception:
        # If we can't extract, return raw so client can inspect
        return {"reply": None, "raw": resp_dict}, 200

    # --- IMPORTANT: return only 'reply' to avoid duplicate output in client ---
    return {"reply": content.strip()}, 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
