import os
from flask import Flask, request, jsonify
from groq import Groq

PORT = int(os.getenv("PORT", 8001))
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

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

    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    for turn in history:
        messages.append({"role": turn.get("role", "user"), "content": turn.get("content", "")})
    messages.append({"role": "user", "content": message})

    # non-streaming call (safer on Render)
    resp = client.chat.completions.create(
        model="llama3-8b-8192",          # choose Groq model you have access to
        messages=messages,
        temperature=0.7,
        max_completion_tokens=512,
        stream=False
    )

    # parse based on returned structure
    try:
        content = resp["choices"][0]["message"]["content"]
    except Exception:
        # fallback: return raw
        return {"reply": None, "raw": resp}, 200

    return {"reply": content.strip(), "raw": resp}, 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
