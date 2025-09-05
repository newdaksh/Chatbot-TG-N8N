# app.py
import os
from flask import Flask, request, jsonify
import requests
from dotenv import load_dotenv
from typing import List, Dict, Any

load_dotenv()  # loads .env if present

HF_API_TOKEN = os.getenv("HF_API_TOKEN")
MODEL_ID = os.getenv("MODEL_ID", "openai/gpt-oss-20b")
HF_API_URL = f"https://api-inference.huggingface.co/models/{MODEL_ID}"
PORT = int(os.getenv("PORT", 8001))

if not HF_API_TOKEN:
    raise EnvironmentError("Set HF_API_TOKEN environment variable with your Hugging Face token.")

HEADERS = {
    "Authorization": f"Bearer {HF_API_TOKEN}",
    "Accept": "application/json",
    "Content-Type": "application/json"
}

app = Flask(__name__)

def call_hf_inference(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Send a POST request to Hugging Face Inference API and return parsed JSON.
    """
    resp = requests.post(HF_API_URL, headers=HEADERS, json=payload, timeout=60)
    # if model is busy, HF may return non-200 or an error JSON; handle gracefully
    try:
        data = resp.json()
    except ValueError:
        resp.raise_for_status()
        return {"error": "Invalid JSON from HF", "status_code": resp.status_code}

    if resp.status_code != 200:
        # forward whatever error info HF returned
        return {"error": data, "status_code": resp.status_code}
    return data

@app.route("/chat", methods=["POST"])
def chat():
    """
    Expect JSON body:
    {
      "message": "Hello, how are you?",
      "history": [
         {"role": "user", "content": "Hi"},
         {"role": "assistant", "content": "Hello! How can I assist?"}
      ]
    }

    Response:
    {
      "reply": "...",
      "raw": <raw response from HF>
    }
    """
    body = request.get_json(force=True, silent=True)
    if not body:
        return jsonify({"error": "JSON body required"}), 400

    message = body.get("message")
    history = body.get("history", [])

    if not message:
        return jsonify({"error": "Field 'message' is required."}), 400

    # Build payload for HF. For many models the simple {"inputs": prompt} works.
    # For chat-capable models you may pass a 'messages' list or structured input as model expects.
    # Here we build a simple chat-like prompt from history + incoming message.
    prompt_parts: List[str] = []
    for turn in history:
        role = turn.get("role", "user")
        content = turn.get("content", "")
        prompt_parts.append(f"{role}: {content}")
    prompt_parts.append(f"user: {message}")
    prompt_parts.append("assistant:")

    final_prompt = "\n".join(prompt_parts)

    payload = {
        "inputs": final_prompt,
        # optional parameters that many models accept:
        "parameters": {
            "max_new_tokens": 256,
            "temperature": 0.7,
            "top_k": 50,
            "top_p": 0.95,
            # "repetition_penalty": 1.1
        },
        "options": {
            "wait_for_model": True  # tell HF to wait for model to spin up if needed
        }
    }

    hf_resp = call_hf_inference(payload)

    # Different models / endpoints return different JSON shapes.
    # Common responses:
    # - [{"generated_text": "..."}]  (text-generation outputs)
    # - {"error": ...} on error
    # - {"generated_text": "..."} (sometimes single dict)
    reply_text = None

    # try a few common shapes:
    if isinstance(hf_resp, list) and len(hf_resp) > 0 and isinstance(hf_resp[0], dict):
        # e.g. [{"generated_text": "..."}]
        reply_text = hf_resp[0].get("generated_text") or hf_resp[0].get("text")
    elif isinstance(hf_resp, dict):
        if "generated_text" in hf_resp:
            reply_text = hf_resp.get("generated_text")
        elif "error" in hf_resp:
            return jsonify(hf_resp), 502
        else:
            # try to extract any text-like value
            for v in hf_resp.values():
                if isinstance(v, str) and len(v) > 0:
                    reply_text = v
                    break

    if reply_text is None:
        # fallback: return full raw hf response so client can inspect structure
        return jsonify({"reply": None, "raw": hf_resp})

    return jsonify({"reply": reply_text.strip(), "raw": hf_resp})


if __name__ == "__main__":
    print(f"Starting server on port {PORT}, forwarding to HF model: {MODEL_ID}")
    app.run(host="0.0.0.0", port=PORT)
