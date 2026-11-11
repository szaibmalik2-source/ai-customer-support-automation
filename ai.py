import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY", "").strip(),
    http_options=types.HttpOptions(timeout=60000),  # 60 seconds
)

# Pehla busy ho to agla khud try hoga
MODELS = ["gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash"]
RETRIES = 2

def ask_ai_safe(prompt):
    for model in MODELS:
        for attempt in range(1, RETRIES + 1):
            try:
                print(f"Calling {model} (attempt {attempt}/{RETRIES})...", flush=True)
                r = client.models.generate_content(model=model, contents=prompt)
                return r.text or ""
            except Exception as e:
                print("AI error:", str(e)[:200], flush=True)
                if attempt < RETRIES:
                    time.sleep(3)
    return ""