import json
import pandas as pd
from ai import ask_ai_safe

FENCE = chr(96) * 3

def clean(text):
    return text.strip().removeprefix(FENCE + "json").removesuffix(FENCE).strip()

rows = pd.read_csv("form_submissions.csv")
results = []

for _, r in rows.iterrows():
    message = str(r["message"]).strip()

    if message == "" or message.lower() == "nan":
        results.append({**r.to_dict(), "category": "empty", "reply": ""})
        print("Processed:", r["name"], "-> empty")
        continue

    prompt = ("You are a support assistant. Return ONLY valid JSON with keys "
              "'category' (billing / technical / general) and 'reply' (short, polite).\n"
              "Message: " + message)
    try:
        data = json.loads(clean(ask_ai_safe(prompt)))
    except Exception:
        data = {"category": "error", "reply": ""}

    print("Processed:", r["name"], "->", data.get("category"))
    results.append({**r.to_dict(), **data})

pd.DataFrame(results).to_csv("results.csv", index=False)
print("Done:", len(results), "rows processed")