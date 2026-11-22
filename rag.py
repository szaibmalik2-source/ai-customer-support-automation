# rag.py
import csv
import time
import chromadb
from pypdf import PdfReader
from ai import ask_ai_safe

def load_pdf(path):
    reader = PdfReader(path)
    return " ".join(page.extract_text() or "" for page in reader.pages)

def chunk_text(text, size=400, overlap=50):
    words = text.split()
    return [" ".join(words[i:i + size]) for i in range(0, len(words), size - overlap)]

# Build the knowledge base
chunks = chunk_text(load_pdf("company documents.pdf"))
db = chromadb.Client()
collection = db.create_collection("docs")
collection.add(documents=chunks, ids=[f"chunk{i}" for i in range(len(chunks))])

def answer(question):
    found = collection.query(query_texts=[question], n_results=3)
    context = "\n\n".join(found["documents"][0])
    prompt = ("Answer the question using ONLY the context below. "
              "If the answer is not in the context, say \"I don't know.\"\n\n"
              "Context:\n" + context + "\n\nQuestion: " + question)
    return ask_ai_safe(prompt)

tests = [
    # Answers ARE in the document (7)
    ("How many leave days do employees get?", "in"),
    ("How many sick days do employees get?", "in"),
    ("What are the working hours?", "in"),
    ("When is salary paid?", "in"),
    ("What is the notice period for resigning?", "in"),
    ("How many days can employees work from home?", "in"),
    ("How long is the probation period?", "in"),
    # Answers are NOT in the document (3)
    ("Who is the CEO of the company?", "out"),
    ("What is the company's stock price?", "out"),
    ("Does the company provide free lunch?", "out"),
]

with open("rag_results.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Question", "Type", "AI Answer"])
    for q, kind in tests:
        a = answer(q)
        print(f"\n[{kind}] Q: {q}\nA: {a}")
        writer.writerow([q, kind, a])
        time.sleep(8)  # pause so the free API limit isn't hit