from flask import Flask, request, render_template
from transformers import T5Tokenizer, T5ForConditionalGeneration
import re

app = Flask(__name__)

# Load the T5 model and tokenizer
model_name = "t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Function to chunk long texts into smaller segments
def chunk_text(text, max_length=500):
    sentences = re.split(r'(?<=[.!?]) +', text)
    current_chunk = []
    current_length = 0
    chunks = []

    for sentence in sentences:
        sentence_length = len(tokenizer.encode(sentence, add_special_tokens=False))
        if current_length + sentence_length > max_length:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0
        current_chunk.append(sentence)
        current_length += sentence_length

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

# Function to summarize text using the T5 model
def summarize_text(text, model, tokenizer):
    chunks = chunk_text(text)
    summaries = []

    for chunk in chunks:
        input_ids = tokenizer.encode("summarize: " + chunk, return_tensors="pt", max_length=512, truncation=True)
        summary_ids = model.generate(input_ids, max_length=300, min_length=100, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summaries.append(summary)

    return " ".join(summaries)

@app.route("/", methods=["GET", "POST"])
def index():
    summary = None
    original_text = ""
    if request.method == "POST":
        original_text = request.form["news_paragraph"]
        summary = summarize_text(original_text, model, tokenizer)
    return render_template("index.html", summary=summary, original_text=original_text)

if __name__ == "__main__":
    app.run(debug=True)
