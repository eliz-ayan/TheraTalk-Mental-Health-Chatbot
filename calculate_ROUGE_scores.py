import pandas as pd
import numpy as np
import re
from sentence_transformers import SentenceTransformer, util
import evaluate


df = pd.read_csv("fine_tuned_model_responses.csv")
with open("CBT_Transcripts.txt", "r", encoding="utf-8") as f:
    cbt_text = f.read()


transcript_chunks = re.split(r'\n\s*\n|(?<=\.)\s+(?=[A-Z])', cbt_text)
transcript_chunks = [chunk.strip() for chunk in transcript_chunks if len(chunk.strip()) > 30]


model = SentenceTransformer("all-MiniLM-L6-v2")
transcript_embeddings = model.encode(transcript_chunks, convert_to_tensor=True)


reference_texts = []
for generated in df["generated_response"]:
    emb = model.encode(generated, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(emb, transcript_embeddings)[0]
    best_idx = int(np.argmax(scores))
    reference_texts.append(transcript_chunks[best_idx])

df["reference_response"] = reference_texts


rouge = evaluate.load("rouge")
scores = [rouge.compute(predictions=[gen], references=[ref]) for gen, ref in zip(df["generated_response"], df["reference_response"])]

df["rouge-1"] = [r["rouge1"] for r in scores]
df["rouge-2"] = [r["rouge2"] for r in scores]
df["rouge-L"] = [r["rougeL"] for r in scores]


df.to_csv("rouge_evaluated_responses2.csv", index=False)
print("✅ Saved: rouge_evaluated_responses.csv")
