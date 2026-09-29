from pathlib import Path

import numpy as np
from openai import OpenAI

client = OpenAI()


def chunk_text(text, chunk_size=50):
    words = text.split()

    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks


def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )

    return response.data[0].embedding


def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)

    return np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )


# -----------------------------------
# Read documents
# -----------------------------------

documents_dir = Path("documents")

chunks = []

for file in documents_dir.glob("*.md"):

    text = file.read_text(encoding="utf-8")

    document_chunks = chunk_text(text)

    for chunk in document_chunks:

        embedding = get_embedding(chunk)

        chunks.append({
            "text": chunk,
            "embedding": embedding,
            "source": file.name
        })


# -----------------------------------
# Ask a question
# -----------------------------------

question = input("\nAsk a question: ")

question_embedding = get_embedding(question)


# -----------------------------------
# Compare question to every chunk
# -----------------------------------

results = []

for chunk in chunks:

    similarity = cosine_similarity(
        question_embedding,
        chunk["embedding"]
    )

    results.append({
        "similarity": similarity,
        "text": chunk["text"],
        "source": chunk["source"]
    })


# -----------------------------------
# Sort by similarity
# -----------------------------------

results.sort(
    key=lambda x: x["similarity"],
    reverse=True
)


# -----------------------------------
# Display results
# -----------------------------------

print("\nMost relevant chunks:\n")

for result in results[:3]:

    print(f"Similarity: {result['similarity']:.4f}")
    print(f"Source: {result['source']}")
    print(result["text"])
    print("-" * 60)