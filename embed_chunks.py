from pathlib import Path
from openai import OpenAI

client = OpenAI()


def chunk_text(text, chunk_size=50):
    words = text.split()

    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks


documents_dir = Path("documents")

for file in documents_dir.glob("*.md"):

    text = file.read_text(encoding="utf-8")

    chunks = chunk_text(text)

    print(f"Document: {file.name}")
    print(f"Chunks: {len(chunks)}")

    for i, chunk in enumerate(chunks):

        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=chunk
        )

        embedding = response.data[0].embedding

        print(f"\n--- Chunk {i + 1} ---")
        print(chunk)

        print(f"\nVector dimensions: {len(embedding)}")
        print(f"First 10 numbers: {embedding[:10]}")