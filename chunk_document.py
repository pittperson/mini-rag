from pathlib import Path


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
    print(f"Number of chunks: {len(chunks)}")

    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i + 1} ---")
        print(chunk)