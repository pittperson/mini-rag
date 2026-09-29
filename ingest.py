from pathlib import Path

from openai import OpenAI
import psycopg


client = OpenAI()

DATABASE_URL = "postgresql://localhost/mini_rag"


def parse_markdown(text):
    lines = text.splitlines()

    title = ""
    current_section = None
    current_content = []

    chunks = []

    for line in lines:

        # H1 = document title
        if line.startswith("# ") and not line.startswith("## "):
            title = line[2:].strip()

        # if line.startswith("title: ") and not line.startswith("## "):
        #             title = line[6:].strip().replace('"','')

        # H2 = new section
        elif line.startswith("## "):

            # Save previous section
            if current_section:
                chunks.append({
                    "title": title,
                    "section": current_section,
                    "content": "\n".join(current_content).strip()
                })

            current_section = line[3:].strip()
            current_content = []

        # Normal content
        else:
            current_content.append(line)

    # Save final section
    if current_section:
        chunks.append({
            "title": title,
            "section": current_section,
            "content": "\n".join(current_content).strip()
        })

    return chunks


def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )

    return response.data[0].embedding


def get_connection():
    return psycopg.connect(DATABASE_URL)


documents_dir = Path("documents")


with get_connection() as conn:

    for file in documents_dir.glob("*.md"):

        print(f"\nProcessing: {file.name}")

        text = file.read_text(encoding="utf-8")

        chunks = parse_markdown(text)

        print(f"Found {len(chunks)} sections")

        for chunk in chunks:

            # Create the text that will actually be embedded.
            embedding_text = (
                f"{chunk['title']}\n\n"
                f"{chunk['section']}\n\n"
                f"{chunk['content']}"
            )

            print(f"Embedding: {chunk['section']}")

            embedding = get_embedding(embedding_text)

            conn.execute(
                """
                INSERT INTO document_chunks
                    (source, section, content, embedding)
                VALUES
                    (%s, %s, %s, %s)
                """,
                (
                    file.name,
                    chunk["section"],
                    chunk["content"],
                    embedding
                )
            )

    conn.commit()

print("\nIngestion complete!")