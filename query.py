from openai import OpenAI
import psycopg


client = OpenAI()

DATABASE_URL = "postgresql://localhost/mini_rag"


def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )

    return response.data[0].embedding


question = input("\nAsk a question: ")

question_embedding = get_embedding(question)


with psycopg.connect(DATABASE_URL) as conn:

    results = conn.execute(
        """
        SELECT
            source,
            section,
            content,
            1 - (embedding <=> %s::vector) AS similarity
        FROM document_chunks
        ORDER BY embedding <=> %s::vector
        LIMIT 3;
        """,
        (
            question_embedding,
            question_embedding
        )
    ).fetchall()


print("\nMost relevant sections:\n")

for source, section, content, similarity in results:

    print("=" * 60)
    print(f"Source: {source}")
    print(f"Section: {section}")
    print(f"Similarity: {similarity:.4f}")
    print()
    print(content)