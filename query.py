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


def search(question):
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

    return results


def generate_answer(question, results):

    context = "\n\n".join(
        f"""
Source: {source}
Section: {section}

{content}
"""
        for source, section, content, similarity in results
    )

    response = client.responses.create(
        model="gpt-5-mini",
        instructions="""
You are answering questions about the user's professional
experience and case studies.

Answer the user's question using ONLY the provided context.

If the context does not contain enough information to answer
the question, say that the information is not available.

Do not invent facts.
""",
        input=f"""
Question:
{question}

Context:
{context}
"""
    )

    return response.output_text


# question = input("\nAsk a question: ")

# results = search(question)

# print("\nRetrieved sections:\n")

# for source, section, content, similarity in results:

#     print("=" * 60)
#     print(f"Source: {source}")
#     print(f"Section: {section}")
#     print(f"Similarity: {similarity:.4f}")
#     print(content)


# answer = generate_answer(question, results)

# print("\n")
# print("=" * 60)
# print("ANSWER")
# print("=" * 60)
# print(answer)