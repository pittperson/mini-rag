from pathlib import Path

from openai import OpenAI
import psycopg
import yaml


# --------------------------------------------------
# Configuration
# --------------------------------------------------

client = OpenAI()

DATABASE_URL = "postgresql://localhost/mini_rag"

DOCUMENTS_DIR = Path("../chrislies/content/case-studies")


# --------------------------------------------------
# Parse YAML frontmatter
# --------------------------------------------------

def parse_frontmatter(text):
  """
  Extract YAML frontmatter from a Markdown document.

  Expected format:

  ---
  title: "Example"
  date: 2025-01-01
  tags: [tag1, tag2]
  ---

  Returns:
      metadata: dictionary containing the frontmatter
      content: Markdown content after the frontmatter
  """

  if not text.startswith("---"):
      return {}, text

  parts = text.split("---", 2)

  if len(parts) < 3:
      raise ValueError("Invalid frontmatter format")

  frontmatter_text = parts[1]
  markdown_content = parts[2].lstrip()

  metadata = yaml.safe_load(frontmatter_text)

  if metadata is None:
      metadata = {}

  return metadata, markdown_content

# --------------------------------------------------
# Parse H2 sections
# --------------------------------------------------

def parse_sections(markdown, title):
    """
    Split Markdown into sections based on H2 headings.

    Example:

    ## Situation

    Some content.

    ## Action

    More content.

    Returns a list of dictionaries.
    """

    lines = markdown.splitlines()

    sections = []

    current_section = None
    current_content = []

    for line in lines:

        stripped = line.strip()

        # Handle:
        # ## Situation
        # **## Situation**
        if stripped.startswith("## "):

            # Save previous section
            if current_section is not None:

                content = "\n".join(current_content).strip()

                if content:
                    sections.append({
                        "title": title,
                        "section": current_section,
                        "content": content
                    })

            current_section = stripped[3:].strip()
            current_content = []

        elif stripped.startswith("**## ") and stripped.endswith("**"):

            # Save previous section
            if current_section is not None:

                content = "\n".join(current_content).strip()

                if content:
                    sections.append({
                        "title": title,
                        "section": current_section,
                        "content": content
                    })

            current_section = stripped[6:-2].strip()
            current_content = []

        else:
            current_content.append(line)

    # Save final section
    if current_section is not None:

        content = "\n".join(current_content).strip()

        if content:
            sections.append({
                "title": title,
                "section": current_section,
                "content": content
            })

    return sections

# --------------------------------------------------
# Create embedding text
# --------------------------------------------------

def create_embedding_text(metadata, section):
    """
    Create the contextual text that will be sent
    to the OpenAI embedding model.
    """

    tags = metadata.get("tags", [])

    tags_text = ", ".join(tags)

    return f"""
Case Study: {metadata.get("title", "")}

Summary:
{metadata.get("summary", "")}

Role:
{metadata.get("role", "")}

Organization:
{metadata.get("org", "")}

Period:
{metadata.get("period", "")}

Tags:
{tags_text}

Section:
{section["section"]}

Content:
{section["content"]}
""".strip()



# --------------------------------------------------
# Create OpenAI embedding
# --------------------------------------------------

def get_embedding(text):

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )

    return response.data[0].embedding

# --------------------------------------------------
# Database connection
# --------------------------------------------------

def get_connection():

    return psycopg.connect(DATABASE_URL)

# --------------------------------------------------
# Ingest documents
# --------------------------------------------------

def ingest_documents():

    files = list(DOCUMENTS_DIR.glob("*.md"))

    if not files:
        print("No Markdown files found.")
        return

    with get_connection() as conn:

        for file in files:

            print()
            print("=" * 60)
            print(f"Processing: {file.name}")
            print("=" * 60)

            # ------------------------------------------
            # Read Markdown
            # ------------------------------------------

            text = file.read_text(encoding="utf-8")

            # ------------------------------------------
            # Parse frontmatter
            # ------------------------------------------

            metadata, markdown = parse_frontmatter(text)

            print(f"Title: {metadata.get('title')}")
            print(f"Organization: {metadata.get('org')}")
            print(f"Role: {metadata.get('role')}")
            print(f"Period: {metadata.get('period')}")

            # ------------------------------------------
            # Skip drafts
            # ------------------------------------------

            if metadata.get("draft", False):

                print("Skipping draft.")

                continue

            # ------------------------------------------
            # Parse H2 sections
            # ------------------------------------------

            sections = parse_sections(
                markdown,
                metadata.get("title", file.stem)
            )

            print(f"Sections found: {len(sections)}")

            # ------------------------------------------
            # Process each section
            # ------------------------------------------

            for section in sections:

                print(f"  Embedding: {section['section']}")

                embedding_text = create_embedding_text(
                    metadata,
                    section
                )

                embedding = get_embedding(
                    embedding_text
                )

                # --------------------------------------
                # Insert into PostgreSQL
                # --------------------------------------

                conn.execute(
                    """
                    INSERT INTO document_chunks (
                        source,
                        title,
                        date,
                        summary,
                        role,
                        org,
                        period,
                        tags,
                        draft,
                        section,
                        content,
                        embedding
                    )
                    VALUES (
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s
                    )
                    """,
                    (
                        file.name,
                        metadata.get("title"),
                        metadata.get("date"),
                        metadata.get("summary"),
                        metadata.get("role"),
                        metadata.get("org"),
                        metadata.get("period"),
                        metadata.get("tags", []),
                        metadata.get("draft", False),
                        section["section"],
                        section["content"],
                        embedding
                    )
                )

        conn.commit()

    print()
    print("=" * 60)
    print("INGESTION COMPLETE")
    print("=" * 60)


# --------------------------------------------------
# Run
# --------------------------------------------------

if __name__ == "__main__":
    ingest_documents()

