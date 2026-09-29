from pathlib import Path

documents_dir = Path("documents")

for file in documents_dir.glob("*.md"):
    print(f"Found: {file}")
    
    text = file.read_text(encoding="utf-8")
    
    print(f"Characters: {len(text)}")
    print()
    print(text[:1000])