from pathlib import Path

from dotenv import load_dotenv

from document_indexer import DocumentIndexer


def main() -> None:
    load_dotenv()
    repo_root = Path(__file__).resolve().parents[1]
    default_doc = repo_root / "docs" / "cracking_the_coding_interview.pdf"

    indexer = DocumentIndexer()
    indexer.create_collection(recreate=True)
    indexer.process_and_index([str(default_doc)])


if __name__ == "__main__":
    main()
