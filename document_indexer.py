import os
import uuid

from dotenv import load_dotenv
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, HnswConfigDiff, PointStruct, VectorParams


class DocumentIndexer:
    def __init__(self) -> None:
        self.embeddings = HuggingFaceBgeEmbeddings(
            model_name=os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5"),
            model_kwargs={
                "device": os.getenv("EMBED_DEVICE", "cpu"),
                "trust_remote_code": False,
            },
            encode_kwargs={"normalize_embeddings": True},
            cache_folder=os.getenv("HF_HOME"),
        )
        self.qdrant = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
        )
        self.collection_name = os.getenv("QDRANT_COLLECTION", "projects_rag")
        self.upsert_batch_size = int(os.getenv("QDRANT_UPSERT_BATCH_SIZE", "128"))
        self.chunk_size = int(os.getenv("RAG_CHUNK_SIZE", "256"))
        self.chunk_overlap = int(os.getenv("RAG_CHUNK_OVERLAP", "30"))

    def create_collection(self, *, recreate: bool = False) -> None:
        existing = {collection.name for collection in self.qdrant.get_collections().collections}
        if self.collection_name in existing:
            if recreate:
                self.qdrant.delete_collection(collection_name=self.collection_name)
            else:
                print(f"Collection already exists: {self.collection_name}")
                return

        self.qdrant.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            hnsw_config=HnswConfigDiff(m=16, ef_construct=200),
        )
        print(f"Collection created: {self.collection_name}")

    def _stable_point_id(self, project_name: str, index: int) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{project_name}:{index}"))

    def _page_number(self, chunk) -> int | None:
        page = chunk.metadata.get("page")
        if isinstance(page, int):
            return page + 1
        return None

    def _load_document(self, doc_path: str):
        ext = os.path.splitext(doc_path)[1].lower()
        if ext == ".docx":
            return Docx2txtLoader(doc_path).load()
        if ext == ".pdf":
            return PyPDFLoader(doc_path).load()
        raise ValueError(f"Unsupported document type: {doc_path}")

    def process_and_index(self, doc_paths: list[str]) -> None:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", "! ", "? ", " "],
        )
        all_points: list[PointStruct] = []

        for doc_path in doc_paths:
            docs = self._load_document(doc_path)
            project_name = os.path.splitext(os.path.basename(doc_path))[0]
            chunks = text_splitter.split_documents(docs)
            texts = [chunk.page_content for chunk in chunks]
            embeddings = self.embeddings.embed_documents(texts)

            for index, (chunk, embedding) in enumerate(zip(chunks, embeddings, strict=False)):
                point_id = self._stable_point_id(project_name, index)
                payload = {
                    "text": chunk.page_content,
                    "project_name": project_name,
                    "doc_id": f"{project_name}_{index}",
                }
                page = self._page_number(chunk)
                if page is not None:
                    payload["page"] = page

                all_points.append(
                    PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload=payload,
                    )
                )

        for start in range(0, len(all_points), self.upsert_batch_size):
            batch = all_points[start : start + self.upsert_batch_size]
            self.qdrant.upsert(collection_name=self.collection_name, points=batch)
            print(
                f"Uploaded batch {start // self.upsert_batch_size + 1} "
                f"({len(batch)} chunks)"
            )
        print(f"Indexed {len(all_points)} chunks from {len(doc_paths)} documents")


if __name__ == "__main__":
    load_dotenv()
    indexer = DocumentIndexer()
    indexer.create_collection(recreate=True)
    indexer.process_and_index(
        [os.getenv("INDEX_DOC_PATH", "docs/cracking_the_coding_interview.pdf")]
    )
