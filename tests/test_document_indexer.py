from langchain_core.documents import Document

from document_indexer import DocumentIndexer


class FakeLoader:
    def __init__(self, path):
        self.path = path

    def load(self):
        return [Document(page_content="Alpha project details")]


class FakeEmbeddings:
    def embed_documents(self, texts):
        return [[0.1] * 384 for _ in texts]


class FakeQdrant:
    def __init__(self):
        self.upserts = []
        self.deleted = []
        self.collections = []

    def upsert(self, **kwargs):
        self.upserts.append(kwargs)

    def get_collections(self):
        return type("Collections", (), {"collections": self.collections})()

    def delete_collection(self, **kwargs):
        self.deleted.append(kwargs)


def test_process_and_index_builds_qdrant_points(monkeypatch):
    indexer = DocumentIndexer.__new__(DocumentIndexer)
    indexer.embeddings = FakeEmbeddings()
    indexer.qdrant = FakeQdrant()
    indexer.collection_name = "projects_rag"
    indexer.upsert_batch_size = 128
    indexer.chunk_size = 256
    indexer.chunk_overlap = 30

    monkeypatch.setattr("document_indexer.Docx2txtLoader", FakeLoader)

    indexer.process_and_index(["data/project_alpha.docx"])

    assert indexer.qdrant.upserts
    point = indexer.qdrant.upserts[0]["points"][0]
    assert point.payload["project_name"] == "project_alpha"
    assert point.id


def test_load_document_uses_pdf_loader(monkeypatch):
    indexer = DocumentIndexer.__new__(DocumentIndexer)

    monkeypatch.setattr("document_indexer.PyPDFLoader", FakeLoader)

    docs = indexer._load_document("docs/cracking_the_coding_interview.pdf")

    assert docs[0].page_content == "Alpha project details"


def test_create_collection_recreates_when_requested():
    indexer = DocumentIndexer.__new__(DocumentIndexer)
    qdrant = FakeQdrant()
    qdrant.collections = [type("Collection", (), {"name": "projects_rag"})()]
    indexer.qdrant = qdrant
    indexer.collection_name = "projects_rag"

    def create_collection(**kwargs):
        qdrant.created = kwargs

    qdrant.create_collection = create_collection

    indexer.create_collection(recreate=True)

    assert qdrant.deleted[0]["collection_name"] == "projects_rag"
