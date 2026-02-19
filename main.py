
from __future__ import annotations

try:
    from chromadb import PersistentClient
    from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
except ImportError:
    PersistentClient = None
    Documents = None
    EmbeddingFunction = None
    Embeddings = None

from embedding import M3E_MODEL_ID, encode as encode_texts


class M3EEmbeddingFunction(EmbeddingFunction):
    """ChromaDB embedding function using moka-ai/m3e-base (shared encoder from embedding.py)."""

    def __call__(self, input: Documents) -> Embeddings:
        emb = encode_texts(list(input), normalize=False)
        return [emb[i].tolist() for i in range(len(input))]


def get_vector_store(
    persist_directory: str = "./data/vector_db",
    collection_name: str = "m3e_docs",
) -> "VectorStore":
    """Create or load a persistent vector store with M3E embeddings."""
    if PersistentClient is None:
        raise RuntimeError("chromadb is required: pip install chromadb")
    client = PersistentClient(path=persist_directory)
    embedding_fn = M3EEmbeddingFunction()
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )
    return VectorStore(collection=collection, embedding_fn=embedding_fn)


class VectorStore:
    """Vector store backed by ChromaDB and moka-ai/m3e-base."""

    def __init__(self, *, collection, embedding_fn: M3EEmbeddingFunction):
        self._collection = collection
        self._embedding_fn = embedding_fn

    def add(
        self,
        documents: list[str],
        ids: list[str] | None = None,
        metadatas: list[dict] | None = None,
    ) -> None:
        """Add documents; they will be embedded with M3E. ids/metadatas optional."""
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(documents))]
        # ChromaDB requires each metadata dict to be non-empty when metadatas is provided
        kwargs: dict = {"documents": documents, "ids": ids}
        if metadatas is not None and all(metadatas):
            kwargs["metadatas"] = metadatas
        self._collection.add(**kwargs)

    def search(
        self,
        query: str,
        n_results: int = 5,
        where: dict | None = None,
    ) -> dict:
        """Return similar documents: ids, documents, metadatas, distances."""
        return self._collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where,
        )

    def count(self) -> int:
        """Return number of documents in the collection."""
        return self._collection.count()


def main():
    # Create persistent vector DB (data under ./data/vector_db)
    store = get_vector_store()


if __name__ == "__main__":
    main()
