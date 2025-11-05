"""Document loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document


def load_text_documents(docs_path: Path) -> list[Document]:
    """Load all .txt documents from the supplied directory."""

    if not docs_path.exists():
        raise FileNotFoundError(f"The directory {docs_path} does not exist. Please create it and add your files.")

    loader = DirectoryLoader(
        path=str(docs_path),
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8", "autodetect_encoding": True},
        show_progress=True,
    )

    documents = loader.load()

    if not documents:
        raise FileNotFoundError(f"No .txt files found in {docs_path}. Please add your documents.")

    return documents


def documents_from_strings(texts: Iterable[str], *, metadata: dict | None = None) -> list[Document]:
    """Helper for constructing Document objects from raw strings."""

    base_metadata = metadata or {}
    return [Document(page_content=text, metadata=base_metadata) for text in texts]
