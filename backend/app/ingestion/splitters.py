"""Text splitting utilities for ingestion."""

from __future__ import annotations

from typing import Iterable

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(
    documents: Iterable[Document],
    *,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> list[Document]:
    """Split documents into overlapping chunks.

    Defaults were chosen for student materials: 800-char chunks fit a
    paragraph or two at a time, and a 150-char overlap keeps sentences that
    straddle chunk boundaries intact — important for citation accuracy.
    """

    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )

    return splitter.split_documents(list(documents))
