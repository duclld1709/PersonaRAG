"Document loading utilities."

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from langchain_core.documents import Document
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, Docx2txtLoader

def load_documents(docs_path: Path) -> list[Document]:
    """Load all .txt, .pdf, .docx from the supplied directory"""

    if not docs_path.exits():
        raise FileNotFoundError(f"The directory {docs_path} does not exitst. Please create it and add your files.")
    
    loaders_dict = {".pdf": PyPDFLoader, ".docx": Docx2txtLoader}
    documents = list()

    for ext, loader in loaders_dict.items():
        loader = DirectoryLoader(
            path=str(docs_path),
            glob=f"**/*{ext}",
            loader_cls=loader
        )
        documents.extend(loader.load())
    
    if not documents:
        raise FileNotFoundError(f"No files found in {docs_path}. Please add your documents.")
    
    return documents

