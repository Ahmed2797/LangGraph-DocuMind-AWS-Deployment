import hashlib
import os
import shutil
import sys
import uuid
from typing import Optional

import fitz
import pandas as pd
import pdfplumber
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.chat_model import get_embeddings
from src.config import get_settings
from src.constants import FAISS_DB_PATH, IMAGE_DIR, INDEX_NAME_CACHE_MEMORY, K
from src.exception import CustomException
from src.logger import logging
from src.vector import create_pincone_database

# FIX: this was `get_settings().pinecone_api_key` (the SECRET KEY, not the index name).
# Use whatever your settings call the index name. None / "" => FAISS mode.
PINECONE_INDEX_NAME = getattr(get_settings(), "pinecone_index_name", None)

FAISS_PROJECTS_ROOT = f"{FAISS_DB_PATH}_projects"
EMBED_BATCH = 100
MIN_IMAGE_SIDE = 120          # skip logos / icons / bullets
MAX_TABLE_CHARS = 6000        # Pinecone metadata limit is 40 KB per vector


def normalize_project_id(project_id) -> str:
    """ONE format everywhere: upload, retriever, FAISS folder, image folder, cache."""
    return str(project_id).lower().replace("_", "-")


def _splitter(chunk_size=500, chunk_overlap=50):
    return RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)


# ---------------------------- extraction ---------------------------------------
def _extract_pdf(path: str, project_id: str, splitter) -> list[Document]:
    """Text chunks, tables (markdown) and ONE image-record per page from a PDF."""
    source = os.path.basename(path)
    docs: list[Document] = []

    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text:
                for chunk in splitter.split_text(text):
                    docs.append(Document(
                        page_content=chunk,
                        metadata={"type": "text", "page": i, "source": source},
                    ))

            for table in page.extract_tables() or []:
                try:
                    if table and len(table) > 1:
                        df = pd.DataFrame(table[1:], columns=table[0])
                        md = df.to_markdown(index=False)[:MAX_TABLE_CHARS]
                        docs.append(Document(
                            page_content=md,
                            metadata={"type": "table", "page": i, "source": source},
                        ))
                except Exception as e:     # one bad table must not kill the upload
                    logging.warning(f"Table skipped on page {i} of {source}: {e}")

    img_dir = os.path.join(IMAGE_DIR, project_id)
    os.makedirs(img_dir, exist_ok=True)
    seen_xrefs: set[int] = set()           # same logo repeated on every page -> save once

    with fitz.open(path) as pdf_doc:
        for page_num in range(len(pdf_doc)):
            page = pdf_doc[page_num]
            context = page.get_text("text")[:500]
            files: list[str] = []

            for img_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                if xref in seen_xrefs:
                    continue
                try:
                    pix = fitz.Pixmap(pdf_doc, xref)
                    if pix.width < MIN_IMAGE_SIDE or pix.height < MIN_IMAGE_SIDE:
                        continue
                    if pix.n - pix.alpha >= 4:          # CMYK etc -> RGB
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                    img_name = f"p{page_num + 1}_img_{img_index}_{uuid.uuid4().hex[:6]}.png"
                    pix.save(os.path.join(img_dir, img_name))
                    pix = None
                    seen_xrefs.add(xref)
                    files.append(f"{project_id}/{img_name}")
                except Exception as e:
                    logging.warning(f"Image skipped (page {page_num + 1}, {source}): {e}")

            # ONE record per page (not one per image) so duplicates don't fill top-k
            if files and context.strip():
                docs.append(Document(
                    page_content=context,
                    metadata={
                        "type": "image",
                        "page": page_num + 1,
                        "source": source,
                        "file": files[0],      # backward compatible
                        "files": files,
                    },
                ))
    return docs


def _extract_text_file(path: str, splitter) -> list[Document]:
    source = os.path.basename(path)
    docs = TextLoader(path, encoding="utf-8", autodetect_encoding=True).load()
    chunks = splitter.split_documents(docs)
    for c in chunks:
        c.metadata.update({"type": "text", "source": source})
    return chunks


def _extract(path: str, project_id: str, splitter) -> list[Document]:
    if path.lower().endswith(".pdf"):
        return _extract_pdf(path, project_id, splitter)
    return _extract_text_file(path, splitter)


def _doc_id(d: Document) -> str:
    """Deterministic: re-uploading the same file overwrites instead of duplicating."""
    m = d.metadata
    raw = f"{m.get('source')}|{m.get('type')}|{m.get('page')}|{d.page_content}"
    return f"{m['type']}_{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:24]}"


# ---------------------------- semantic cache -----------------------------------
def clear_semantic_cache(project_id: str) -> bool:
    """Called automatically after every index/delete so old answers are never served."""
    ns = "cache-" + normalize_project_id(project_id)   # same prefix as retrieval_pipeline
    try:
        index = create_pincone_database(index_name=INDEX_NAME_CACHE_MEMORY)
        index.delete(delete_all=True, namespace=ns)
        logging.info(f"Semantic cache cleared for project {ns}.")
        return True
    except Exception as e:      # e.g. namespace does not exist yet
        logging.warning(f"Semantic cache clear skipped for {ns}: {e}")
        return False


# ---------------------------- FAISS backend ------------------------------------
def _faiss_dir(project_id: str) -> str:
    return os.path.join(FAISS_PROJECTS_ROOT, normalize_project_id(project_id))


class _EmptyRetriever:
    """Project with no uploads yet. Empty result -> graph goes to no_answer_found."""

    @staticmethod
    def similarity_search_with_score(query: str, k: int = K, **kwargs):
        return []


def _index_faiss(project_id: str, docs: list[Document], sources: set[str]) -> None:
    embeddings = get_embeddings()
    path = _faiss_dir(project_id)
    if os.path.exists(path):
        store = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
        try:   # re-upload of a changed file: drop its old chunks first
            old = [i for i, d in store.docstore._dict.items() if d.metadata.get("source") in sources]
            if old:
                store.delete(old)
        except Exception as e:
            logging.warning(f"FAISS old-chunk delete skipped: {e}")
        store.add_documents(docs)
    else:
        store = FAISS.from_documents(docs, embeddings)
    store.save_local(path)


# ---------------------------- Pinecone backend ---------------------------------
def _index_pinecone(project_id: str, docs: list[Document], sources: set[str]) -> None:
    index = create_pincone_database(index_name=PINECONE_INDEX_NAME)
    embeddings = get_embeddings()

    for src in sources:   # re-upload of a changed file: drop its old chunks first
        try:
            index.delete(filter={"source": {"$eq": src}}, namespace=project_id)
        except Exception as e:
            logging.warning(f"Pinecone delete-by-source skipped for {src}: {e}")

    for start in range(0, len(docs), EMBED_BATCH):
        batch = docs[start:start + EMBED_BATCH]
        vectors = embeddings.embed_documents([d.page_content for d in batch])
        index.upsert(
            vectors=[
                (_doc_id(d), vec, {**d.metadata, "text": d.page_content})
                for d, vec in zip(batch, vectors)
            ],
            namespace=project_id,
        )


# ---------------------------- public API ---------------------------------------
def index_files(project_id: str, file_paths: list[str], chunk_size=500, chunk_overlap=50) -> int:
    """Extract, chunk, embed and store files for one project. Returns chunks added."""
    try:
        pid = normalize_project_id(project_id)
        splitter = _splitter(chunk_size, chunk_overlap)
        docs: list[Document] = []
        for path in file_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Document not found: {path}")
            logging.info(f"Extracting: {path}")
            docs.extend(_extract(path, pid, splitter))

        if not docs:
            logging.warning("No extractable text found (scanned PDF?).")
            return 0

        sources = {d.metadata["source"] for d in docs if d.metadata.get("source")}
        if PINECONE_INDEX_NAME:
            _index_pinecone(pid, docs, sources)
        else:
            _index_faiss(pid, docs, sources)

        clear_semantic_cache(pid)     # documents changed -> cached answers are stale
        logging.info(f"Indexed {len(docs)} chunks for project {pid}.")
        return len(docs)
    except Exception as e:
        logging.error(f"Error indexing files: {e}")
        raise CustomException(e, sys)


def get_project_retriever(project_id: str, pinecone_index_name: Optional[str] = None):
    """
    Vector store limited to ONE project's documents.
    The graph calls `similarity_search_with_score(query, k=...)` on the result.
    Backend is chosen by the SAME setting used for indexing (no mismatch possible).
    """
    pid = normalize_project_id(project_id)
    embeddings = get_embeddings()
    index_name = pinecone_index_name or PINECONE_INDEX_NAME

    if index_name:
        index = create_pincone_database(index_name=index_name)
        logging.info(f"Pinecone index {index_name} ready (namespace={pid}).")
        return PineconeVectorStore(index=index, embedding=embeddings, namespace=pid)

    path = _faiss_dir(pid)
    if not os.path.exists(path):
        logging.info(f"FAISS database {path} does not exist (no uploads yet).")
        return _EmptyRetriever()

    store = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
    logging.info(f"FAISS database {path} loaded.")
    return store


def delete_index(project_id: str, pinecone_index_name: Optional[str] = None) -> None:
    """Remove a project's vectors, extracted images and cached answers."""
    pid = normalize_project_id(project_id)
    shutil.rmtree(os.path.join(IMAGE_DIR, pid), ignore_errors=True)
    index_name = pinecone_index_name or PINECONE_INDEX_NAME

    if index_name:
        index = create_pincone_database(index_name=index_name)
        try:
            index.delete(delete_all=True, namespace=pid)
        except Exception as e:      # namespace may not exist yet
            logging.warning(f"Pinecone namespace delete skipped: {e}")
    else:
        shutil.rmtree(_faiss_dir(pid), ignore_errors=True)
    clear_semantic_cache(pid)

    