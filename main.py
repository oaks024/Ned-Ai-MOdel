"""Ingestion CLI — refresh the NED knowledge base.

Commands:
    python main.py scrape    # crawl the website + download admission PDFs
    python main.py build     # chunk + embed everything currently in data/raw
    python main.py refresh   # scrape, then rebuild embeddings from scratch
    python main.py stats     # show how many chunks are stored
    python main.py reset     # wipe the vector collection
"""
import argparse
import json
import os
import datetime
import time

from src.config import Config
from src.scraper import NEDScraper
from src.pdf_loader import extract_pdf_pages
from src.chunker import chunk_documents
from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore
from src.utils import get_logger

logger = get_logger("main")


def cmd_scrape(cfg: Config) -> dict:
    logger.info("Starting NED website scrape...")
    scraper = NEDScraper(
        cfg.ned_base_url, cfg.raw_data_path, max_pages=cfg.max_crawl_pages
    )
    pages, pdfs = scraper.crawl()
    snapshot = {
        "scraped_at": datetime.datetime.utcnow().isoformat(),
        "pages": pages,
        "pdfs": pdfs,
    }
    os.makedirs(cfg.raw_data_path, exist_ok=True)
    out = os.path.join(cfg.raw_data_path, "snapshot.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved {len(pages)} pages and {len(pdfs)} PDFs to {out}.")
    return snapshot


def cmd_build(cfg: Config, rebuild: bool = False) -> None:
    snapshot_path = os.path.join(cfg.raw_data_path, "snapshot.json")
    if not os.path.exists(snapshot_path):
        logger.error("snapshot.json missing. Run `python main.py scrape` first.")
        return
    with open(snapshot_path, encoding="utf-8") as f:
        snapshot = json.load(f)

    documents: list[dict] = []
    for p in snapshot.get("pages", []):
        if not p.get("text", "").strip():
            continue
        documents.append({
            "text": p["text"],
            "source_url": p["url"],
            "title": p.get("title") or p["url"],
            "doc_type": "html",
            "scraped_at": snapshot["scraped_at"],
        })
    for pdf in snapshot.get("pdfs", []):
        for page in extract_pdf_pages(
            pdf["path"], source_url=pdf["url"], title=pdf.get("title")
        ):
            documents.append({
                "text": page["text"],
                "source_url": page["source_url"],
                "title": page["title"],
                "page_number": page["page_number"],
                "file_name": page["file_name"],
                "doc_type": "pdf",
                "scraped_at": snapshot["scraped_at"],
            })

    logger.info(f"Loaded {len(documents)} raw documents. Chunking...")
    chunks = chunk_documents(documents, chunk_tokens=1000, overlap_tokens=150)
    logger.info(f"Built {len(chunks)} chunks. Embedding...")

    embedder = EmbeddingModel(cfg.embedding_model)
    store = VectorStore(cfg.chroma_db_path)
    if rebuild:
        logger.info("Resetting existing collection before rebuild.")
        store.reset()

    BATCH = 64
    ts = int(time.time() * 1000)
    for i in range(0, len(chunks), BATCH):
        batch = chunks[i:i + BATCH]
        texts = [c["text"] for c in batch]
        metas = [c["metadata"] for c in batch]
        embs = embedder.embed(texts)
        ids = [f"chunk_{ts}_{i + j}" for j in range(len(batch))]
        store.add(texts, embs, metas, ids=ids)
        logger.info(f"Indexed {min(i + BATCH, len(chunks))}/{len(chunks)}")

    logger.info(f"Done. Vector store now has {store.count()} chunks.")


def cmd_stats(cfg: Config) -> None:
    store = VectorStore(cfg.chroma_db_path)
    print(f"Total chunks indexed: {store.count()}")
    snapshot_path = os.path.join(cfg.raw_data_path, "snapshot.json")
    if os.path.exists(snapshot_path):
        with open(snapshot_path, encoding="utf-8") as f:
            snapshot = json.load(f)
        print(f"Last scrape: {snapshot.get('scraped_at')}")
        print(f"Pages scraped: {len(snapshot.get('pages', []))}")
        print(f"PDFs downloaded: {len(snapshot.get('pdfs', []))}")


def cmd_reset(cfg: Config) -> None:
    store = VectorStore(cfg.chroma_db_path)
    store.reset()
    print("Vector collection cleared.")


def main():
    parser = argparse.ArgumentParser(description="NED admission RAG ingestion CLI")
    parser.add_argument(
        "command", choices=["scrape", "build", "refresh", "stats", "reset"]
    )
    args = parser.parse_args()
    cfg = Config()

    if args.command == "scrape":
        cmd_scrape(cfg)
    elif args.command == "build":
        cmd_build(cfg, rebuild=False)
    elif args.command == "refresh":
        cmd_scrape(cfg)
        cmd_build(cfg, rebuild=True)
    elif args.command == "stats":
        cmd_stats(cfg)
    elif args.command == "reset":
        cmd_reset(cfg)


if __name__ == "__main__":
    main()
