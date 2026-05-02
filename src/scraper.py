"""Polite, breadth-first crawler for the NED University website.

It starts at the base URL, visits internal pages, and prioritises pages whose
URL or anchor text mentions admission-related keywords. PDFs that look
admission-related are downloaded into ``data/raw/pdf``.
"""
import os
import time
import hashlib
import warnings
from collections import deque
from urllib.parse import urljoin, urlparse, urldefrag

import requests
from bs4 import BeautifulSoup
import urllib3

from src.utils import get_logger

# NED's https certificate is sometimes mis-configured. We fall back to
# unverified requests but suppress the noisy warning so the log stays readable.
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings("ignore", category=urllib3.exceptions.InsecureRequestWarning)

logger = get_logger("scraper")

ADMISSION_KEYWORDS = [
    "admission", "admissions", "undergraduate", "postgraduate",
    "bachelor", "master", "phd", "fee", "fees", "merit", "eligibility",
    "entry test", "entry-test", "entrytest", "prospectus", "apply",
    "documents", "schedule", "notice", "announcement", "intake",
]


class NEDScraper:
    def __init__(self, base_url: str, raw_dir: str, max_pages: int = 80, delay: float = 0.8):
        self.base_url = base_url.rstrip("/") + "/"
        self.domain = urlparse(self.base_url).netloc
        self.raw_dir = raw_dir
        self.max_pages = max_pages
        self.delay = delay

        self.html_dir = os.path.join(self.raw_dir, "html")
        self.pdf_dir = os.path.join(self.raw_dir, "pdf")
        os.makedirs(self.html_dir, exist_ok=True)
        os.makedirs(self.pdf_dir, exist_ok=True)

        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "NEDAdmissionAssistant/1.0 (+educational)"
        })

        self.visited: set[str] = set()
        self.pdfs_downloaded: set[str] = set()
        self.pages: list[dict] = []
        self.pdf_metadata: list[dict] = []

    # --------------- helpers ---------------
    def _is_internal(self, url: str) -> bool:
        netloc = urlparse(url).netloc
        return netloc == "" or netloc.endswith("neduet.edu.pk")

    def _looks_admission_related(self, *texts: str) -> bool:
        haystack = " ".join(t.lower() for t in texts if t)
        return any(k in haystack for k in ADMISSION_KEYWORDS)

    @staticmethod
    def _normalize(url: str) -> str:
        url, _ = urldefrag(url)
        return url.rstrip("/")

    def _download_pdf(self, url: str, anchor: str) -> str | None:
        if url in self.pdfs_downloaded:
            return None
        try:
            r = self.session.get(url, timeout=30, verify=False)
            r.raise_for_status()
        except Exception as e:
            logger.warning(f"PDF download failed: {url} -> {e}")
            return None

        digest = hashlib.md5(url.encode()).hexdigest()[:10]
        original = os.path.basename(urlparse(url).path) or "file.pdf"
        if not original.lower().endswith(".pdf"):
            original = "file.pdf"
        safe = "".join(ch for ch in original if ch.isalnum() or ch in "._-")[:60]
        path = os.path.join(self.pdf_dir, f"{digest}_{safe}")
        with open(path, "wb") as f:
            f.write(r.content)
        self.pdfs_downloaded.add(url)
        logger.info(f"Downloaded PDF: {url}")
        self.pdf_metadata.append({"url": url, "path": path, "title": anchor or original})
        return path

    def _parse_html(self, html: str, base: str) -> tuple[str, str, list[tuple[str, str]]]:
        soup = BeautifulSoup(html, "lxml")
        title = (soup.title.string if soup.title and soup.title.string else "").strip()

        # Remove non-content tags so the extracted text is mostly real prose.
        for tag in soup(["script", "style", "nav", "footer", "header", "form", "noscript"]):
            tag.decompose()
        text = soup.get_text("\n", strip=True)

        links: list[tuple[str, str]] = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            anchor = (a.get_text() or "").strip()
            if not href or href.startswith(("javascript:", "mailto:", "tel:")):
                continue
            full = self._normalize(urljoin(base, href))
            links.append((full, anchor))
        return title, text, links

    # --------------- main loop ---------------
    def crawl(self) -> tuple[list[dict], list[dict]]:
        queue: deque[tuple[str, str, int]] = deque()
        queue.append((self.base_url, "NED University Home", 0))

        while queue and len(self.visited) < self.max_pages:
            url, anchor, depth = queue.popleft()
            url = self._normalize(url)
            if url in self.visited or not self._is_internal(url):
                continue
            self.visited.add(url)

            if url.lower().endswith(".pdf"):
                if self._looks_admission_related(url, anchor):
                    self._download_pdf(url, anchor)
                continue

            try:
                r = self.session.get(url, timeout=30, verify=False)
                if "text/html" not in r.headers.get("Content-Type", ""):
                    continue
                r.raise_for_status()
            except Exception as e:
                logger.warning(f"Fetch failed: {url} -> {e}")
                continue

            title, text, links = self._parse_html(r.text, url)
            self.pages.append({"url": url, "title": title or anchor, "text": text})
            logger.info(f"Crawled ({len(self.visited)}/{self.max_pages}): {url}")
            time.sleep(self.delay)

            for link, link_anchor in links:
                if link in self.visited or not self._is_internal(link):
                    continue
                # Always queue admission-relevant links. Also follow the home
                # page's links once so we discover the admissions sub-tree.
                if self._looks_admission_related(link, link_anchor) or depth == 0:
                    if link.lower().endswith(".pdf"):
                        # Only fetch admission-related PDFs to keep noise low.
                        if self._looks_admission_related(link, link_anchor):
                            self._download_pdf(link, link_anchor)
                    else:
                        queue.append((link, link_anchor, depth + 1))

        logger.info(
            f"Crawl complete. Pages={len(self.pages)} PDFs={len(self.pdf_metadata)}"
        )
        return self.pages, self.pdf_metadata
