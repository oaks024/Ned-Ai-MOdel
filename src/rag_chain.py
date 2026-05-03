"""Retrieval-Augmented Generation pipeline.

Flow:
  1. Embed the user's question with the local embedding model.
  2. Retrieve the top-k most relevant chunks from ChromaDB.
  3. Build a strict system+user prompt that forces the model to answer
     ONLY from retrieved context. Mix in the user's profile (memory) for
     personalization, but never as factual admission data.
  4. Call the model via LiteLLM (configured for Groq).
  5. Return the answer plus a deduplicated list of source URLs.
"""
from litellm import completion

from src.config import Config
from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore
from src.memory import SessionMemory
from src.utils import get_logger

logger = get_logger("rag")

SYSTEM_PROMPT = (
    "You are an official-style NED admission information assistant. "
    "You answer questions only using the provided official NED University context. "
    "If the answer is not present in the context, say clearly: "
    "'I could not find this in the official NED data.' "
    "Do not guess. Do not invent dates, fees, eligibility rules, or admission criteria. "
    "Always include source links when available. Keep answers clear, structured, and "
    "student-friendly. For time-sensitive information (deadlines, fee deposits, schedules), "
    "remind the student to confirm with the NED admission office."
)

NO_CONTEXT_MESSAGE = (
    "I could not find this in the official NED data. "
    "Please run the ingestion pipeline (`python main.py refresh`) or visit "
    "https://www.neduet.edu.pk/ for current information."
)


class RAGChain:
    def __init__(self, config: Config):
        self.config = config
        self.embedder = EmbeddingModel(config.embedding_model)
        self.store = VectorStore(config.chroma_db_path)
        self.memory = SessionMemory(config.memory_db_path)
        self.model = config.litellm_model
        self.fallbacks = config.litellm_fallbacks

    # --------------- formatting helpers ---------------
    @staticmethod
    def _format_context(chunks: list[dict]) -> str:
        parts = []
        for i, ch in enumerate(chunks, 1):
            meta = ch.get("metadata") or {}
            src = meta.get("source_url") or meta.get("url") or ""
            title = meta.get("title") or ""
            page = meta.get("page_number")
            label = f"[{i}] {title}".strip()
            if page:
                label += f" (page {page})"
            if src:
                label += f" — {src}"
            parts.append(f"{label}\n{ch['text']}")
        return "\n\n---\n\n".join(parts)

    @staticmethod
    def _format_profile(profile: dict) -> str:
        if not profile:
            return ""
        lines = []
        if profile.get("user_name"):
            lines.append(f"- Name: {profile['user_name']}")
        if profile.get("interested_program"):
            lines.append(f"- Interested program: {profile['interested_program']}")
        if profile.get("education_level"):
            lines.append(f"- Education level: {profile['education_level']}")
        if profile.get("category"):
            lines.append(f"- Admission category: {profile['category']}")
        if not lines:
            return ""
        return (
            "USER PROFILE (use only to personalize the response, NOT as factual "
            "admission data — admission facts must come from the OFFICIAL NED "
            "CONTEXT below):\n" + "\n".join(lines)
        )

    @staticmethod
    def _dedupe_sources(chunks: list[dict]) -> list[dict]:
        seen: set[str] = set()
        sources = []
        for ch in chunks:
            meta = ch.get("metadata") or {}
            url = meta.get("source_url") or meta.get("url")
            if not url or url in seen:
                continue
            seen.add(url)
            sources.append({
                "url": url,
                "title": meta.get("title") or url,
                "page_number": meta.get("page_number"),
            })
        return sources

    # --------------- main entry ---------------
    def answer(self, question: str, session_id: str = "default") -> dict:
        if self.store.count() == 0:
            return {"answer": NO_CONTEXT_MESSAGE, "sources": [], "chunks": []}

        try:
            q_emb = self.embedder.embed(question)[0]
        except Exception as e:
            logger.exception("Embedding failed")
            return {"answer": f"Embedding failed: {e}", "sources": [], "chunks": []}

        chunks = self.store.query(q_emb, top_k=self.config.top_k)
        if not chunks:
            return {"answer": NO_CONTEXT_MESSAGE, "sources": [], "chunks": []}

        profile = self.memory.get_profile(session_id)
        history = self.memory.get_history(session_id, limit=self.config.history_limit)

        context_block = self._format_context(chunks)
        profile_block = self._format_profile(profile)

        user_prompt = (
            (profile_block + "\n\n" if profile_block else "")
            + "OFFICIAL NED CONTEXT:\n"
            + context_block
            + "\n\n"
            + f"QUESTION:\n{question}\n\n"
            + "INSTRUCTIONS:\n"
            "- Answer using ONLY the OFFICIAL NED CONTEXT above.\n"
            "- If the answer is not in the context, reply exactly: "
            "'I could not find this in the official NED data.'\n"
            "- After the answer, add a 'Sources:' section listing the URLs you used.\n"
            "- For deadlines, fees, and schedules, add a one-line reminder to "
            "confirm with the NED admission office."
        )

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(history)
        messages.append({"role": "user", "content": user_prompt})

        # Try the primary model first, then walk the fallback chain on
        # rate-limit / quota / availability errors. Each model is independent
        # — if Groq's daily cap is hit, we transparently try the next model
        # (possibly on a different provider like Gemini) so the user never
        # sees a dead app.
        models_to_try = [self.model] + list(self.fallbacks)
        last_error: Exception | None = None
        text: str | None = None
        for candidate in models_to_try:
            try:
                resp = completion(
                    model=candidate,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=self.config.max_tokens,
                )
                text = resp["choices"][0]["message"]["content"]
                if candidate != self.model:
                    logger.warning("Primary model failed; answered via fallback %s", candidate)
                break
            except Exception as e:
                last_error = e
                logger.warning("Model %s failed: %s", candidate, e)
                continue

        if text is None:
            logger.exception("All models failed", exc_info=last_error)
            return {
                "answer": (
                    "All language models are currently rate-limited or "
                    "unavailable. Please try again in a few minutes. "
                    f"(last error: {last_error})"
                ),
                "sources": [],
                "chunks": chunks,
            }

        sources = self._dedupe_sources(chunks)
        # Persist the bare question/answer pair so future turns have history.
        self.memory.add_message(session_id, "user", question)
        self.memory.add_message(session_id, "assistant", text)
        return {"answer": text, "sources": sources, "chunks": chunks}
