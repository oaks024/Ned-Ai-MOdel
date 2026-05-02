"""Central configuration loaded from environment variables.

We deliberately keep all tunables here so beginners only have one place
to look when something needs tweaking.
"""
import os
from dotenv import load_dotenv

# Load variables from a local .env file if present.
load_dotenv()


class Config:
    def __init__(self):
        self.groq_api_key = os.getenv("GROQ_API_KEY", "").strip()
        self.litellm_model = os.getenv("LITELLM_MODEL", "groq/llama-3.3-70b-versatile")
        self.embedding_model = os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.chroma_db_path = os.getenv("CHROMA_DB_PATH", "./data/processed/chroma")
        self.memory_db_path = os.getenv("MEMORY_DB_PATH", "./data/processed/memory.db")
        self.ned_base_url = os.getenv("NED_BASE_URL", "https://www.neduet.edu.pk/")
        self.raw_data_path = os.getenv("RAW_DATA_PATH", "./data/raw")
        self.max_crawl_pages = int(os.getenv("MAX_CRAWL_PAGES", "80"))
        self.top_k = int(os.getenv("TOP_K", "5"))

        # LiteLLM reads GROQ_API_KEY from the environment. Make sure it is set
        # for this process even if the user only put it in .env.
        if self.groq_api_key:
            os.environ["GROQ_API_KEY"] = self.groq_api_key

    def validate(self):
        """Raises a clear error if anything required is missing."""
        if not self.groq_api_key:
            raise ValueError(
                "GROQ_API_KEY is missing. Copy .env.example to .env and add your key."
            )
