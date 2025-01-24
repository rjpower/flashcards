import hashlib
import json
from pathlib import Path
from typing import List, Set

import litellm
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration settings for the SRT flashcard generator"""

    model_config = SettingsConfigDict(
        env_prefix="SRT_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Paths
    cache_dir: Path = Field(
        default=Path("cache"), description="Directory for caching LLM responses"
    )
    output_dir: Path = Field(
        default=Path("out"), description="Directory for generated Anki decks"
    )
    upload_dir: Path = Field(
        default=Path("uploads"), description="Directory for temporary uploaded files"
    )

    # LLM settings
    block_size: int = Field(
        default=100, description="Number of text blocks to process in each LLM batch"
    )
    llm_model: str = Field(
        default="gemini/gemini-1.5-flash",
        description="LLM model to use for vocabulary analysis",
    )


settings = Settings()


def get_cache_path(text: str, suffix: str = "json") -> Path:
    """Generate cache file path based on input text hash"""
    settings.cache_dir.mkdir(exist_ok=True)
    text_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
    return settings.cache_dir / f"{text_hash}.{suffix}"


def load_known_words() -> Set[str]:
    """Load already known words to filter them out"""
    known_path = Path("data/known.txt")
    if not known_path.exists():
        return set()

    with open(known_path, encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def cached_completion(messages: List[dict], **kw) -> str:
    """Execute LLM completion with caching

    Args:
        prompt: The prompt to send to the LLM
        response_format: Expected response format ("json_object" or "text")

    Returns:
        Parsed response from LLM
    """
    filtered_kw = {
        key: value
        for key, value in kw.items()
        if isinstance(value, (int, str, float, dict, list))
    }
    filtered_kw["model"] = settings.llm_model

    cache_key = json.dumps({"messages": messages, **filtered_kw}, sort_keys=True)
    cache_path = get_cache_path(cache_key)
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8")

    response = litellm.completion(model=settings.llm_model, messages=messages, **kw)

    result = response.choices[0].message.content
    cache_path.write_text(result)
    return result
