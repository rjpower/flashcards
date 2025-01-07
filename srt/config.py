from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration settings for the SRT flashcard generator"""

    # Paths
    data_dir: Path = Field(
        default=Path("data"), description="Directory containing vocabulary data files"
    )
    cache_dir: Path = Field(
        default=Path("cache"), description="Directory for caching LLM responses"
    )
    output_dir: Path = Field(
        default=Path("out/anki"), description="Directory for generated Anki decks"
    )
    upload_dir: Path = Field(
        default=Path("uploads"), description="Directory for temporary uploaded files"
    )

    # LLM settings
    block_size: int = Field(
        default=100, description="Number of text blocks to process in each LLM batch"
    )
    llm_model: str = Field(
        default="gemini/gemini-2.0-flash-exp",
        description="LLM model to use for vocabulary analysis",
    )

    # Anki card settings
    anki_model_id: int = Field(
        default=1091735104, description="Unique identifier for the Anki note type"
    )
    anki_deck_id: int = Field(
        default=2059400110, description="Unique identifier for the Anki deck"
    )

    gcloud_project_id: str = Field(
        default="",
        description="Google Cloud project ID for accessing LLM API",
    )

    model_config = SettingsConfigDict(
        env_prefix="SRT_", env_file=".env", env_file_encoding="utf-8"
    )


# Create a global settings instance
settings = Settings()
