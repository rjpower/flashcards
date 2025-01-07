from enum import Enum
from typing import Optional

from pydantic import BaseModel


class OutputFormat(str, Enum):
    ANKI_PKG = "apkg"
    PDF = "pdf"


class VocabItem(BaseModel):
    term: str
    reading: Optional[str] = None
    meaning: Optional[str] = None
    context_jp: Optional[str] = None
    context_en: Optional[str] = None
    level: Optional[str] = None
    source: Optional[str] = None


class SourceMapping(BaseModel):
    term: str
    reading: Optional[str] = None
    meaning: Optional[str] = None
    context_jp: Optional[str] = None
    context_en: Optional[str] = None
    level: Optional[str] = None
