import io
import json
import logging
import multiprocessing.dummy
import re
from pathlib import Path
from typing import List, Optional, Sequence

import pandas as pd
import pysrt
from pydantic import BaseModel

from srt.config import cached_completion, settings
from srt.generate_anki import create_anki_package, generate_audio_for_cards
from srt.generate_pdf_html import PDFGeneratorConfig, create_flashcard_pdf
from srt.schema import (
    OutputFormat,
    ProgressLogger,
    SourceMapping,
    VocabItem,
)


class SRTProcessConfig(BaseModel):
    srt_path: Path
    output_path: Path
    output_format: str
    include_audio: bool = False
    deck_name: Optional[str] = None
    ignore_words: set[str] = set()
    progress_logger: ProgressLogger


class CSVProcessConfig(BaseModel):

    class Config:
        arbitrary_types_allowed = True

    df: pd.DataFrame
    output_path: Path
    output_format: str
    include_audio: bool = False
    deck_name: Optional[str] = None
    field_mapping: SourceMapping
    ignore_words: set[str] = set()
    progress_logger: ProgressLogger


def load_csv_items(
    df: pd.DataFrame,
    mapping: SourceMapping,
) -> List[VocabItem]:
    """Load vocabulary items from a DataFrame using the specified field mapping

    Args:
        df: Pandas DataFrame containing vocabulary data
        mapping: Field mapping configuration
        chunk_size: Number of items to process in each LLM batch

    Returns:
        List of validated vocabulary items with inferred fields
    """
    logging.debug("Processing DataFrame with %d rows", len(df))
    rows = []
    for i, row in df.iterrows():
        item_data = {
            "term": row.get(mapping.term, "") if mapping.term else "",
            "reading": row.get(mapping.reading, "") if mapping.reading else "",
            "meaning": row.get(mapping.meaning, "") if mapping.meaning else "",
            "context_native": (
                row.get(mapping.context_native) if mapping.context_native else None
            ),
            "context_en": row.get(mapping.context_en) if mapping.context_en else None,
            "source": "csv_import",
        }

        # Only add items that have at least one non-empty main field
        if any([item_data["term"], item_data["reading"], item_data["meaning"]]):
            item = VocabItem.model_validate(item_data)
            rows.append(item)
    return rows


def _infer_fields(
    chunk: Sequence[VocabItem], progress_logger: ProgressLogger = logging.info
) -> List[VocabItem]:
    """Process a chunk of DataFrame rows into vocabulary items"""
    complete_records = [
        item for item in chunk if item.term and item.reading and item.meaning
    ]
    incomplete_records = [item for item in chunk if item not in complete_records]

    if incomplete_records:
        progress_logger(
            f"Inferring fields for {len(incomplete_records)} incomplete records"
        )
        return complete_records + infer_missing_fields(incomplete_records)
    return complete_records


def infer_missing_fields_parallel(
    rows: Sequence[VocabItem],
    progress_logger: ProgressLogger = logging.info,
    infer_chunk_size: int = 25,
):
    with multiprocessing.dummy.Pool(processes=4) as pool:
        chunk_results = pool.starmap(
            _infer_fields,
            [
                (rows[i : i + infer_chunk_size], progress_logger)
                for i in range(0, len(rows), infer_chunk_size)
            ],
            chunksize=1,
        )

        # Flatten results
        return [item for chunk in chunk_results for item in chunk]


def process_srt(config: SRTProcessConfig):
    """Process SRT file to generate flashcards"""
    config.progress_logger("Extracting text from SRT")
    text_blocks = extract_text_from_srt(config.srt_path)

    config.progress_logger("Analyzing vocabulary")
    vocab_items = analyze_vocabulary(text_blocks, config)

    # Filter known words and duplicates
    vocab_items = filter_known_vocabulary(vocab_items, config.ignore_words)
    vocab_items = remove_duplicate_terms(vocab_items)

    audio_mapping = {}
    if config.include_audio:
        config.progress_logger("Generating audio files")
        audio_mapping = generate_audio_for_cards(vocab_items, config.progress_logger)

    config.progress_logger(f"Exporting to {config.output_format}")

    if config.output_format == OutputFormat.ANKI_PKG:
        create_anki_package(
            config.output_path,
            vocab_items,
            config.deck_name or clean_filename(config.srt_path.name),
            audio_mapping=audio_mapping,
        )
    else:
        gen_config = PDFGeneratorConfig(
            cards=vocab_items,
            output_path=config.output_path,
        )
        create_flashcard_pdf(gen_config)


def process_csv(config: CSVProcessConfig):
    """Process CSV file to generate flashcards."""
    config.progress_logger("Loading CSV data")

    vocab_items = load_csv_items(config.df, config.field_mapping)
    vocab_items = filter_known_vocabulary(vocab_items, config.ignore_words)
    vocab_items = remove_duplicate_terms(vocab_items)
    config.progress_logger(
        f"{len(vocab_items)} vocabulary items after filtering and dedup."
    )
    vocab_items = infer_missing_fields_parallel(vocab_items, config.progress_logger)
    vocab_items = filter_known_vocabulary(vocab_items, config.ignore_words)
    config.progress_logger(
        f"{len(vocab_items)} vocabulary items after inference and filtering"
    )

    if config.include_audio:
        audio_mapping = generate_audio_for_cards(vocab_items, config.progress_logger)
    else:
        audio_mapping = {}

    # Export based on format
    config.progress_logger(f"Exporting to {config.output_format}")

    if config.output_format == OutputFormat.ANKI_PKG:
        create_anki_package(
            config.output_path,
            vocab_items,
            config.deck_name or "csv_import_deck",
            audio_mapping=audio_mapping,
        )
    else:
        gen_config = PDFGeneratorConfig(
            cards=vocab_items,
            output_path=config.output_path,
        )
        create_flashcard_pdf(gen_config)


def extract_text_from_srt(srt_path: Path) -> List[str]:
    """Extract all text from SRT file, combining consecutive subtitles"""
    subs = pysrt.open(srt_path)
    text_blocks = []

    for sub in subs:
        clean_text = re.sub(r"<[^>]+>", "", sub.text)
        text_blocks.append(clean_text)

    return text_blocks


def analyze_srt_section(text: str) -> List[VocabItem]:
    prompt = f"""Extract vocabulary items from the following text.
For each vocabulary item, find an actual example sentence from the provided text that uses it.
Return a JSON array of objects with these fields:

* term: The native language term
* reading: The reading of the term (e.g. Hiragana or Katakana for Japanese, Pinyin for Chinese) 
* meaning: The English meaning of the term
* context_native: The native example sentence with the term in context
* context_en: The English translation of the example sentence

[
{{
"term": "獣医",
"reading": "じゅうい",
"meaning": "Veterinarian",
"context_native": "<ruby>医者<rt>いしゃ</rt></ruby>より<ruby>獣医<rt>じゅうい</rt></ruby>になりたい",
"context_en": "I want to become a veterinarian rather than a doctor",
}},
{{
"term": "病院",
"reading": "びょういん",
"meaning": "Hospital",
"context_native": "<ruby>病院<rt>びょういん</rt></ruby>に行く",
"context_en": "Go to the hospital",
}}
]

Text to analyze:
{text}

Return only valid JSON, no other text."""

    chunk_results = json.loads(
        cached_completion(
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
    )
    return [VocabItem.model_validate(row) for row in chunk_results]


def analyze_vocabulary(text_blocks: List[str], config: SRTProcessConfig):
    """Submit text to LLM for vocabulary analysis with caching"""
    config.progress_logger("Starting vocabulary analysis")

    all_results = []
    num_chunks = len(text_blocks) // settings.block_size + 1

    for i in range(0, len(text_blocks), settings.block_size):
        text = "\n".join(text_blocks[i : i + settings.block_size])
        current_chunk = i // settings.block_size + 1
        progress = (current_chunk * 100) // num_chunks
        chunk_results = analyze_srt_section(text)

        config.progress_logger(f"Processing chunk {current_chunk}/{num_chunks}")
        all_results.extend(chunk_results)

    config.progress_logger("Vocabulary analysis complete")


def infer_missing_fields(vocab_items: List[VocabItem]) -> List[VocabItem]:
    """Infer missing fields for vocabulary items using LLM.

    Args:
        vocab_items: List of vocabulary items, each must have at least a term field

    Returns:
        List of vocabulary items with inferred fields filled in
    """
    # Convert items to dict for LLM processing
    items_data = [item.model_dump(exclude_unset=False) for item in vocab_items]

    prompt = f"""Given these vocabulary items, infer any missing fields.
Required fields: term, reading, meaning
Optional fields: context_native, context_en 

For each item:
- If reading is missing, provide the _reading_, e.g. Hiragana or Katakana for Japanese, Pinyin for Chinese
- If meaning is missing, provide the English meaning
- If context is missing, generate natural example sentences

Example of a complete item:
{{
    "term": "図書館",
    "reading": "としょかん",
    "meaning": "library",
    "context_native": "<ruby>図書館<rt>としょかん</rt></ruby>で<ruby>本<rt>ほん</rt></ruby>を<ruby>借<rt>か</rt></ruby>りました",
    "context_en": "I borrowed a book from the library",
}}

Input items:
{json.dumps(items_data, ensure_ascii=False, indent=2)}

Return only valid JSON array with complete items in same format."""

    completed_items = json.loads(
        cached_completion(
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
    )
    return [VocabItem.model_validate(item) for item in completed_items]


def filter_known_vocabulary(
    vocab_items: List[VocabItem], ignore_words: set = set()
) -> List[VocabItem]:
    """Remove vocabulary items that are already known or should be ignored"""
    kept = []
    for item in vocab_items:
        if item.term in ignore_words or item.meaning in ignore_words:
            continue
        kept.append(item)
    return kept


def remove_duplicate_terms(vocab_items: List[VocabItem]) -> List[VocabItem]:
    """Remove items with duplicate terms, keeping the first occurrence"""
    seen_terms = set()
    unique_items = []

    for item in vocab_items:
        if item.term not in seen_terms:
            seen_terms.add(item.term)
            unique_items.append(item)

    return unique_items


def clean_filename(filename: str) -> str:
    """Convert filename to clean format with dashes instead of spaces/special chars"""
    filename = Path(filename).stem
    cleaned = re.sub(r"[^.\w\s-]", "", filename)
    cleaned = re.sub(r"[-\s]+", "-", cleaned).strip()
    return cleaned.lower()


def read_csv(file_content: str) -> tuple[str, pd.DataFrame]:
    """Analyze CSV structure and return separator, column letters, and preview rows"""
    separators = [",", "\t", ";"]

    # Find best separator by trying each
    best_separator = ","
    max_columns = 0
    for sep in separators:
        try:
            df = pd.read_csv(io.StringIO(file_content), sep=sep, nrows=1, dtype=str)
            if len(df.columns) > max_columns:
                max_columns = len(df.columns)
                best_separator = sep
        except Exception:
            logging.debug("Failed to read CSV with separator: %s", sep)
            continue

    # Read preview with best separator
    df = pd.read_csv(io.StringIO(file_content), sep=best_separator, dtype=str)

    # Generate column letters (A, B, C, etc.)
    num_cols = len(df.columns)
    col_letters = [chr(65 + i) for i in range(num_cols)]  # A=65 in ASCII

    df.columns = col_letters

    return best_separator, df


def infer_field_mapping(df: pd.DataFrame) -> dict:
    """Get LLM suggestions for CSV field mapping using column letters"""
    logging.debug("Inferring field mapping for CSV data")
    preview_rows = df.head(25).fillna("").astype(str).values.tolist()
    sample_data = "\n".join(
        [",".join(df.columns), *[",".join(row) for row in preview_rows]]
    )

    prompt = f"""Analyze this CSV data and suggest mappings for a vocabulary flashcard system.
The system needs these fields: term, reading, meaning, and optionally context_native, context_en.
The columns are labeled with letters (A, B, C, etc.).
Look at the content in each column to suggest the best mapping.

CSV Data (first few rows):
{sample_data}

"term" is mandatory, and should be the native word or phrase.
"reading" is the pronunciation of the term, e.g. Hiragana or Katakana for Japanese, Pinyin for Chinese, etc.
"meaning" is the English translation of the term.
"context_native" is a native sentence using the term.
"context_en" is the English translation of the sentence.

Return only valid JSON in this format:
{{
    "suggested_mapping": {{
        "term": "A",
        "reading": "B",
        "meaning": "C",
        "context_native": "D" or null,
        "context_en": "E" or null,
    }},
    "confidence": "high|medium|low",
    "reasoning": "Brief explanation of why each column was mapped based on its content"
}}"""

    return json.loads(
        cached_completion(
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
    )
