import hashlib
import io
import json
import logging
import multiprocessing
import re
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import List, Optional, Set

import genanki
import litellm
import pandas as pd
import pysrt
from pydantic import BaseModel

from srt.schema import OutputFormat, SourceMapping, VocabItem

from .config import settings
from .flashcard_pdf import create_flashcard_pdf

ANKI_CARD_CSS = """
.card {
    font-family: "Hiragino Sans", "Hiragino Kaku Gothic Pro", "Yu Gothic", Meiryo, sans-serif;
    font-size: 24px;
    text-align: center;
    color: #2c3e50;
    background-color: #f8f9fa;
    max-width: 800px;
    margin: 20px auto;
    padding: 20px;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    border-radius: 8px;
}
.term {
    font-size: 32px;
    color: #2c3e50;
    margin-bottom: 15px;
    font-weight: bold;
}
.reading {
    font-size: 20px;
    color: #666;
    margin: 15px 0;
    font-family: "Hiragino Sans", sans-serif;
}
.meaning {
    font-size: 22px;
    color: #34495e;
    margin: 15px 0;
    padding: 10px;
    background-color: #e9ecef;
    border-radius: 5px;
}
ruby {
    font-size: 20px;
}
rt {
    font-size: 12px;
    color: #666;
}
hr#answer {
    border: none;
    border-top: 2px solid #dee2e6;
    margin: 20px 0;
}
.example {
    font-size: 18px;
    color: #495057;
    margin: 15px 0;
    line-height: 1.6;
    padding: 15px;
    background-color: #fff;
    border-left: 4px solid #4CAF50;
    border-radius: 4px;
}
.example-translation {
    font-size: 16px;
    color: #666;
    font-style: italic;
    margin: 10px 0;
    padding: 10px;
    background-color: #f8f9fa;
    border-radius: 4px;
}
"""


class SRTConversionStage(str, Enum):
    STARTING = "starting"
    PROCESSING = "processing"
    ANALYZING = "analyzing"
    GENERATING_AUDIO = "generating_audio"
    COMPLETE = "complete"
    ERROR = "error"


class SRTConversionProgress(BaseModel):
    stage: SRTConversionStage
    message: str
    progress: int = 0
    result: Optional[List[VocabItem]] = None
    error: Optional[str] = None
    filename: Optional[str] = None


class SRTProcessConfig(BaseModel):
    srt_path: Path
    output_path: Path
    output_format: str
    include_audio: bool = False
    deck_name: Optional[str] = None


class CSVProcessConfig(BaseModel):

    class Config:
        arbitrary_types_allowed = True

    df: pd.DataFrame
    output_path: Path
    output_format: str
    include_audio: bool = False
    deck_name: Optional[str] = None
    field_mapping: SourceMapping


def _process_chunk(chunk_df: pd.DataFrame, mapping: SourceMapping) -> List[VocabItem]:
    """Process a chunk of DataFrame rows into vocabulary items"""
    chunk_items = []

    for _, row in chunk_df.iterrows():
        item_data = {
            "term": row.get(mapping.term, "") if mapping.term else "",
            "reading": row.get(mapping.reading, "") if mapping.reading else "",
            "meaning": row.get(mapping.meaning, "") if mapping.meaning else "",
            "context_jp": row.get(mapping.context_jp) if mapping.context_jp else None,
            "context_en": row.get(mapping.context_en) if mapping.context_en else None,
            "level": row.get(mapping.level) if mapping.level else None,
            "source": "csv_import",
        }

        # Only add items that have at least one non-empty main field
        if any([item_data["term"], item_data["reading"], item_data["meaning"]]):
            item = VocabItem.model_validate(item_data)
            chunk_items.append(item)

    complete_records = [
        item for item in chunk_items if item.term and item.reading and item.meaning
    ]
    incomplete_records = [item for item in chunk_items if item not in complete_records]

    # Only process incomplete chunks
    if incomplete_records:
        logging.info("Processing %d incomplete records", len(incomplete_records))
        return complete_records + infer_missing_fields(incomplete_records)
    return complete_records


def load_csv_items(
    df: pd.DataFrame, mapping: SourceMapping, infer_chunk_size: int = 25
) -> List[VocabItem]:
    """Load vocabulary items from a DataFrame using the specified field mapping

    Args:
        df: Pandas DataFrame containing vocabulary data
        mapping: Field mapping configuration
        chunk_size: Number of items to process in each LLM batch

    Returns:
        List of validated vocabulary items with inferred fields
    """
    logging.info("Processing DataFrame with %d rows", len(df))

    # Split DataFrame into chunks
    chunks = [
        df.iloc[i : i + infer_chunk_size] for i in range(0, len(df), infer_chunk_size)
    ]

    with multiprocessing.Pool(processes=4) as pool:
        try:
            chunk_results = pool.starmap(
                _process_chunk, [(chunk, mapping) for chunk in chunks]
            )
        except Exception as e:
            logging.error("Error processing chunks: %s", str(e))
            raise

        # Flatten results
        return [item for chunk in chunk_results for item in chunk]


def process_srt(config: SRTProcessConfig):
    """Process SRT file to generate flashcards, yielding progress updates"""
    yield SRTConversionProgress(
        stage=SRTConversionStage.STARTING, message="Starting SRT processing", progress=0
    )

    # Extract text
    yield SRTConversionProgress(
        stage=SRTConversionStage.PROCESSING,
        message="Extracting text from SRT",
        progress=10,
    )
    text_blocks = extract_text_from_srt(config.srt_path)

    # Analyze vocabulary
    vocab_items = []
    for progress in analyze_vocabulary(text_blocks):
        if progress.stage == SRTConversionStage.COMPLETE:
            vocab_items = progress.result
        yield progress

    # Filter known words and duplicates
    known_words = load_known_words()
    vocab_items = filter_known_vocabulary(vocab_items, known_words)
    vocab_items = remove_duplicate_terms(vocab_items)

    # Generate audio if requested
    if config.include_audio:
        yield SRTConversionProgress(
            stage=SRTConversionStage.GENERATING_AUDIO,
            message="Generating audio files",
            progress=70,
        )

        media_files = []
        for i, item in enumerate(vocab_items):
            audio_data = generate_audio(item.term)
            if audio_data:
                audio_filename = (
                    f"audio_{i}_{hashlib.md5(item.term.encode()).hexdigest()[:8]}.mp3"
                )
                media_files.append((audio_filename, audio_data))
            yield SRTConversionProgress(
                stage=SRTConversionStage.GENERATING_AUDIO,
                message=f"Generated audio for {item.term}",
                progress=(i + 1) * 100 // len(vocab_items),
            )

    # Export based on format
    yield SRTConversionProgress(
        stage=SRTConversionStage.PROCESSING,
        message=f"Exporting to {config.output_format}",
        progress=90,
    )

    if config.output_format == OutputFormat.ANKI_PKG:
        deck, media_files = create_anki_deck(
            vocab_items,
            config.deck_name or clean_filename(config.srt_path.name),
            config.include_audio,
        )

        # Create package with media files
        package = genanki.Package(deck)
        package.media_files = media_files
        package.write_to_file(str(config.output_path))

        # Clean up temporary media files
        for file in media_files:
            Path(file).unlink(missing_ok=True)
        temp_dir = settings.cache_dir / "temp_media"
        temp_dir.rmdir()
    else:
        filename = export_pdf(vocab_items, config.output_path)

    yield SRTConversionProgress(
        stage=SRTConversionStage.COMPLETE,
        message="Processing complete",
        progress=100,
        filename=config.output_path.name,
    )


def process_csv(config: CSVProcessConfig):
    """Process CSV file to generate flashcards, yielding progress updates"""
    yield SRTConversionProgress(
        stage=SRTConversionStage.STARTING, message="Starting CSV processing", progress=0
    )

    # Load and filter CSV
    yield SRTConversionProgress(
        stage=SRTConversionStage.PROCESSING, message="Loading CSV data", progress=30
    )

    vocab_items = load_csv_items(config.df, config.field_mapping)
    known_words = load_known_words()
    vocab_items = filter_known_vocabulary(vocab_items, known_words)
    vocab_items = remove_duplicate_terms(vocab_items)

    # Generate audio if requested
    if config.include_audio:
        yield SRTConversionProgress(
            stage=SRTConversionStage.GENERATING_AUDIO,
            message="Generating audio files",
            progress=70,
        )

        media_files = []
        for i, item in enumerate(vocab_items):
            audio_data = generate_audio(item.term)
            if audio_data:
                audio_filename = (
                    f"audio_{i}_{hashlib.md5(item.term.encode()).hexdigest()[:8]}.mp3"
                )
                media_files.append((audio_filename, audio_data))

    # Export based on format
    yield SRTConversionProgress(
        stage=SRTConversionStage.PROCESSING,
        message=f"Exporting to {config.output_format}",
        progress=90,
    )

    if config.output_format == OutputFormat.ANKI_PKG:
        deck, _ = create_anki_deck(
            vocab_items,
            config.deck_name or clean_filename(config.csv_path.name),
            config.include_audio,
        )
        genanki.Package(deck).write_to_file(str(config.output_path))
    else:
        filename = export_pdf(vocab_items, config.output_path)

    yield SRTConversionProgress(
        stage=SRTConversionStage.COMPLETE,
        message="Processing complete",
        progress=100,
        filename=config.output_path.name,
    )


def export_pdf(vocab_items: List[VocabItem], output_path: Path) -> str:
    """Export vocabulary items to PDF flashcards

    Returns:
        str: The filename of the generated PDF
    """
    create_flashcard_pdf(vocab_items, output_path)
    return output_path.name


def load_known_words() -> Set[str]:
    """Load already known words to filter them out"""
    known_path = Path("data/known.txt")
    if not known_path.exists():
        return set()

    with open(known_path, encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def extract_text_from_srt(srt_path: Path) -> List[str]:
    """Extract all text from SRT file, combining consecutive subtitles"""
    subs = pysrt.open(srt_path)
    text_blocks = []

    for sub in subs:
        clean_text = re.sub(r"<[^>]+>", "", sub.text)
        text_blocks.append(clean_text)

    return text_blocks


def cached_completion(prompt: str, response_format: str = "json_object") -> dict:
    """Execute LLM completion with caching

    Args:
        prompt: The prompt to send to the LLM
        response_format: Expected response format ("json_object" or "text")

    Returns:
        Parsed response from LLM
    """
    cache_path = get_cache_path(prompt)
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    response = litellm.completion(
        model=settings.llm_model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": response_format},
    )

    result = json.loads(response.choices[0].message.content)
    cache_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def get_cache_path(text: str, suffix: str = "json") -> Path:
    """Generate cache file path based on input text hash"""
    settings.cache_dir.mkdir(exist_ok=True)
    text_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
    return settings.cache_dir / f"{text_hash}.{suffix}"


def generate_audio(term: str) -> Optional[bytes]:
    """Generate TTS audio for a term using Vertex AI TTS with SSML"""
    cache_path = get_cache_path(f"tts_{term}", "mp3")
    if cache_path.exists():
        return cache_path.read_bytes()

    ssml = f"""
    <speak>
        <prosody rate="slow">
            <p>{term}</p>
        </prosody>
    </speak>
    """

    response = litellm.speech(
        input=ssml,
        model="vertex_ai/cloud-tts",
        project_id=settings.gcloud_project_id,
        voice={
            "languageCode": "ja-JP",
            "name": "ja-JP-Neural2-B",
        },
        audioConfig={
            "audioEncoding": "MP3",
            "speakingRate": 0.8,
            "pitch": 0.0,
        },
    )

    if response:
        temp_path = get_cache_path(f"tts_temp_{term}", "mp3")
        response.write_to_file(temp_path)
        audio_data = temp_path.read_bytes()
        temp_path.unlink()

        if audio_data:
            cache_path.write_bytes(audio_data)
            return audio_data


def analyze_chunk(text: str) -> List[VocabItem]:
    prompt = f"""Extract Japanese vocabulary items from the following text.
For each vocabulary item, find an actual example sentence from the provided text that uses it.
Return a JSON array of objects with these fields:

* term: The Japanese term
* reading: The reading of the term using Hiragana or Katakana
* meaning: The English meaning of the term
* context_jp: The Japanese example sentence with the term in context
* context_en: The English translation of the example sentence

[
{{
"term": "獣医",
"reading": "じゅうい",
"meaning": "Veterinarian",
"context_jp": "<ruby>医者<rt>いしゃ</rt></ruby>より<ruby>獣医<rt>じゅうい</rt></ruby>になりたい",
"context_en": "I want to become a veterinarian rather than a doctor",
}},
{{
"term": "病院",
"reading": "びょういん",
"meaning": "Hospital",
"context_jp": "<ruby>病院<rt>びょういん</rt></ruby>に行く",
"context_en": "Go to the hospital",
}}
]

Text to analyze:
{text}

Return only valid JSON, no other text."""

    chunk_results = cached_completion(prompt, "json_object")
    return [VocabItem.model_validate(row) for row in chunk_results]


def analyze_vocabulary(text_blocks: List[str]):
    """Submit text to LLM for vocabulary analysis with caching"""
    yield SRTConversionProgress(
        stage=SRTConversionStage.STARTING, message="Starting vocabulary analysis"
    )

    all_results = []
    num_chunks = len(text_blocks) // settings.block_size + 1

    for i in range(0, len(text_blocks), settings.block_size):
        text = "\n".join(text_blocks[i : i + settings.block_size])
        current_chunk = i // settings.block_size + 1
        progress = (current_chunk * 100) // num_chunks
        chunk_results = analyze_chunk(text)

        yield SRTConversionProgress(
            stage=SRTConversionStage.PROCESSING,
            message=f"Processing chunk {current_chunk}/{num_chunks}",
            progress=progress,
        )
        all_results.extend(chunk_results)

    yield SRTConversionProgress(
        stage=SRTConversionStage.COMPLETE,
        message="Vocabulary analysis complete",
        result=all_results,
    )


def infer_missing_fields(vocab_items: List[VocabItem]) -> List[VocabItem]:
    """Infer missing fields for vocabulary items using LLM.

    Args:
        vocab_items: List of vocabulary items, each must have at least a term field

    Returns:
        List of vocabulary items with inferred fields filled in
    """
    # Convert items to dict for LLM processing
    items_data = [item.model_dump(exclude_unset=False) for item in vocab_items]

    prompt = f"""Given these Japanese vocabulary items, infer any missing fields.
Required fields: term, reading, meaning
Optional fields: context_jp, context_en, level

For each item:
- If reading is missing, provide the hiragana/katakana reading
- If meaning is missing, provide the English meaning
- If context is missing, generate natural example sentences
- If level is missing, estimate JLPT level (N5-N1)

Example of a complete item:
{{
    "term": "図書館",
    "reading": "としょかん",
    "meaning": "library",
    "context_jp": "<ruby>図書館<rt>としょかん</rt></ruby>で<ruby>本<rt>ほん</rt></ruby>を<ruby>借<rt>か</rt></ruby>りました",
    "context_en": "I borrowed a book from the library",
    "level": "N5"
}}

Input items:
{json.dumps(items_data, ensure_ascii=False, indent=2)}

Return only valid JSON array with complete items in same format."""

    completed_items = cached_completion(prompt, "json_object")
    return [VocabItem.model_validate(item) for item in completed_items]


def filter_known_vocabulary(
    vocab_items: List[VocabItem], known_words: set
) -> List[VocabItem]:
    """Remove vocabulary items that are already known"""
    matched = [item for item in vocab_items if item.term not in known_words]
    matched = [item for item in matched if item.meaning not in known_words]
    return matched


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


def create_anki_deck(
    vocab_items: List[VocabItem], deck_name: str, include_audio: bool = False
) -> tuple[genanki.Deck, list]:
    """Create an Anki deck from vocabulary items"""
    model = genanki.Model(
        settings.anki_model_id,
        "Japanese Vocabulary",
        fields=[
            {"name": "Term"},
            {"name": "Reading"},
            {"name": "Meaning"},
            {"name": "Example"},
            {"name": "ExampleTranslation"},
            {"name": "Audio"},
        ],
        templates=[
            {
                "name": "Japanese to English",
                "qfmt": """
                    <div class="term">{{Term}}</div>
                    {{#Audio}}{{Audio}}{{/Audio}}
                    <div class="example">{{Example}}</div>
                """,
                "afmt": """
                    {{FrontSide}}
                    <hr id="answer">
                    <div class="reading">{{Reading}}</div>
                    <div class="meaning">{{Meaning}}</div>
                    <div class="example-translation">{{ExampleTranslation}}</div>
                """,
            },
            {
                "name": "English to Japanese",
                "qfmt": """
                    <div class="meaning">{{Meaning}}</div>
                    <div class="example-translation">{{ExampleTranslation}}</div>
                """,
                "afmt": """
                    {{FrontSide}}
                    <hr id="answer">
                    <div class="term">{{Term}}</div>
                    {{#Audio}}{{Audio}}{{/Audio}}
                    <div class="reading">{{Reading}}</div>
                    <div class="example">{{Example}}</div>
                """,
            },
        ],
        css=ANKI_CARD_CSS,
    )

    deck = genanki.Deck(settings.anki_deck_id, deck_name)
    media_files = []

    # Create temporary directory for media files
    temp_dir = settings.cache_dir / "temp_media"
    temp_dir.mkdir(exist_ok=True, parents=True)

    try:
        for i, item in enumerate(vocab_items):
            fields = [
                item.term,
                item.reading,
                item.meaning,
                item.context_jp or "",
                item.context_en or "",
                "",  # Audio field placeholder
            ]

            if include_audio:
                audio_data = generate_audio(item.term)
                if audio_data:
                    # Create a unique filename based on content hash
                    audio_filename = (
                        f"audio_{hashlib.md5(item.term.encode()).hexdigest()[:8]}.mp3"
                    )
                    audio_path = temp_dir / audio_filename

                    # Write audio file to temporary directory
                    audio_path.write_bytes(audio_data)

                    # Add to media files list (path will be used by Package)
                    media_files.append(str(audio_path))

                    # Reference the audio file in the note
                    fields[5] = f"[sound:{audio_filename}]"

            note = genanki.Note(model=model, fields=fields)
            deck.add_note(note)

        return deck, media_files
    except Exception as e:
        # Clean up temp files in case of error
        for file in media_files:
            Path(file).unlink(missing_ok=True)
        temp_dir.rmdir()
        raise e


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
            logging.info("Failed to read CSV with separator: %s", sep)
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
    logging.info("Inferring field mapping for CSV data")
    preview_rows = df.head(5).fillna("").astype(str).values.tolist()
    sample_data = "\n".join(
        [",".join(df.columns), *[",".join(row) for row in preview_rows]]
    )

    prompt = f"""Analyze this CSV data and suggest mappings for a Japanese vocabulary flashcard system.
The system needs these fields: term, reading, meaning, and optionally context_jp, context_en, and level.
The columns are labeled with letters (A, B, C, etc.).
Look at the content in each column to suggest the best mapping.

CSV Data (first few rows):
{sample_data}

Return only valid JSON in this format:
{{
    "suggested_mapping": {{
        "term": "A",
        "reading": "B",
        "meaning": "C",
        "context_jp": "D" or null,
        "context_en": "E" or null,
        "level": "F" or null,
    }},
    "confidence": "high|medium|low",
    "reasoning": "Brief explanation of why each column was mapped based on its content"
}}"""

    return cached_completion(prompt, "json_object")
