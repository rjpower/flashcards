# Langauge Learning Tools

A collection of tools for creating language learning materials from SRT
subtitles and CSV files. Generate Anki flashcards and PDF study materials with
automatic vocabulary analysis and optional text-to-speech audio.

## Features

- Extract vocabulary from SRT subtitle files using LLM analysis
- Import vocabulary from CSV files with flexible field mapping
- Generate Anki decks (.apkg) with:
  - Term, reading, meaning
  - Context sentences with translations
  - Optional text-to-speech audio
- Create PDF flashcards with ruby annotations
- Web interface for file uploads and processing
- Command-line interface for batch processing

## Installation

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

# or

wget -qO- https://astral.sh/uv/install.sh | sh
```

## Usage

### Web Interface

Start the web server:

```bash
uv run scripts/web.py
```

Then open http://localhost:8000 in your browser to:
- Upload SRT files or CSV files
- Preview and customize field mappings
- Generate Anki decks or PDF flashcards
- Download processed files

### Command Line Interface

Process SRT files to create Anki decks:

```bash
# Basic usage
uv run scripts/main.py path/to/subtitles.ja.srt

# Include TTS audio
uv run scripts/main.py path/to/subtitles.ja.srt --audio
```

Generate PDF flashcards from SRT or CSV:

```bash
# From SRT
uv run scripts/main.py export_pdf path/to/subtitles.ja.srt

# From CSV with automatic field mapping
uv run scripts/main.py export_pdf ./data/vocabulary.csv

# From CSV with manual field mapping
uv run scripts/main.py export_pdf ./data/vocabulary.csv -m "term=word,reading=kana"
```

## Development

Run tests:

```bash
uv run pytest tests/test_ruby_annotation.py
```

## License

Apache 2.0
