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

This project uses the normal `pyproject.toml` configuration.  You can use
(uv)[https://astral.sh/uv/] to automatically handle environments and
installation.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Usage

By default, the app uses Gemini Flash to handle translation and example sentence
context. You'll need a Gemini API Key to run it locally. The quickest way to get
a key is via AI Studio: https://aistudio.google.com/apikey. Set the key in your
environment before running the app:

```
export GEMINI_API_KEY=...
```


### Command Line Interface

You can generate flashcards in either Anki (apkg) or PDF format. When generating
Anki packages, use the `--audio` flag to additionally generate TTS audio. TTS
uses Google Cloud APIs, so you'll need to register for a Cloud Account setup the
CLI and use `gcloud auth application-default login` if you want to generate
audio.

The main entry points are `flashcards_from_csv` and `flashcards_from_srt`. When
generating from a CSV, you can have the LLM infer a field mapping, or specify
your own. By default the output file is written to `out/[input].{pdf|apkg}`.

```bash
uv run scripts/main.py flashcards_from_csv ./sample/n5.csv # inferred mapping
uv run scripts/main.py flashcards_from_csv ./sample/n5.csv --mapping=level=A,term=B
```

When generating an Anki package, you can specify the deck name and optionally
output TTS audio for the term and translation:

```bash
uv run scripts/main.py flashcards_from_csv ./sample/n5.csv [--audio] [--deck-name=JLPT::N5]
```

You can exclude terms based on the term or translation by specifying a `--filter` argument:

```bash
uv run scripts/main.py flashcards_from_csv ./sample/n3.csv --filter=known.txt --filter=./sample/n4.csv --filter=./sample/n5.csv --output=pdf
```

SRTs operate the same way:

```bash
uv run scripts/main.py flashcards_from_srt ./saikik-ep11.srt --output=pdf
```

### Web Interface

Start the web server:

```bash
uv run scripts/web.py
```

Navigate to http://localhost:8000 :
- Upload SRT files or CSV files
- Preview and customize field mappings
- Generate Anki decks or PDF flashcards
- Download processed files

## License

Apache 2.0
