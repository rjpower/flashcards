import logging
import sys
from glob import glob
from pathlib import Path
from typing import List, Optional

import pandas as pd
import typer
from rich import print

from srt.analyze_pdf import AnalyzePdfConfig, CardType, process_pdf_images
from srt.config import settings
from srt.lib import (
    CSVProcessConfig,
    SRTProcessConfig,
    clean_filename,
    infer_field_mapping,
    process_csv,
    process_srt,
    read_csv,
)
from srt.schema import OutputFormat, SourceMapping, VocabItem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stderr),
    ],
)
logger = logging.getLogger(__name__)

app = typer.Typer()


def progress_logger(message: str):
    print(f"[blue]{message}[/blue]", file=sys.stderr)


@app.command(name="flashcards_from_pdf")
def flashcards_from_pdf(
    pdf_path: Path = typer.Argument(
        ...,
        help="Path to the PDF file to analyze",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
    output_format: str = typer.Option(
        "apkg",
        "--format",
        "-f",
        help="Output format (apkg or pdf)",
    ),
    card_type: CardType = typer.Option(
        CardType.VOCAB,
        "--card-type",
        "-c",
        help="Type of card to generate (vocab or sentences)",
    ),
    deck_name: Optional[str] = typer.Option(
        None,
        "--deck-name",
        "-d",
        help="Name of the Anki deck to create",
    ),
    output_file: Optional[Path] = typer.Option(
        None,
        "--output_file",
        "-o",
        help="Path to save the output file",
    ),
):
    """Extract learning content from PDF files and create flashcards."""
    clean_name = clean_filename(pdf_path.stem)
    deck_name = deck_name or clean_name

    if output_file:
        output_path = output_file
    else:
        output_dir = settings.output_dir
        output_dir.mkdir(exist_ok=True, parents=True)
        output_path = output_dir / f"{deck_name}_cards.{output_format}"

    print(f"[blue]Analyzing PDF: {pdf_path}[/blue]")

    config = AnalyzePdfConfig(
        pdf_path=pdf_path,
        output_path=output_path,
        output_format=OutputFormat(output_format),
        card_type=card_type,
        progress_logger=progress_logger,
    )

    process_pdf_images(config)

    print(f"[green]Output written to {output_path}[/green]")


@app.command(name="flashcards_from_csv")
def flashcards_from_csv(
    input_path: Path = typer.Argument(
        ...,
        help="Path to the SRT or CSV file to process",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
    output_format: str = typer.Option(
        "apkg",
        "--format",
        "-f",
        help="Output format (apkg or pdf)",
    ),
    mapping: Optional[str] = typer.Option(
        None,
        "--mapping",
        "-m",
        help="Field mapping for CSV files (e.g. 'term=word,reading=kana')",
    ),
    filter_path: List[Path] = typer.Option(
        [],
        "--filter",
        "-i",
        help="Glob pattern or path to CSV files containing words to ignore",
    ),
    include_audio: bool = typer.Option(
        False,
        "--audio",
        "-a",
        help="Include TTS audio in the output",
    ),
    deck_name: Optional[str] = typer.Option(
        None,
        "--deck-name",
        "-d",
        help="Name of the Anki deck to create",
    ),
    output_file: Optional[Path] = typer.Option(
        None,
        "--output_file",
        "-o",
        help="Path to save the output file",
    ),
):
    """Export vocabulary from CSV files directly to PDF flashcards."""
    clean_name = input_path.stem.lower()
    deck_name = deck_name or clean_name

    if output_file:
        output_path = output_file
    else:
        output_dir = settings.output_dir
        output_dir.mkdir(exist_ok=True, parents=True)
        output_path = output_dir / f"{deck_name}.{output_format}"

    # Parse mapping string or infer mapping
    field_mapping = None
    with open(input_path, "r", encoding="utf-8") as f:
        content = f.read()
    separator, df = read_csv(content)

    if mapping:
        pairs = dict(pair.split("=") for pair in mapping.split(","))
        field_mapping = SourceMapping(
            term=pairs.get("term"),
            reading=pairs.get("reading"),
            meaning=pairs.get("meaning"),
            context_native=pairs.get("context_native"),
            context_en=pairs.get("context_en"),
        )
    else:
        # Infer mapping from CSV content
        result = infer_field_mapping(df)
        field_mapping = SourceMapping.model_validate(result["suggested_mapping"])
        print(f"[yellow]Inferred mapping ({result['confidence']} confidence):[/yellow]")
        print(f"[yellow]{result['reasoning']}[/yellow]")

    # load first few items using field mapping
    for i, row in df.head(5).iterrows():
        vocab_item = VocabItem(
            term=row[field_mapping.term],
            reading=row.get(field_mapping.reading, ""),
            meaning=row.get(field_mapping.meaning, ""),
            context_native=row.get(field_mapping.context_native, ""),
            context_en=row.get(field_mapping.context_en, ""),
        )
        print(f"[yellow]{vocab_item}[/yellow]")

    # Load ignore words if filter path provided
    ignore_words = set()
    filter_paths_expanded = []
    for path in filter_path:
        if "*" in str(path):
            filter_paths_expanded.extend(Path(p) for p in glob(str(path)))
        else:
            filter_paths_expanded.append(path)
    filter_path = filter_paths_expanded

    for f in filter_path:
        if f.suffix.lower() == ".csv":
            filter_df = pd.read_csv(f, dtype=str)
            # Add all non-empty values from all columns
            for col in filter_df.columns:
                ignore_words.update(
                    x.strip() for x in filter_df[col].dropna().astype(str)
                )
        else:
            filter_list = set([w.strip() for w in f.read_text().splitlines()])
            ignore_words = ignore_words.union(filter_list)

        print(f"[yellow]Loaded {len(ignore_words)} words to ignore[/yellow]")

    config = CSVProcessConfig(
        df=df,
        output_path=output_path,
        output_format=output_format,
        include_audio=include_audio,
        deck_name=deck_name,
        field_mapping=field_mapping,
        ignore_words=ignore_words,
        progress_logger=progress_logger,
    )
    process_csv(config)

    print(f"[green]{output_format} written to {output_path}[/green]", file=sys.stderr)


@app.command(name="flashcards_from_srt")
def flashcard_from_srt(
    srt_path: Path = typer.Argument(
        ...,
        help="Path to the SRT file to analyze",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
    output_format: str = typer.Option(
        "apkg",
        "--format",
        "-f",
        help="Output format (apkg or pdf)",
    ),
    include_audio: bool = typer.Option(
        False,
        "--audio",
        "-a",
        help="Include TTS audio in the output",
    ),
    deck_name: Optional[str] = typer.Option(
        None,
        "--deck-name",
        "-d",
        help="Name of the Anki deck to create",
    ),
    output_file: Optional[Path] = typer.Option(
        None,
        "--output_file",
        "-o",
        help="Path to save the output file",
    ),
):
    """Extract vocabulary from SRT subtitle files using LLM analysis."""
    clean_name = srt_path.stem.lower()
    deck_name = deck_name or clean_name

    if output_file:
        output_path = output_file
    else:
        output_dir = settings.output_dir
        output_dir.mkdir(exist_ok=True, parents=True)
        output_path = output_dir / f"{deck_name}.{output_format}"

    config = SRTProcessConfig(
        srt_path=srt_path,
        output_path=output_path,
        output_format=OutputFormat(output_format),
        include_audio=include_audio,
        deck_name=deck_name,
        progress_logger=progress_logger,
    )

    config.progress_logger = progress_logger
    process_srt(config)

    print(f"[green]Output written to {output_path}[/green]", file=sys.stderr)


if __name__ == "__main__":
    app()
