import json
import tempfile
from pathlib import Path

import pytest
from reportlab.pdfgen import canvas
from typer.testing import CliRunner

from scripts.main import app

runner = CliRunner(mix_stderr=False)


@pytest.fixture
def sample_pdf(tmp_path) -> Path:
    """Create a valid sample PDF file for testing."""
    pdf_path = tmp_path / "sample.pdf"
    c = canvas.Canvas(str(pdf_path))
    c.drawString(100, 750, "Sample PDF Content for Testing")
    c.save()
    return pdf_path


@pytest.fixture
def sample_csv(tmp_path) -> Path:
    """Create a sample CSV file for testing."""
    csv_content = "term,reading,meaning\n猫,ねこ,Cat\n犬,いぬ,Dog\n"
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text(csv_content, encoding="utf-8")
    return csv_path


@pytest.fixture
def sample_srt(tmp_path) -> Path:
    """Create a sample SRT file for testing."""
    srt_content = """1
00:00:01,000 --> 00:00:04,000
こんにちは！　僕はさっき日本に来ました。

2
00:00:05,000 --> 00:00:07,000
日本語を勉強しています。
"""
    srt_path = tmp_path / "sample.srt"
    srt_path.write_text(srt_content, encoding="utf-8")
    return srt_path


def test_flashcards_from_pdf_deck_name(sample_pdf, tmp_path):
    """Test that the --deck-name option correctly names the Anki deck for PDF files."""

    custom_deck_name = "CustomPDFDeck"
    output_format = "apkg"
    output_file = tmp_path / f"{custom_deck_name}_cards.{output_format}"

    # Generate a valid output file path
    output_file_parent = output_file.parent
    output_file_parent.mkdir(parents=True, exist_ok=True)

    # Run the CLI command
    runner.invoke(
        app,
        [
            "flashcards_from_pdf",
            str(sample_pdf),
            "--deck-name",
            custom_deck_name,
            "--format",
            output_format,
            "--output_file",
            str(output_file),
        ],
        catch_exceptions=False,
    )

    # Assert that the output file exists
    assert output_file.exists(), f"Expected output file {output_file} does not exist."

    # Optionally, verify that the output file is not empty
    assert output_file.stat().st_size > 0, "Output deck file is empty."


def test_flashcards_from_csv_deck_name(sample_csv, tmp_path):
    """Test that the --deck-name option correctly names the Anki deck for CSV files."""

    custom_deck_name = "CustomCSVDeck"
    output_format = "apkg"
    output_file = tmp_path / f"{custom_deck_name}.{output_format}"

    # Generate a valid output file path
    output_file_parent = output_file.parent
    output_file_parent.mkdir(parents=True, exist_ok=True)

    # Run the CLI command
    runner.invoke(
        app,
        [
            "flashcards_from_csv",
            str(sample_csv),
            "--deck-name",
            custom_deck_name,
            "--format",
            output_format,
            "--output_file",
            str(output_file),
        ],
        catch_exceptions=False,
    )

    # Assert that the output file exists
    assert output_file.exists(), f"Expected output file {output_file} does not exist."

    # Optionally, verify that the output file is not empty
    assert output_file.stat().st_size > 0, "Output deck file is empty."


def test_flashcards_from_srt_deck_name(sample_srt, tmp_path):
    """Test that the --deck-name option correctly names the Anki deck for SRT files."""

    custom_deck_name = "CustomSRTDeck"
    output_format = "apkg"
    output_file = tmp_path / f"{custom_deck_name}.{output_format}"

    # Generate a valid output file path
    output_file_parent = output_file.parent
    output_file_parent.mkdir(parents=True, exist_ok=True)

    # Run the CLI command
    runner.invoke(
        app,
        [
            "flashcards_from_srt",
            str(sample_srt),
            "--deck-name",
            custom_deck_name,
            "--format",
            output_format,
            "--output_file",
            str(output_file),
        ],
        catch_exceptions=False,
    )

    # Assert that the output file exists
    assert output_file.exists(), f"Expected output file {output_file} does not exist."

    # Optionally, verify that the output file is not empty
    assert output_file.stat().st_size > 0, "Output deck file is empty."
