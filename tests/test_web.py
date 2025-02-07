import io
import json
from pathlib import Path

import pytest
from flask import url_for

from scripts.web import app
from srt.schema import ConversionProgress, ConversionStatus


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def get_completion_filename(response):
    """Helper to extract filename from completion event"""
    for line in response.get_data().decode().split("\n\n"):
        if line.startswith("data: "):
            print(line)
            event_data = json.loads(line[6:])
            progress = ConversionProgress.model_validate(event_data)
            if progress.status == ConversionStatus.DONE:
                return progress.payload
    return None


def csv_upload(client, tmp_path, filter_words=False):
    """Helper to upload CSV file with optional known words filter"""
    app.config["UPLOAD_FOLDER"] = tmp_path

    csv_content = "A,B,C\n1,漢字,meaning\n2,言葉,word\n3,本,book"
    csv_file = io.BytesIO(csv_content.encode("utf-8"))

    if filter_words:
        filter_content = "漢字\n本"  # Filter out 2 of 3 words
        filter_file = io.BytesIO(filter_content.encode("utf-8"))
    else:
        filter_file = io.BytesIO(b"")  # Empty filter

    data = {
        "file": (csv_file, "test.csv"),
        "filter_file": (filter_file, "known.txt"),
        "term_field": "B",
        "reading_field": "",
        "meaning_field": "C",
        "context_native_field": "",
        "context_en_field": "",
        "separator": ",",
        "format": "pdf",
    }

    response = client.post(
        "/upload/csv",
        data=data,
        content_type="multipart/form-data",
        follow_redirects=True,
    )

    assert response.status_code == 200
    filename = get_completion_filename(response)
    assert filename is not None, "Did not receive completion event"

    # Download the PDF
    response = client.get(f"/download/{filename}")
    assert response.status_code == 200
    pdf_content = response.data
    return len(pdf_content)  # Return size for comparison


def test_csv_upload_with_known_words(client, tmp_path):
    csv_upload(client, tmp_path, filter_words=True)


def test_csv_upload_without_known_words(client, tmp_path):
    """Test CSV upload without known words filter"""
    unfiltered_size = csv_upload(client, tmp_path, filter_words=False)
    filtered_size = csv_upload(client, tmp_path, filter_words=True)
    assert (
        filtered_size < unfiltered_size
    ), "Filtered PDF should be smaller than unfiltered PDF"


def srt_upload(client, tmp_path, include_audio=False, filter_words=None):
    """Helper to upload SRT file"""
    app.config["UPLOAD_FOLDER"] = tmp_path

    srt_content = "1\n00:00:01,000 --> 00:00:02,000\nHello\n\n2\n00:00:03,000 --> 00:00:04,000\nWorld"
    srt_file = io.BytesIO(srt_content.encode("utf-8"))
    filter_content = "Hello" if filter_words else ""
    filter_file = io.BytesIO(filter_content.encode("utf-8")) if filter_words else None

    data = {
        "file": (srt_file, "test.srt"),
        "format": "apkg",
    }
    if filter_file:
        data["filter_file"] = (filter_file, "known.txt")
    if include_audio:
        data["include_audio"] = "1"

    response = client.post(
        "/upload/srt",
        data=data,
        content_type="multipart/form-data",
        follow_redirects=True,
    )

    assert response.status_code == 200
    filename = get_completion_filename(response)
    assert filename is not None, "Did not receive completion event"

    # Download the generated file
    response = client.get(f"/download/{filename}")
    assert response.status_code == 200
    return response.data, filename


def test_srt_upload_without_filter(client, tmp_path):
    """Test SRT upload without known words filter"""
    srt_data, filename = srt_upload(client, tmp_path)
    assert filename.endswith(".apkg"), "Output should be Anki package"
    assert len(srt_data) > 0, "Anki package should not be empty"


def test_srt_upload_with_filter(client, tmp_path):
    """Test SRT upload with known words filter"""
    srt_data, filename = srt_upload(client, tmp_path, filter_words=True)
    assert filename.endswith(".apkg"), "Output should be Anki package"
    assert len(srt_data) > 0, "Anki package should not be empty"
    # Additional assertions can be added to verify filtered content


def test_srt_upload_with_audio(client, tmp_path):
    """Test SRT upload with audio generation"""
    srt_data, filename = srt_upload(client, tmp_path, include_audio=True)
    assert filename.endswith(".apkg"), "Output should be Anki package with audio"
    assert len(srt_data) > 0, "Anki package should not be empty"
    # Additional assertions can be added to verify audio inclusion


def test_csv_upload_with_audio(client, tmp_path):
    """Test CSV upload with audio generation"""
    app.config["UPLOAD_FOLDER"] = tmp_path

    # Read the sample CSV file
    csv_path = Path(__file__).parent.parent / "sample" / "tiny.csv"
    with open(csv_path, "rb") as f:
        csv_file = io.BytesIO(f.read())

    data = {
        "file": (csv_file, "tiny.csv"),
        "term_field": "term",  # These field names will need to be adjusted
        "reading_field": "reading",  # based on the actual CSV structure
        "meaning_field": "meaning",
        "context_native_field": "",
        "context_en_field": "",
        "separator": ",",
        "format": "apkg",
        "include_audio": "1",
    }

    response = client.post(
        "/upload/csv",
        data=data,
        content_type="multipart/form-data",
        follow_redirects=True,
    )

    assert response.status_code == 200
    filename = get_completion_filename(response)
    assert filename is not None, "Did not receive completion event"
    assert filename.endswith(".apkg"), "Output should be Anki package"

    # Download and verify the Anki package
    response = client.get(f"/download/{filename}")
    assert response.status_code == 200
    assert len(response.data) > 0, "Anki package should not be empty"
