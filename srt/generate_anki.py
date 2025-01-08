import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import genanki
import litellm

from srt.config import get_cache_path, settings
from srt.schema import FlashCard, VocabItem

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

    return None


# we might include audio for back & context later
@dataclass
class AudioData:
    term: str
    data: bytes


def generate_audio_for_cards(
    items: Sequence[FlashCard],
) -> dict[str, AudioData]:
    # Generate audio if requested
    audio_mapping = {}
    for i, item in enumerate(items):
        audio_data = generate_audio(item.front)
        if audio_data:
            audio_mapping[item.front] = AudioData(item.front, audio_data)
    return audio_mapping


def create_anki_package(
    output_path: Path,
    vocab_items: List[VocabItem],
    deck_name: str,
    audio_mapping: dict[str, AudioData],
) -> genanki.Package:
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

    for i, item in enumerate(vocab_items):
        fields = [
            item.term,
            item.reading,
            item.meaning,
            item.context_jp or "",
            item.context_en or "",
            "",  # Audio field placeholder
        ]

        # Create a unique filename based on content hash
        if item.term in audio_mapping:
            audio_filename = (
                f"audio_{hashlib.md5(item.term.encode()).hexdigest()[:8]}.mp3"
            )
            audio_path = temp_dir / audio_filename
            audio_path.write_bytes(audio_mapping[item.term].data)
            media_files.append(str(audio_path))
            fields[5] = f"[sound:{audio_filename}]"

        note = genanki.Note(model=model, fields=fields)
        deck.add_note(note)

    package = genanki.Package(deck)
    package.media_files = media_files
    package.write_to_file(output_path)
    return package
