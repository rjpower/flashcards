import hashlib
import logging
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence

import genanki
import litellm

from srt.config import get_cache_path, settings
from srt.schema import FlashCard, VocabItem


def _id_from_name(name: str) -> int:
    # Use first 8 chars of md5 as hex, convert to int
    return int(hashlib.md5(name.encode()).hexdigest()[:8], 16)


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


@dataclass
class AudioModel:
    language_code: str
    model_name: str


AUDIO_MODELS = {
    "english": AudioModel("en-US", "en-US-Neural2-C"),
    "japanese": AudioModel("ja-JP", "ja-JP-Neural2-B"),
    "spanish": AudioModel("es-ES", "es-ES-Neural"),
    "chinese": AudioModel("cmn-CN", "cmn-CN-Standard-A"),
    "arabic": AudioModel("ar-XA", "ar-XA-Neural2-A"),
    "basque": AudioModel("eu-ES", "eu-ES-Standard-A"),
    "bengali": AudioModel("bn-IN", "bn-IN-Neural2-A"),
    "bulgarian": AudioModel("bg-BG", "bg-BG-Standard-A"),
    "catalan": AudioModel("ca-ES", "ca-ES-Standard-A"),
    "czech": AudioModel("cs-CZ", "cs-CZ-Wavenet-A"),
    "danish": AudioModel("da-DK", "da-DK-Neural2-D"),
    "dutch_be": AudioModel("nl-BE", "nl-BE-Standard-A"),
    "dutch_nl": AudioModel("nl-NL", "nl-NL-Neural2-A"),
    "filipino": AudioModel("fil-PH", "fil-PH-Neural2-A"),
    "finnish": AudioModel("fi-FI", "fi-FI-Wavenet-A"),
    "french_ca": AudioModel("fr-CA", "fr-CA-Neural2-A"),
    "french_fr": AudioModel("fr-FR", "fr-FR-Neural2-A"),
    "galician": AudioModel("gl-ES", "gl-ES-Standard-A"),
    "german": AudioModel("de-DE", "de-DE-Neural2-A"),
    "greek": AudioModel("el-GR", "el-GR-Neural2-A"),
    "gujarati": AudioModel("gu-IN", "gu-IN-Wavenet-A"),
    "hebrew": AudioModel("he-IL", "he-IL-Neural2-A"),
    "hindi": AudioModel("hi-IN", "hi-IN-Neural2-A"),
    "hungarian": AudioModel("hu-HU", "hu-HU-Wavenet-A"),
    "indonesian": AudioModel("id-ID", "id-ID-Wavenet-A"),
    "italian": AudioModel("it-IT", "it-IT-Neural2-A"),
    "kannada": AudioModel("kn-IN", "kn-IN-Wavenet-A"),
    "korean": AudioModel("ko-KR", "ko-KR-Neural2-A"),
    "latvian": AudioModel("lv-LV", "lv-LV-Standard-A"),
    "lithuanian": AudioModel("lt-LT", "lt-LT-Standard-A"),
    "malay": AudioModel("ms-MY", "ms-MY-Wavenet-A"),
    "malayalam": AudioModel("ml-IN", "ml-IN-Wavenet-A"),
    "mandarin_cn": AudioModel("cmn-CN", "cmn-CN-Neural2-A"),
    "mandarin_tw": AudioModel("cmn-TW", "cmn-TW-Wavenet-A"),
    "marathi": AudioModel("mr-IN", "mr-IN-Wavenet-A"),
    "norwegian": AudioModel("nb-NO", "nb-NO-Wavenet-A"),
    "polish": AudioModel("pl-PL", "pl-PL-Wavenet-A"),
    "portuguese_br": AudioModel("pt-BR", "pt-BR-Neural2-A"),
    "portuguese_pt": AudioModel("pt-PT", "pt-PT-Wavenet-A"),
    "punjabi": AudioModel("pa-IN", "pa-IN-Wavenet-A"),
    "romanian": AudioModel("ro-RO", "ro-RO-Wavenet-A"),
    "russian": AudioModel("ru-RU", "ru-RU-Neural2-A"),
    "serbian": AudioModel("sr-RS", "sr-RS-Standard-A"),
    "slovak": AudioModel("sk-SK", "sk-SK-Wavenet-A"),
    "spanish_es": AudioModel("es-ES", "es-ES-Neural2-A"),
    "spanish_us": AudioModel("es-US", "es-US-Neural2-A"),
    "swedish": AudioModel("sv-SE", "sv-SE-Wavenet-A"),
    "tamil": AudioModel("ta-IN", "ta-IN-Wavenet-A"),
    "telugu": AudioModel("te-IN", "te-IN-Standard-A"),
    "thai": AudioModel("th-TH", "th-TH-Neural2-C"),
    "turkish": AudioModel("tr-TR", "tr-TR-Neural2-A"),
    "ukrainian": AudioModel("uk-UA", "uk-UA-Wavenet-A"),
    "vietnamese": AudioModel("vi-VN", "vi-VN-Neural2-A"),
}


def generate_audio(term: str, language: str) -> Optional[bytes]:
    """Generate TTS audio for a term using Vertex AI TTS with SSML"""
    cache_path = get_cache_path(f"tts_{term}_{language}", "mp3")
    if cache_path.exists():
        return cache_path.read_bytes()

    response = litellm.speech(
        input=term,
        model="vertex_ai/cloud-tts",
        voice={
            "languageCode": AUDIO_MODELS[language].language_code,
            "name": AUDIO_MODELS[language].model_name,
        },
        audioConfig={
            "audioEncoding": "MP3",
            "speakingRate": 0.8,
            "pitch": 0.0,
        },
    )

    if response:
        audio_data = response.content

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
    language: str,
    logger: Callable[[str], None],
    max_workers: int = 16,
) -> dict[str, AudioData]:
    """Generate audio for cards using parallel processing"""
    audio_mapping = {}
    # Create a list of all terms we need to generate audio for
    items_to_process = []
    for item in items:
        if item.front:
            items_to_process.append(
                (language, item.front_sub if item.front_sub else item.front)
            )
        if item.back:
            items_to_process.append(("english", item.back))

    total = len(items_to_process)
    completed = 0

    def _generate_audio_task(lang: str, term: str) -> tuple[str, Optional[bytes]]:
        return term, generate_audio(term=term, language=lang)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_term = {
            executor.submit(_generate_audio_task, lang, term): term
            for (lang, term) in items_to_process
        }

        # Process completed tasks as they finish
        for future in as_completed(future_to_term):
            term = future_to_term[future]
            completed += 1

            logger(
                f"Generated audio {term} -- {completed}/{total} ({completed/total*100:.1f}%)"
            )

            try:
                term, audio_data = future.result()
                if audio_data:
                    audio_mapping[term] = AudioData(term, audio_data)
            except Exception as e:
                logger(f"Error generating audio for {term}: {str(e)}")

    logger(f"Completed audio generation for {completed} terms")
    return audio_mapping


def create_anki_package(
    output_path: Path,
    vocab_items: List[VocabItem],
    deck_name: str,
    audio_mapping: dict[str, AudioData],
    tgt_language: str,
    src_lang: str = "English",
) -> genanki.Package:
    """Create an Anki deck from vocabulary items"""
    model = genanki.Model(
        _id_from_name(deck_name),
        f"{tgt_language} Vocabulary",
        fields=[
            {"name": "Term"},
            {"name": "Reading"},
            {"name": "Meaning"},
            {"name": "Example"},
            {"name": "ExampleTranslation"},
            {"name": "TermAudio"},
            {"name": "MeaningAudio"},
        ],
        templates=[
            {
                "name": f"{tgt_language} to {src_lang}",
                "qfmt": """
                    <div class="term">{{Term}}</div>
                    {{TermAudio}}
                    <div class="example">{{Example}}</div>
                """,
                "afmt": """
                    {{FrontSide}}
                    <hr id="answer">
                    {{MeaningAudio}}
                    <div class="reading">{{Reading}}</div>
                    <div class="meaning">{{Meaning}}</div>
                    <div class="example-translation">{{ExampleTranslation}}</div>
                """,
            },
            {
                "name": f"{src_lang} to {tgt_language}",
                "qfmt": """
                    <div class="meaning">{{Meaning}}</div>
                    {{MeaningAudio}}
                    <div class="example-translation">{{ExampleTranslation}}</div>
                """,
                "afmt": """
                    {{FrontSide}}
                    <hr id="answer">
                    <div class="term">{{Term}}</div>
                    {{TermAudio}}
                    <div class="reading">{{Reading}}</div>
                    <div class="example">{{Example}}</div>
                """,
            },
        ],
        css=ANKI_CARD_CSS,
    )

    deck = genanki.Deck(_id_from_name(deck_name), deck_name)
    media_files = []

    # Create temporary directory for media files
    temp_dir = tempfile.TemporaryDirectory()

    for i, item in enumerate(vocab_items):
        fields = [
            item.front,
            item.front_sub,
            item.back,
            item.front_context or "",
            item.back_context or "",
            "",  # Term audio placeholder
            "",  # Meaning audio placeholder
        ]

        # Create a unique filename based on content hash
        def _add_audio(term: str) -> str:
            if term not in audio_mapping:
                return ""

            audio_filename = f"audio_{hashlib.md5(term.encode()).hexdigest()[:8]}.mp3"
            audio_path = Path(temp_dir.name) / audio_filename
            audio_path.write_bytes(audio_mapping[term].data)
            media_files.append(str(audio_path))
            return f"[sound:{audio_filename}]"

        fields[5] = _add_audio(item.front_sub if item.front_sub else item.front)
        fields[6] = _add_audio(item.back)

        note = genanki.Note(model=model, fields=fields)
        deck.add_note(note)

    package = genanki.Package(deck)
    package.media_files = media_files
    package.write_to_file(output_path)
    return package
