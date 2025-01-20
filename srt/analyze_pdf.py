import base64
import enum
import json
import logging
import multiprocessing.dummy
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, List, Sequence

import litellm
from openai import audio

from srt.config import cached_completion
from srt.generate_pdf_html import PDFGeneratorConfig, create_flashcard_pdf
from srt.images_from_pdf import ImageData, PdfOptions, extract_images_from_pdf
from srt.lib import (
    create_anki_package,
)
from srt.schema import OutputFormat, ProgressLogger, RawFlashCard

logger = logging.getLogger(__name__)

VOCAB_PROMPT = """
Analyze these pages from a book and create vocabulary flashcards for each word.
You may also create cards to explain novel grammar items.

For each item you find, create a flashcard with the following fields:

    front: str - the vocabulary word or grammar term
    front_sub: str - the reading or subtext for the front of the card
    front_context: str - a sentence demonstrating the word or grammar point
    back: str - the English translation or explanation
    back_context: str - a translation of the context sentence

The front and back context should be minimal sentences which demonstrate the
word or grammar point. Use sentences from the text, but only if they highlight the
word we are interested in. Otherwise create your own.

e.g. for a vocabulary word like "猫" (cat), you might create a card like:

{
    "front": "猫",
    "front_sub": "ねこ",
    "front_context": "猫は可愛いです。",
    "back": "cat",
    "back_context": "Cats are cute."
}

for a grammar term like "〜たり〜たりする" (doing things like...):

{
    "front": "〜たり〜たりする",
    "front_context": "映画を見たり、本を読んだりします。",
    "back": "doing things like...",
    "back_context": "I do things like watch movies and read books."
}

Return only valid JSON array of flashcards."""

SENTENCE_PROMPT = """
Analyze these pages from a book and create sentence flashcards for each sentence you find.

For each item you find, create a flashcard with the following fields:

    front: str - the sentence in the original language
    front_sub: str - the reading or subtext for the front of the card if needed
    back: str - English translation of the sentence

e.g. for a sentence like "案内してあげるよ？" you might create a card like:

{
  "front": "案内してあげるよ？",
  "front_sub": "あんないしてあげるよ？",
  "back": "Should I show you the way?"
}

Assume the reader has a rough understanding of the language, don't include basic sentences.

Return only valid JSON array of flashcards.
"""


class CardType(enum.StrEnum):
    VOCAB = "vocab"
    SENTENCE = "sentence"


def _analyze_image_batch(
    image_batch: Sequence[ImageData], card_type: CardType
) -> List[RawFlashCard]:
    """Process a batch of images into flashcards"""
    # Convert images to base64
    image_contents = []
    for image in image_batch:
        image_b64 = base64.b64encode(image.content).decode("utf-8")
        image_contents.append(
            {
                "type": "image_url",
                "image_url": f"data:{image.mime_type};base64,{image_b64}",
            }
        )

    prompt = VOCAB_PROMPT if card_type == CardType.VOCAB else SENTENCE_PROMPT

    response = cached_completion(
        messages=[
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}] + image_contents,
            }
        ],
        safety_settings=[
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_ONLY_HIGH",
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_ONLY_HIGH",
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_ONLY_HIGH",
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_ONLY_HIGH",
            },
        ],
        response_format={"type": "json_object"},
    )

    cards = []
    result = json.loads(response)
    for card_data in result:
        card = RawFlashCard(
            front=card_data["front"],
            front_sub=card_data.get("front_sub") or "",
            front_context=card_data.get("front_context") or "",
            back=card_data["back"],
            back_context=card_data.get("back_context") or "",
        )
        cards.append(card)
    return cards


def analyze_pdf_images(
    images: List[ImageData], card_type: CardType, batch_size: int = 4
) -> List[RawFlashCard]:
    """Analyze all PDF page images and extract flashcards using parallel processing"""

    with multiprocessing.dummy.Pool(processes=4) as pool:
        # Process images in batches
        batch_results = pool.starmap(
            _analyze_image_batch,
            [
                (images[i : i + batch_size], card_type)
                for i in range(0, len(images), batch_size)
            ],
            chunksize=1,
        )

        # Flatten results and remove duplicates
        all_cards = [card for batch in batch_results for card in batch]
        return list({card.front: card for card in all_cards}.values())


@dataclass
class AnalyzePdfConfig:
    pdf_path: Path
    output_path: Path
    output_format: OutputFormat
    card_type: CardType
    progress_logger: ProgressLogger


def process_pdf_images(
    config: AnalyzePdfConfig,
):
    """Process PDF file by converting to images and analyzing with Gemini Vision"""
    # Extract images from PDF
    config.progress_logger("Extracting images from PDF")
    images = extract_images_from_pdf(config.pdf_path, PdfOptions())

    config.progress_logger(f"Analyzing {len(images)} pages")
    all_cards = analyze_pdf_images(images, card_type=config.card_type)
    config.progress_logger(f"Exporting to {config.output_format}")

    if config.output_format == OutputFormat.ANKI_PKG:
        create_anki_package(
            config.output_path,
            vocab_items=all_cards,
            deck_name=config.pdf_path.stem,
            audio_mapping={},
        )
    else:
        pdf_config = PDFGeneratorConfig(
            columns=3 if config.card_type == CardType.VOCAB else 2,
            rows=6 if config.card_type == CardType.VOCAB else 4,
            cards=all_cards,
            output_path=config.output_path,
        )
        create_flashcard_pdf(pdf_config)
