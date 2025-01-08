import base64
import json
import logging
from pathlib import Path
from typing import Generator, List

from openai import audio

from srt.config import cached_completion
from srt.generate_pdf import PDFGeneratorConfig, create_flashcard_pdf
from srt.images_from_pdf import ImageData, PdfOptions, extract_images_from_pdf
from srt.lib import (
    ConversionProgress,
    ConversionStage,
    create_anki_package,
)
from srt.schema import OutputFormat, RawFlashCard

logger = logging.getLogger(__name__)

VOCAB_PROMPT = """Analyze these pages from a book and create vocabulary flashcards for each word.

Focus on vocabulary that would be valuable for learning.
You may also create cards to explain novel grammar items.

For each item you find, create a flashcard with the following fields:

    front: str - the vocabulary word or grammar term
    front_sub: str - the reading or subtext for the front of the card
    front_context: str - a sentence demonstrating the word or grammar point
    back: str - the English translation or explanation
    back_context: str - a translation of the context sentence

The front and back context should be minimal sentences which demonstrate the
word or grammar point. They should _not_ be sentences from the text.

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

Assume the reader has a rough understanding of the language, don't include basic
words like "the" or "and" unless they are used in a novel way.

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
  "front_sub": "[あんない] [してあげる] よ？",
  "back": "Should I show you the way?"
}

Assume the reader has a rough understanding of the language, don't include basic sentences.

Return only valid JSON array of flashcards.
"""


def analyze_pdf_images(images: List[ImageData]) -> List[RawFlashCard]:
    """Analyze all PDF page images and extract flashcards"""

    # Convert all images to base64
    image_contents = []
    for image in images:
        image_b64 = base64.b64encode(image.content).decode("utf-8")
        image_contents.append(
            {
                "type": "image_url",
                "image_url": f"data:{image.mime_type};base64,{image_b64}",
            }
        )

    response = cached_completion(
        messages=[
            {
                "role": "user",
                "content": [{"type": "text", "text": SENTENCE_PROMPT}] + image_contents,
            }
        ],
        response_format={"type": "json_object"},
    )

    cards = []
    result = json.loads(response)
    for card_data in result:
        card = RawFlashCard(
            front=card_data["front"],
            front_sub=card_data.get("front_sub"),
            front_context=card_data.get("front_context"),
            back=card_data["back"],
            back_context=card_data.get("back_context"),
        )
        cards.append(card)

    # dedup
    cards = list({card.front: card for card in cards}.values())
    return cards


def process_pdf_images(
    pdf_path: Path, output_path: Path, output_format: OutputFormat
) -> Generator[ConversionProgress, None, None]:
    """Process PDF file by converting to images and analyzing with Gemini Vision"""

    yield ConversionProgress(
        stage=ConversionStage.STARTING, message="Starting PDF processing", progress=0
    )

    # Extract images from PDF
    yield ConversionProgress(
        stage=ConversionStage.PROCESSING,
        message="Extracting images from PDF",
        progress=10,
    )

    images = extract_images_from_pdf(pdf_path, PdfOptions())

    # Analyze all images together
    yield ConversionProgress(
        stage=ConversionStage.ANALYZING,
        message=f"Analyzing {len(images)} pages",
        progress=50,
    )

    all_cards = analyze_pdf_images(images)

    # Export cards
    yield ConversionProgress(
        stage=ConversionStage.PROCESSING,
        message=f"Exporting to {output_format}",
        progress=90,
    )

    if output_format == OutputFormat.ANKI_PKG:
        create_anki_package(output_path, all_cards, pdf_path.stem, audio_mapping={})
    else:
        config = PDFGeneratorConfig(
            columns=2,
            rows=4,
            cards=all_cards,
            output_path=output_path,
        )
        create_flashcard_pdf(config)

    yield ConversionProgress(
        stage=ConversionStage.COMPLETE,
        message="Processing complete",
        progress=100,
        filename=output_path.name,
    )
