from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import jinja2
from playwright.sync_api import sync_playwright

from srt.schema import FlashCard

CSS = """
@page {
    margin: 0;
    size: letter;
}

@media print {
    body {
        margin: 0;
    }
}

body {
    font-family: 'Noto Sans JP', sans-serif;
    margin: 0;
    padding: 0;
}

.page {
    width: 8.5in;
    height: 11in;
    position: relative;
    page-break-after: always;
    display: grid;
}

.grid {
    grid-template-columns: repeat(var(--columns), calc(8.5in / var(--columns)));
    grid-template-rows: repeat(var(--rows), calc(11in / var(--rows)));
}

.card {
    border-right: 1px solid black;
    border-bottom: 1px solid black;
    padding: 0.2in;
    height: calc(11in / var(--rows));
    width: calc(8.5in / var(--columns));
    display: flex;
    flex-direction: column;
    justify-content: flex-start;
    overflow: hidden;
    box-sizing: border-box;
}

.page {
    border-top: 1px solid black;
    border-left: 1px solid black;
}

.card-front {
    text-align: center;
}

.term {
    font-size: 20px;
    flex: 0 0 auto;
}

.reading {
    font-size: 16px;
    color: #666;
    margin-bottom: 0.1in;
    flex: 0 0 auto;
}

.context {
    font-size: 14px;
    color: #333;
    text-align: left;
    margin-top: 0.1in;
    flex: 1 1 auto;
    overflow: hidden;
}

.card-back {
    text-align: left;
}

.meaning {
    font-size: 16px;
    flex: 0 0 auto;
}

.back-context {
    font-size: 14px;
    color: #333;
    flex: 1 1 auto;
    overflow: hidden;
}

ruby {
    ruby-align: center;
}

rt {
    font-size: 0.5em;
    color: #666;
}
"""

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <link href="https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@400;700&display=swap" rel="stylesheet">
    <style>
        {{ css }}
    </style>
</head>
<body>
    {% for batch in batches %}
    {# Front side #}
    <div class="page grid" style="--columns: {{ config.columns }}; --rows: {{ config.rows }}">
        {% for card in batch %}
        <div class="card card-front">
            <div class="term">{{ card.front }}</div>
            {% if card.front_sub and card.front_sub != card.front %}
            <div class="reading">{{ card.front_sub }}</div>
            {% endif %}
            {% if card.front_context %}
            <div class="context">{{ card.front_context }}</div>
            {% endif %}
        </div>
        {% endfor %}
    </div>

    {# Back side (reversed horizontally for double-sided printing) #}
    <div class="page grid" style="--columns: {{ config.columns }}; --rows: {{ config.rows }}">
        {% for row in batch|batch(config.columns) %}
            {% for card in row|reverse %}
        <div class="card card-back">
            <div class="meaning">{{ card.back }}</div>
            {% if card.back_context %}
            <div class="back-context">{{ card.back_context }}</div>
            {% endif %}
        </div>
        {% endfor %}
        {% endfor %}
    </div>
    {% endfor %}
</body>
</html>
"""


@dataclass
class PDFGeneratorConfig:
    """Configuration for PDF generation"""

    cards: Sequence[FlashCard]
    output_path: Path
    columns: int = 3
    rows: int = 8


def batch_cards(cards: Sequence[FlashCard], batch_size: int):
    """Split cards into batches of specified size"""
    return [cards[i : i + batch_size] for i in range(0, len(cards), batch_size)]


def create_flashcard_pdf(config: PDFGeneratorConfig):
    """Generate PDF with flashcards using HTML and Playwright"""
    # Create Jinja environment with custom filters
    env = jinja2.Environment()
    env.filters["batch"] = lambda seq, size: [
        seq[i : i + size] for i in range(0, len(seq), size)
    ]

    # Prepare template
    template = env.from_string(HTML_TEMPLATE)

    # Calculate batches
    cards_per_page = config.columns * config.rows
    batches = batch_cards(config.cards, cards_per_page)

    # Render HTML
    html = template.render(css=CSS, batches=batches, config=config)

    # Generate PDF using Playwright
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True, args=["--no-sandbox", "--disable-web-security"]
        )

        page = browser.new_page()
        page.set_content(html)

        # Configure print options
        print_options = {
            "scale": 1.0,
            "margin": {"top": "0", "right": "0", "bottom": "0", "left": "0"},
            # "pageRanges": "1-",
            "format": "Letter",
        }

        # Generate PDF
        page.wait_for_function("document.fonts.ready")
        page.pdf(path=str(config.output_path), **print_options)
        browser.close()
