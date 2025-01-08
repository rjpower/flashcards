from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import List, MutableSequence, Sequence, Tuple

import bs4
import pydantic
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.pdfmetrics import registerFontFamily
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    PageBreak,
    PageTemplate,
)
from reportlab.platypus.flowables import Flowable

from srt.schema import FlashCard

LINE_SPACING = 2
RUBY_PADDING = 2
DEFAULT_FONT_SIZE = 12
DEFAULT_RUBY_SIZE = 8
CONTEXT_FONT_SIZE = 10
MEANING_FONT_SIZE = 12
TERM_FONT_SIZE = 12

# Layout constants
CARD_MARGIN = 0.2 * inch
TEXT_BLOCK_SPACING = 12

PAGE_MARGIN = 0 * inch


# Register Japanese font
FONT_PATH = resources.files("srt") / "fonts" / "NotoSansJP-Regular.ttf"
pdfmetrics.registerFont(TTFont("NotoSans", str(FONT_PATH), asciiReadable=True))
registerFontFamily("NotoSans", normal="NotoSans")


class AnnotatedText(pydantic.BaseModel):
    """Represents ruby text: base<ruby>ruby<rt>annotation</rt></ruby>"""

    base: str
    ruby: str
    ruby_size: int = DEFAULT_RUBY_SIZE
    base_size: int = DEFAULT_FONT_SIZE

    def get_width(self, canvas) -> float:
        """Calculate the width this text will consume"""
        return canvas.stringWidth(self.base, "NotoSans", self.base_size)

    def get_height(self) -> float:
        """Calculate total height including ruby text"""
        return self.base_size + self.ruby_size + RUBY_PADDING

    def draw_at(self, canvas, x: float, y: float) -> float:
        """Draw at specified position and return width consumed"""
        canvas.saveState()

        base_width = self.get_width(canvas)
        ruby_width = canvas.stringWidth(self.ruby, "NotoSans", self.ruby_size)
        x_offset = (base_width - ruby_width) / 2

        # Draw ruby text
        canvas.setFont("NotoSans", self.ruby_size)
        canvas.drawString(x + x_offset, y + self.base_size + 2, self.ruby)

        # Draw base text
        canvas.setFont("NotoSans", self.base_size)
        canvas.drawString(x, y, self.base)

        canvas.restoreState()
        return base_width


class Text(pydantic.BaseModel):
    """Represents plain text"""

    text: str
    font_name: str
    font_size: int = 12

    def get_width(self, canvas) -> float:
        """Calculate the width this text will consume"""
        return canvas.stringWidth(self.text, self.font_name, self.font_size)

    def get_height(self) -> float:
        """Calculate height of text"""
        return self.font_size

    def draw_at(self, canvas, x: float, y: float) -> float:
        """Draw at specified position and return width consumed"""
        canvas.saveState()
        canvas.setFont(self.font_name, self.font_size)
        canvas.drawString(x, y, self.text)
        width = self.get_width(canvas)
        canvas.restoreState()
        return width


def draw_wrapped_text(
    canvas,
    text: str,
    x: float,
    y: float,
    width: float,
    font_name: str = "NotoSans",
    font_size: int = CONTEXT_FONT_SIZE,
) -> None:
    """Draw wrapped text at specified position

    Args:
        canvas: The ReportLab canvas to draw on
        text: The text to draw
        x: Starting x position
        y: Starting y position
        width: Available width for text
        font_name: Font to use
        font_size: Font size to use
    """
    if not text:
        return

    canvas.setFont(font_name, font_size)
    flowables = parse_ruby_markup(text, font_name)

    available_width = width - 2 * x
    current_x = x
    current_y = y
    line_start_x = x
    current_line: MutableSequence[Flowable] = []

    for item in flowables:
        space_width = (
            canvas.stringWidth(" ", font_name, font_size) if current_line else 0
        )

        # does the current item fit on the line, if so, add and continue
        if item.get_width(canvas) + current_x - line_start_x <= available_width:
            current_line.append(item)
            current_x += item.get_width(canvas) + space_width
            continue

        # If item is not text, draw current line and continue
        if not isinstance(item, Text):
            current_y = draw_text(
                canvas,
                current_line,
                line_start_x,
                current_y,
                font_name,
                font_size,
            )
            current_line = [item]
            current_x = line_start_x + item.get_width(canvas)
            continue

        # split the line, draw the current line and move on
        remaining_width = available_width - (current_x - line_start_x)
        current_text, remaining = wrap_text(
            item,
            canvas,
            remaining_width - space_width,
        )

        if current_text:
            current_line.append(current_text)

        current_y = draw_text(
            canvas,
            current_line,
            line_start_x,
            current_y,
            font_name,
            font_size,
        )
        current_line = []

        if remaining:
            current_line.append(remaining)
            current_x = line_start_x + remaining.get_width(canvas)

    # Draw any remaining items
    if current_line:
        current_y = draw_text(
            canvas,
            current_line,
            line_start_x,
            current_y,
            font_name,
            font_size,
        )


def draw_text(canvas, current_line, line_start_x, current_y, font_name, font_size):
    """Draw the current line of text and return the new y position"""
    if not current_line:
        return current_y

    draw_x = line_start_x
    for item in current_line:
        if isinstance(item, AnnotatedText):
            # Ruby text goes above baseline
            width = item.draw_at(canvas, draw_x, current_y - item.base_size)
        else:
            # Plain text aligns to baseline
            width = item.draw_at(canvas, draw_x, current_y - item.font_size)
        draw_x += width + canvas.stringWidth(" ", font_name, font_size)

    # Calculate line height based on actual content
    line_height = max(
        (item.base_size if isinstance(item, AnnotatedText) else item.font_size)
        for item in current_line
    )
    return current_y - line_height - LINE_SPACING


def wrap_text(
    text_obj: Text,
    canvas,
    remaining_width: float,
) -> Tuple[Text, Text]:
    """Split Text object into current line and remainder based on available width.

    Args:
        text_obj: The Text object to wrap
        canvas: The ReportLab canvas to measure text width
        remaining_width: Available width in points

    Returns:
        Tuple of (current_line, remaining_text) where:
        - current_line is Text object that fits within remaining_width
        - remaining_text is Text object that needs to wrap to next line
    """
    words = text_obj.text.split()
    current_line = []
    next_line = []

    current_width = 0
    space_width = canvas.stringWidth(" ", text_obj.font_name, text_obj.font_size)

    while words:
        word = words[0]
        word_width = canvas.stringWidth(word, text_obj.font_name, text_obj.font_size)
        if current_width + word_width <= remaining_width:
            current_line.append(word)
            current_width += word_width + space_width
            words.pop(0)
        else:
            break

    next_line = words

    return (
        Text(
            text=" ".join(current_line),
            font_name=text_obj.font_name,
            font_size=text_obj.font_size,
        ),
        Text(
            text=" ".join(next_line),
            font_name=text_obj.font_name,
            font_size=text_obj.font_size,
        ),
    )


def parse_ruby_markup(text: str, font_name: str) -> Sequence[Flowable]:
    """Process a potentially Ruby HTML string and return a list of Text and AnnotatedText objects"""
    if not text:
        return []

    # parse with bs4
    soup = bs4.BeautifulSoup(text, "html.parser")
    # iterate over all elements, adding Text and AnnotatedText objects to the result
    result = []
    for element in soup.children:
        if element.name == "ruby":
            base = element.contents[0]
            rt_tag = element.find("rt")
            ruby = rt_tag.string if rt_tag else ""
            result.append(AnnotatedText(base=str(base), ruby=str(ruby)))
        else:
            result.append(Text(text=str(element), font_name=font_name))

    return result


@dataclass
class CardFront(Flowable):
    """Custom flowable for drawing a flashcard"""

    card: FlashCard
    width: float
    height: float
    font_name: str

    def draw(self):
        """Draw the flashcard content"""
        # Draw border
        self.canv.setStrokeColor(colors.black)
        self.canv.rect(0, 0, self.width, self.height)

        # Draw main term
        draw_wrapped_text(
            self.canv,
            self.card.front,
            CARD_MARGIN,
            self.height - CARD_MARGIN,
            self.width,
            font_name=self.font_name,
            font_size=TERM_FONT_SIZE,
        )

        # Draw reading if present and different from term
        if self.card.front_sub and self.card.front_sub != self.card.front:
            draw_wrapped_text(
                self.canv,
                self.card.front_sub,
                CARD_MARGIN,
                self.height - CARD_MARGIN - TERM_FONT_SIZE - TEXT_BLOCK_SPACING,
                self.width,
                font_name=self.font_name,
                font_size=TERM_FONT_SIZE,
            )

        # Draw context if available
        if self.card.front_context:
            draw_wrapped_text(
                self.canv,
                self.card.front_context,
                CARD_MARGIN,
                CARD_MARGIN + 16,
                self.width,
                font_name=self.font_name,
            )


@dataclass
class CardBack(Flowable):
    """Custom flowable for drawing the back of a flashcard"""

    card: FlashCard
    width: float
    height: float
    font_name: str

    def draw(self):
        # Draw border
        self.canv.setStrokeColor(colors.black)
        self.canv.rect(0, 0, self.width, self.height)

        self.canv.setFont(self.font_name, MEANING_FONT_SIZE)
        meaning = self.card.back
        draw_wrapped_text(
            self.canv,
            meaning,
            CARD_MARGIN,
            self.height - CARD_MARGIN - 12,
            self.width,
            font_name=self.font_name,
            font_size=MEANING_FONT_SIZE,
        )
        # self.canv.drawString(CARD_MARGIN, self.height - CARD_MARGIN - 12, meaning)

        if self.card.back_context:
            draw_wrapped_text(
                self.canv,
                self.card.back_context,
                CARD_MARGIN,
                CARD_MARGIN + 16,
                self.width,
                font_name=self.font_name,
            )


@dataclass
class PDFGeneratorConfig:
    """Configuration for PDF generation"""

    cards: Sequence[FlashCard]
    output_path: Path
    columns: int = 3
    rows: int = 6

    front_font: str = "NotoSans"
    back_font: str = "Times-Roman"


def create_flashcard_pdf(config: PDFGeneratorConfig):
    """Generate PDF with flashcards in a grid layout"""
    config.output_path.unlink(missing_ok=True)

    doc = BaseDocTemplate(
        str(config.output_path),
        pagesize=letter,
        rightMargin=PAGE_MARGIN,
        leftMargin=PAGE_MARGIN,
        topMargin=PAGE_MARGIN,
        bottomMargin=PAGE_MARGIN,
    )

    # Calculate frame positions for 3x6 grid
    page_width = letter[0] - 2 * PAGE_MARGIN
    page_height = letter[1] - 2 * PAGE_MARGIN

    col_width = page_width / config.columns
    row_height = page_height / config.rows

    frames = []
    for row in range(config.rows):
        for col in range(config.columns):
            x = PAGE_MARGIN + col * col_width
            y = PAGE_MARGIN + row * row_height
            frame = Frame(
                x,
                y,
                col_width,
                row_height,
                leftPadding=0,
                bottomPadding=0,
                rightPadding=0,
                topPadding=0,
            )
            frames.append(frame)

    # Create page template with frames
    template = PageTemplate(id="card_grid", frames=frames)
    doc.addPageTemplates([template])

    # Create flowables
    elements = []
    cards_per_page = config.columns * config.rows

    for i in range(0, len(config.cards), cards_per_page):
        batch = config.cards[i : i + cards_per_page]

        # Front side of batch
        for item in batch:
            elements.append(
                CardFront(
                    card=item,
                    width=col_width,
                    height=row_height,
                    font_name=config.front_font,
                )
            )

        elements.append(PageBreak())

        # Back side of same batch (reversed horizontally for double-sided printing)
        # Reorder items to match horizontal flip
        back_batch: MutableSequence[FlashCard] = []
        for row in range(config.rows):
            row_start = row * config.columns
            row_items = batch[row_start : row_start + config.columns]
            back_batch.extend(reversed(row_items))

        for item in back_batch:
            if item:
                elements.append(
                    CardBack(
                        card=item,
                        width=col_width,
                        height=row_height,
                        font_name=config.back_font,
                    )
                )

        if i != len(config.cards) - 1:
            elements.append(PageBreak())

    # Build PDF
    doc.build(elements)
