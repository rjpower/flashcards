import re
from importlib import resources
from pathlib import Path
from typing import List, Sequence, Tuple

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
    NextPageTemplate,
    PageBreak,
    PageTemplate,
)
from reportlab.platypus.flowables import Flowable

from srt.schema import VocabItem

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

MARGIN = 0.2 * inch
PAGE_MARGIN = 0 * inch
COLUMNS = 3
ROWS = 8


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
    if not text_obj.text:
        return Text(
            text="", font_name=text_obj.font_name, font_size=text_obj.font_size
        ), Text(text="", font_name=text_obj.font_name, font_size=text_obj.font_size)

    words = text_obj.text.split()
    current_line = []
    next_line = []

    current_width = 0
    space_width = canvas.stringWidth(" ", text_obj.font_name, text_obj.font_size)

    for word in words:
        word_width = canvas.stringWidth(word, text_obj.font_name, text_obj.font_size)
        if current_width + word_width <= remaining_width:
            current_line.append(word)
            current_width += word_width + space_width
        else:
            next_line.append(word)

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


class ContextText(pydantic.BaseModel):
    """Represents wrapped context text"""

    text: str
    font_size: int = CONTEXT_FONT_SIZE
    font_name: str = "NotoSans"

    def draw_at(self, canvas, x: float, y: float, width: float) -> None:
        """Draw wrapped context text at specified position"""
        if not self.text:
            return

        canvas.setFont(self.font_name, self.font_size)
        flowables = parse_ruby_markup(self.text, self.font_name)

        available_width = width - 2 * x
        current_x = x
        current_y = y
        line_start_x = x
        current_line = []

        for item in flowables:
            space_width = (
                canvas.stringWidth(" ", self.font_name, self.font_size)
                if current_line
                else 0
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
                    self.font_name,
                    self.font_size,
                )
                current_line = []
                continue

            # split the line, draw the current line and move on
            remaining_width = available_width - (current_x - line_start_x)
            current_text, remaining = wrap_text(
                item,
                canvas,
                remaining_width - space_width,
            )

            if current_text.text:
                current_line.append(current_text)

            current_y = draw_text(
                canvas,
                current_line,
                line_start_x,
                current_y,
                self.font_name,
                self.font_size,
            )
            current_line = []

            if remaining:
                current_line.append(remaining)
                current_x = line_start_x + remaining.get_width(canvas)

        # Draw any remaining items
        if current_line:
            if "cool" in str(current_line):
                print(current_line)
            current_y = draw_text(
                canvas,
                current_line,
                line_start_x,
                current_y,
                self.font_name,
                self.font_size,
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


def truncate_text(text: str, max_length: int) -> str:
    """Truncate text to max_length adding ellipsis if needed"""
    if not text:
        return ""
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


class FlashCard(Flowable):
    """Custom flowable for drawing a flashcard"""

    def __init__(
        self, vocab_item: VocabItem, side: str, card_width: float, card_height: float
    ):
        super().__init__()
        self.vocab_item = vocab_item
        self.side = side
        self.width = card_width
        self.height = card_height

    def draw(self):
        """Draw the flashcard content"""
        # Draw border
        self.canv.setStrokeColor(colors.black)
        self.canv.rect(0, 0, self.width, self.height)

        if self.side == "front":
            self.canv.setFont("NotoSans", TERM_FONT_SIZE)
            # Process and draw term with ruby text
            current_x = MARGIN
            current_y = self.height - CARD_MARGIN - TERM_FONT_SIZE

            flowables = parse_ruby_markup(self.vocab_item.term, font_name="NotoSans")
            for item in flowables:
                current_x += item.draw_at(self.canv, current_x, current_y)

            # Draw reading if different from term
            if self.vocab_item.reading:
                term_without_ruby = "".join(
                    item.base if isinstance(item, AnnotatedText) else item.text
                    for item in flowables
                )
                if self.vocab_item.reading != term_without_ruby:
                    text = Text(text=self.vocab_item.reading, font_name="NotoSans")
                    text.draw_at(
                        self.canv,
                        CARD_MARGIN,
                        self.height - CARD_MARGIN - TERM_FONT_SIZE - TEXT_BLOCK_SPACING,
                    )

            # Draw context if available
            if self.vocab_item.context_jp:
                context = ContextText(
                    text=self.vocab_item.context_jp, font_name="NotoSans"
                )
                context.draw_at(self.canv, MARGIN, MARGIN + 16, self.width)
        else:
            self.canv.setFont("Times-Roman", MEANING_FONT_SIZE)
            meaning = truncate_text(self.vocab_item.meaning, 40)
            self.canv.drawString(MARGIN, self.height - MARGIN - 12, meaning)

            if self.vocab_item.context_en:
                context = ContextText(
                    text=self.vocab_item.context_en, font_name="Times-Roman"
                )
                context.draw_at(self.canv, MARGIN, MARGIN + 16, self.width)


def create_flashcard_pdf(vocab_items: List[VocabItem], output_path: Path):
    """Generate PDF with flashcards in a grid layout"""
    output_path.unlink(missing_ok=True)

    doc = BaseDocTemplate(
        str(output_path),
        pagesize=letter,
        rightMargin=PAGE_MARGIN,
        leftMargin=PAGE_MARGIN,
        topMargin=PAGE_MARGIN,
        bottomMargin=PAGE_MARGIN,
    )

    # Calculate frame positions for 3x6 grid
    page_width = letter[0] - 2 * PAGE_MARGIN
    page_height = letter[1] - 2 * PAGE_MARGIN

    col_width = page_width / COLUMNS
    row_height = page_height / ROWS

    frames = []
    for row in range(ROWS):
        for col in range(COLUMNS):
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

    # Process cards in batches of 24 (fills 3x8 grid on front and back)
    for i in range(0, len(vocab_items), 24):
        batch = vocab_items[i : i + 24]

        # Front side of batch
        for item in batch:
            elements.append(FlashCard(item, "front", col_width, row_height))

        elements.append(PageBreak())

        # Back side of same batch (reversed horizontally for double-sided printing)
        # Reorder items to match horizontal flip
        back_batch = []
        for row in range(ROWS):
            row_start = row * COLUMNS
            row_items = batch[row_start : row_start + COLUMNS]
            back_batch.extend(reversed(row_items))

        for item in back_batch:
            if item:  # Check for None in case batch isn't full
                elements.append(FlashCard(item, "back", col_width, row_height))

        if i != len(vocab_items) - 1:
            elements.append(PageBreak())

    # Build PDF
    doc.build(elements)
