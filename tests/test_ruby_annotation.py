from io import BytesIO

import pytest
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from srt.flashcard_pdf import AnnotatedText, ContextText, Text, parse_ruby_markup


@pytest.fixture
def test_canvas():
    buffer = BytesIO()
    return canvas.Canvas(buffer, pagesize=letter)


@pytest.mark.parametrize(
    "input_text,expected_output",
    [
        (
            "Plain text",
            [Text(text="Plain text", font_size=12)],
        ),
        (
            "<ruby>漢字<rt>かんじ</rt></ruby>",
            [AnnotatedText(base="漢字", ruby="かんじ", ruby_size=8, base_size=12)],
        ),
        (
            "Mix of <ruby>漢字<rt>かんじ</rt></ruby> and text",
            [
                Text(text="Mix of ", font_size=12),
                AnnotatedText(base="漢字", ruby="かんじ", ruby_size=8, base_size=12),
                Text(text=" and text", font_size=12),
            ],
        ),
        (
            "",  # Empty string
            [],
        ),
        (
            "<ruby>食<rt>た</rt></ruby>べる",  # Common Japanese pattern
            [
                AnnotatedText(base="食", ruby="た", ruby_size=8, base_size=12),
                Text(text="べる", font_size=12),
            ],
        ),
    ],
)
def test_process_ruby_text(input_text, expected_output):
    """Test process_ruby_text with various input patterns"""
    result = parse_ruby_markup(input_text)
    assert len(result) == len(expected_output)

    for actual, expected in zip(result, expected_output):
        assert type(actual) is type(expected), f"Expected {expected} but got {actual}"
        if isinstance(actual, Text):
            assert actual.text == expected.text
            assert actual.font_size == expected.font_size
        elif isinstance(actual, AnnotatedText):
            assert actual.base == expected.base
            assert actual.ruby == expected.ruby
            assert actual.ruby_size == expected.ruby_size
            assert actual.base_size == expected.base_size


def test_text_draw_at(test_canvas):
    """Test Text.draw_at returns correct width"""
    text = Text(text="Test")
    width = text.draw_at(test_canvas, 100, 100)
    assert width > 0


def test_annotated_text_draw_at(test_canvas):
    """Test AnnotatedText.draw_at returns correct width"""
    text = AnnotatedText(base="漢字", ruby="かんじ")
    width = text.draw_at(test_canvas, 100, 100)
    assert width > 0


def test_annotated_text_get_width(test_canvas):
    """Test AnnotatedText.get_width returns expected width"""
    text = AnnotatedText(base="漢字", ruby="かんじ")
    width = text.get_width(test_canvas)
    assert width > 0
    # Width should match what draw_at returns
    assert width == text.draw_at(test_canvas, 100, 100)


def test_context_text_wrap_and_draw(test_canvas):
    """Test ContextText wrapping and drawing"""
    # Test with default font
    text = ContextText(text="This is a long text that should wrap to multiple lines")
    text.draw_at(test_canvas, 100, 100, 200)  # Width of 200 should force wrapping

    # Test with specific font
    text = ContextText(text="This is a long text", font_name="Times-Roman")
    text.draw_at(test_canvas, 100, 100, 200)

    # Test mixed content with ruby
    mixed_text = "Plain text with <ruby>漢字<rt>かんじ</rt></ruby> and more text"
    text = ContextText(text=mixed_text)
    text.draw_at(test_canvas, 100, 100, 200)


def test_text_wrapping():
    """Test text wrapping function"""
    from reportlab.pdfgen import canvas as pdf_canvas

    test_canvas = pdf_canvas.Canvas(BytesIO())
    from srt.flashcard_pdf import wrap_text, Text

    text_obj = Text(
        text="This is a test sentence", font_name="Times-Roman", font_size=12
    )
    current, remaining = wrap_text(text_obj, test_canvas, 50)
    assert current.text  # Should have some text on current line
    assert remaining.text  # Should have some text remaining
    assert current.font_name == "Times-Roman"
    assert current.font_size == 12

    # Test empty string
    empty_text = Text(text="", font_name="Times-Roman", font_size=12)
    current, remaining = wrap_text(empty_text, test_canvas, 50)
    assert current.text == ""
    assert remaining.text == ""
    assert current.font_name == empty_text.font_name
    assert current.font_size == empty_text.font_size


def test_text_heights():
    """Test height calculations for different text types"""
    plain_text = Text(text="Test", font_size=12)
    assert plain_text.get_height() == 12

    ruby_text = AnnotatedText(base="漢字", ruby="かんじ", base_size=12, ruby_size=8)
    # Height should include base size + ruby size + padding
    assert ruby_text.get_height() == 24  # 12 + 8 + 4
