"""Standard Manuscript Format (Shunn / Publisher Ready) PDF Exporter.

Adheres strictly to traditional publishing guidelines:
- 1-inch margins all around.
- Standard typography (Courier or Times-Roman 12pt).
- Double-spaced lines (or standard 1.5 line height for clean reading).
- Header with author surname / SHORT TITLE / Page Number on right.
- Professional title page:
  - Top-left: Author Contact / Representation info.
  - Top-right: Approximate word count (rounded to nearest 1,000).
  - Center: Full Book Title & Byline.
- Clean chapter breaks starting 1/3 down the page.
- Centered "#" or "***" scene separators.
"""

from pathlib import Path
import io
import re
from typing import Optional, List, Dict, Any

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    PageBreak,
)
from reportlab.pdfgen import canvas


class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to dynamically compute and print running headers with total page counts."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_running_header(num_pages)
            super().showPage()
        super().save()

    def draw_running_header(self, page_count: int):
        # Omit header on the cover/title page (page 1)
        if self._pageNumber <= 1:
            return

        self.saveState()
        self.setFont("Courier", 10)
        
        # Standard Running Header: AUTHOR / TITLE / PAGE
        header_text = getattr(self, "running_header_text", "Vance / THE AFFLICTION OF LOVING YOU")
        text = f"{header_text}  /  {self._pageNumber}"
        
        # Place in top margin, 0.5 inch from top edge, right aligned
        self.drawRightString(8.5 * inch - 1.0 * inch, 11 * inch - 0.55 * inch, text)
        self.restoreState()


def clean_markdown_prose(text: str) -> List[str]:
    """Splits raw markdown into clean paragraph blocks suitable for reportlab."""
    paragraphs = []
    # Normalize line breaks
    normalized = re.sub(r'\r\n', '\n', text)
    blocks = re.split(r'\n\s*\n', normalized)
    
    for b in blocks:
        clean = b.strip()
        if not clean:
            continue
        # Strip header markers but keep text
        clean = re.sub(r'^#+\s*', '', clean)
        # Escape XML chars for ReportLab Paragraph
        clean = clean.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        paragraphs.append(clean)
    return paragraphs


def generate_manuscript_pdf(
    title: str,
    author_name: str = "David Vance",
    contact_email: str = "contact@aiuthor.io",
    genre: str = "Commercial Fiction",
    chapters: Optional[List[Dict[str, Any]]] = None,
    output_path: Optional[Path] = None,
    running_header: Optional[str] = None
) -> bytes:
    """Generates a publisher-grade Standard Manuscript Format PDF."""
    chapters = chapters or []
    
    # Calculate approximate word count
    total_words = 0
    for ch in chapters:
        text = ch.get("content", "")
        total_words += len(text.split())
    rounded_words = f"about {max(1, round(total_words / 1000) * 1000):,} words" if total_words > 0 else "Sample Chapter"

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer if output_path is None else str(output_path),
        pagesize=letter,
        leftMargin=1.0 * inch,
        rightMargin=1.0 * inch,
        topMargin=1.0 * inch,
        bottomMargin=1.0 * inch,
    )

    # Styles mimicking Courier 12pt Standard Manuscript
    body_style = ParagraphStyle(
        'ManuscriptBody',
        fontName='Courier',
        fontSize=11,
        leading=18,  # Standard clean line-height
        firstLineIndent=0.5 * inch,
        spaceAfter=0,
    )

    body_no_indent = ParagraphStyle(
        'ManuscriptBodyNoIndent',
        fontName='Courier',
        fontSize=11,
        leading=18,
        firstLineIndent=0,
        spaceAfter=0,
    )

    title_style = ParagraphStyle(
        'ManuscriptTitle',
        fontName='Courier-Bold',
        fontSize=16,
        leading=22,
        alignment=1,  # Center
    )

    byline_style = ParagraphStyle(
        'ManuscriptByline',
        fontName='Courier',
        fontSize=12,
        leading=16,
        alignment=1,
    )

    header_meta_left = ParagraphStyle(
        'MetaLeft',
        fontName='Courier',
        fontSize=10,
        leading=14,
    )

    header_meta_right = ParagraphStyle(
        'MetaRight',
        fontName='Courier',
        fontSize=10,
        leading=14,
        alignment=2,  # Right
    )

    chapter_head_style = ParagraphStyle(
        'ChapterHead',
        fontName='Courier-Bold',
        fontSize=13,
        leading=18,
        alignment=1,
    )

    scene_break_style = ParagraphStyle(
        'SceneBreak',
        fontName='Courier',
        fontSize=11,
        leading=18,
        alignment=1,
    )

    story = []

    # --- TITLE PAGE ---
    # Top info block (Left: Contact / Right: Word count)
    contact_block = f"{author_name}<br/>Represented by Aiuthor Engine<br/>{contact_email}"
    word_block = f"{rounded_words}<br/>{genre}"

    story.append(Paragraph(contact_block, header_meta_left))
    # Negative spacer trick or dual table can work, or simpler:
    story.append(Spacer(1, -0.4 * inch))
    story.append(Paragraph(word_block, header_meta_right))

    # Center Title block (about 1/3 down the page)
    story.append(Spacer(1, 2.5 * inch))
    story.append(Paragraph(title.upper(), title_style))
    story.append(Spacer(1, 0.25 * inch))
    story.append(Paragraph(f"by<br/><br/>{author_name}", byline_style))

    story.append(PageBreak())

    # --- CHAPTERS ---
    clean_short_title = re.sub(r'[^A-Za-z0-9 ]+', '', title).strip().upper()
    if len(clean_short_title) > 28:
        clean_short_title = clean_short_title[:25] + "..."
    author_surname = author_name.split()[-1] if author_name else "Author"

    for idx, ch in enumerate(chapters):
        ch_num = ch.get("chapter_number", idx + 1)
        ch_title = ch.get("title", f"Chapter {ch_num}")
        ch_content = ch.get("content", "")

        # Chapters start ~1/3 down the page in standard manuscript format
        story.append(Spacer(1, 1.5 * inch))
        story.append(Paragraph(f"CHAPTER {ch_num}", chapter_head_style))
        if ch_title and not ch_title.lower().startswith("chapter"):
            story.append(Spacer(1, 0.15 * inch))
            story.append(Paragraph(ch_title.upper(), chapter_head_style))
        story.append(Spacer(1, 0.5 * inch))

        paras = clean_markdown_prose(ch_content)
        if not paras:
            story.append(Paragraph("[No prose drafted for this chapter.]", body_no_indent))
        else:
            for p in paras:
                if p in ("***", "---", "#", "* * *"):
                    story.append(Spacer(1, 0.2 * inch))
                    story.append(Paragraph("#", scene_break_style))
                    story.append(Spacer(1, 0.2 * inch))
                else:
                    story.append(Paragraph(p, body_style))

        # Page break after every chapter
        if idx < len(chapters) - 1:
            story.append(PageBreak())

    # Build canvas with running headers
    def canvas_maker(*args, **kwargs):
        c = NumberedCanvas(*args, **kwargs)
        c.running_header_text = running_header or f"{author_surname} / {clean_short_title}"
        return c

    doc.build(story, canvasmaker=canvas_maker)

    if output_path:
        return output_path.read_bytes()
    return buffer.getvalue()
