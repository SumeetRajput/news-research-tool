"""
report.py
---------
Turns a research session into files the user can keep: PDF, Markdown, or a CSV
of sources.

Everything returns bytes rather than writing to disk, because Streamlit's
st.download_button() wants bytes and the app may be running on a server where
there is nowhere sensible to write.
"""

from __future__ import annotations

import csv
import io
import re
from datetime import datetime
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import HRFlowable, Paragraph, SimpleDocTemplate, Spacer

ACCENT = colors.HexColor("#1f4e79")


def _escape(text: str) -> str:
    """ReportLab Paragraphs parse a mini-XML, so bare & < > must be escaped."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _markup(text: str) -> str:
    """Convert the bits of markdown the model emits into ReportLab tags."""
    safe = _escape(text)
    safe = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", safe)
    # Make citation markers visually distinct.
    safe = re.sub(r"\[(\d+)\]", r'<font color="#1f4e79"><b>[\1]</b></font>', safe)
    return safe


# --------------------------------------------------------------------------
# Markdown
# --------------------------------------------------------------------------

def to_markdown(topic: str, entries: list[dict[str, Any]]) -> bytes:
    lines = [
        f"# Research brief: {topic}",
        "",
        f"*Generated {datetime.now():%d %B %Y, %H:%M} — Equity Research News Tool*",
        "",
    ]
    for i, entry in enumerate(entries, 1):
        lines += [f"## {i}. {entry['question']}", "", entry["answer"], ""]
        if entry.get("sources"):
            lines.append("**Sources**")
            lines.append("")
            for s in entry["sources"]:
                lines.append(f"{s['n']}. [{s['title']}]({s['url']}) — {s['source']}, {s['published_at']}")
            lines.append("")
        lines.append("---")
        lines.append("")
    lines.append(
        "*Answers are generated from the cited articles only. Verify against the "
        "original sources before relying on any figure.*"
    )
    return "\n".join(lines).encode("utf-8")


# --------------------------------------------------------------------------
# CSV of sources
# --------------------------------------------------------------------------

def sources_to_csv(entries: list[dict[str, Any]]) -> bytes:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["question", "ref", "title", "publication", "published", "url"])
    for entry in entries:
        for s in entry.get("sources", []):
            writer.writerow(
                [entry["question"], s["n"], s["title"], s["source"], s["published_at"], s["url"]]
            )
    return buffer.getvalue().encode("utf-8")


# --------------------------------------------------------------------------
# PDF
# --------------------------------------------------------------------------

def to_pdf(topic: str, entries: list[dict[str, Any]]) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title=f"Research brief: {topic}",
        author="Equity Research News Tool",
    )

    base = getSampleStyleSheet()
    styles = {
        "title": ParagraphStyle(
            "T", parent=base["Title"], fontSize=20, textColor=ACCENT, spaceAfter=4
        ),
        "meta": ParagraphStyle(
            "M", parent=base["Normal"], fontSize=8.5, textColor=colors.grey, spaceAfter=14
        ),
        "q": ParagraphStyle(
            "Q",
            parent=base["Heading2"],
            fontSize=12.5,
            textColor=ACCENT,
            spaceBefore=14,
            spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "B",
            parent=base["Normal"],
            fontSize=10,
            leading=15,
            alignment=TA_JUSTIFY,
            spaceAfter=6,
        ),
        "bullet": ParagraphStyle(
            "BU",
            parent=base["Normal"],
            fontSize=10,
            leading=15,
            leftIndent=12,
            spaceAfter=3,
        ),
        "srchead": ParagraphStyle(
            "SH", parent=base["Normal"], fontSize=9, textColor=ACCENT, spaceBefore=8, spaceAfter=4
        ),
        "src": ParagraphStyle("S", parent=base["Normal"], fontSize=8.5, leading=12),
        "foot": ParagraphStyle(
            "F", parent=base["Normal"], fontSize=8, textColor=colors.grey, spaceBefore=16
        ),
    }

    story: list[Any] = [
        Paragraph(_escape(f"Research brief: {topic}"), styles["title"]),
        Paragraph(
            f"Generated {datetime.now():%d %B %Y at %H:%M} &middot; "
            f"{len(entries)} question{'s' if len(entries) != 1 else ''}",
            styles["meta"],
        ),
        HRFlowable(width="100%", thickness=1, color=ACCENT, spaceAfter=4),
    ]

    for i, entry in enumerate(entries, 1):
        story.append(Paragraph(f"{i}. {_escape(entry['question'])}", styles["q"]))

        for block in [b for b in entry["answer"].split("\n") if b.strip()]:
            stripped = block.strip()
            if stripped.startswith(("- ", "* ", "• ")):
                story.append(
                    Paragraph(
                        f"&bull;&nbsp;&nbsp;{_markup(stripped[2:])}",
                        styles["bullet"],
                    )
                )
            else:
                story.append(Paragraph(_markup(stripped), styles["body"]))

        if entry.get("sources"):
            story.append(Paragraph("<b>Sources</b>", styles["srchead"]))
            for s in entry["sources"]:
                story.append(
                    Paragraph(
                        f"<b>[{s['n']}]</b> {_escape(s['title'])} &mdash; "
                        f"{_escape(s['source'])}, {s['published_at']}<br/>"
                        f'<font size="7.5" color="#666666">{_escape(s["url"])}</font>',
                        styles["src"],
                    )
                )
        story.append(Spacer(1, 6))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey))

    story.append(
        Paragraph(
            "Answers are generated from the cited articles only. Verify against the "
            "original sources before relying on any figure.",
            styles["foot"],
        )
    )

    doc.build(story)
    return buffer.getvalue()


def safe_filename(topic: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", topic).strip("-").lower()[:40] or "research"
    return f"{slug}-{datetime.now():%Y%m%d-%H%M}"
