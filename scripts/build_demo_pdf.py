"""Rebuild the shipped PDF from UTF-8 source and an embedded redistributable font."""

import sys
from html import escape
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

ROOT = Path(__file__).resolve().parents[1]


def build(output=ROOT / "output/pdf/mayak-demo.pdf"):
    pdfmetrics.registerFont(TTFont("Demo", str(ROOT / "assets/DejaVuSans.ttf")))
    styles = {
        "h1": ParagraphStyle(
            "h1",
            fontName="Demo",
            fontSize=22,
            leading=28,
            textColor=colors.HexColor("#163B46"),
            spaceAfter=20,
        ),
        "h2": ParagraphStyle(
            "h2",
            fontName="Demo",
            fontSize=13,
            leading=18,
            textColor=colors.HexColor("#163B46"),
            spaceBefore=14,
            spaceAfter=7,
        ),
        "p": ParagraphStyle(
            "p", fontName="Demo", fontSize=10.5, leading=16, spaceAfter=9, alignment=TA_LEFT
        ),
    }
    flow = []
    for block in (ROOT / "data/mayak-demo.txt").read_text().split("\n\n"):
        for line in block.strip().splitlines():
            if line == "---PAGE---":
                flow.append(PageBreak())
            elif line.startswith("# "):
                flow.append(Paragraph(escape(line[2:]), styles["h1"]))
            elif line.startswith("## "):
                flow.append(Paragraph(escape(line[3:]), styles["h2"]))
            elif line:
                flow.append(Paragraph(escape(line), styles["p"]))
        flow.append(Spacer(1, 2))

    def footer(canvas, doc):
        canvas.setFont("Demo", 8)
        canvas.setFillColor(colors.HexColor("#53656C"))
        canvas.drawString(44, 28, "Маяк / демонстрационный документ / вымышленные факты")
        canvas.drawRightString(A4[0] - 44, 28, str(doc.page))

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    SimpleDocTemplate(
        str(output),
        pagesize=A4,
        rightMargin=44,
        leftMargin=44,
        topMargin=40,
        bottomMargin=48,
        title="Маяк: демонстрационный документ",
        author="RAG-lite demo",
        invariant=1,
    ).build(flow, onFirstPage=footer, onLaterPages=footer)
    print(output)


if __name__ == "__main__":
    build(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "output/pdf/mayak-demo.pdf")
