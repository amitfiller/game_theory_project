#!/usr/bin/env python3
"""בניית PDF מעוצב מקובץ ה-markdown של האיבוק (עברית, RTL)."""
import re
from pathlib import Path

import markdown
from weasyprint import HTML

BASE = Path(__file__).parent
FONTS = Path("/home/user/fonts")
MD_FILE = BASE / "ai-guide-hebrew.md"
OUT_FILE = BASE / "ai-guide-hebrew.pdf"

PURPLE = "#5b21b6"
DARK = "#1e1b4b"
AMBER = "#f59e0b"

CSS = f"""
@font-face {{ font-family: 'Heebo'; src: url('{FONTS}/Heebo-Regular.ttf'); font-weight: 400; }}
@font-face {{ font-family: 'Heebo'; src: url('{FONTS}/Heebo-Medium.ttf'); font-weight: 500; }}
@font-face {{ font-family: 'Heebo'; src: url('{FONTS}/Heebo-Bold.ttf'); font-weight: 700; }}
@font-face {{ font-family: 'Heebo'; src: url('{FONTS}/Heebo-ExtraBold.ttf'); font-weight: 800; }}
@font-face {{ font-family: 'Heebo'; src: url('{FONTS}/Heebo-Black.ttf'); font-weight: 900; }}
@font-face {{ font-family: 'Assistant'; src: url('{FONTS}/Assistant-Regular.ttf'); font-weight: 400; }}
@font-face {{ font-family: 'Assistant'; src: url('{FONTS}/Assistant-Bold.ttf'); font-weight: 700; }}
@font-face {{ font-family: 'Noto Emoji'; src: url('{FONTS}/NotoEmoji-Safe.ttf'); }}

@page {{
    size: A4;
    margin: 2.2cm 2cm 2.4cm 2cm;
    @bottom-center {{
        content: counter(page);
        font-family: 'Heebo';
        font-size: 9pt;
        color: {PURPLE};
    }}
    @bottom-right {{
        content: "AI בלי להיות טכנולוגי";
        font-family: 'Heebo';
        font-size: 8pt;
        color: #9ca3af;
    }}
}}
@page cover {{
    margin: 0;
    @bottom-center {{ content: none; }}
    @bottom-right {{ content: none; }}
}}

html {{ direction: rtl; }}
body {{
    font-family: 'Heebo', 'DejaVu Sans', sans-serif;
    font-size: 11pt;
    line-height: 1.65;
    color: #1f2937;
    direction: rtl;
    text-align: right;
}}

/* ----- cover ----- */
.cover {{
    page: cover;
    page-break-after: always;
    width: 21cm; height: 29.7cm;
    box-sizing: border-box;
    background: linear-gradient(160deg, {DARK} 0%, #312e81 45%, {PURPLE} 100%);
    color: white;
    text-align: center;
    padding-top: 6.5cm;
}}
.cover .badge {{
    display: inline-block;
    background: {AMBER};
    color: {DARK};
    font-weight: 800;
    font-size: 11pt;
    padding: 0.18cm 0.7cm;
    border-radius: 1cm;
    margin-bottom: 1.2cm;
}}
.cover h1 {{
    font-size: 34pt;
    font-weight: 900;
    margin: 0 1.5cm 0.8cm 1.5cm;
    line-height: 1.25;
    page-break-before: avoid;
    background: none;
    border: none;
    padding: 0;
    color: white;
}}
.cover .subtitle {{
    font-size: 15pt;
    font-weight: 400;
    color: #ddd6fe;
    margin: 0 2.2cm;
    line-height: 1.6;
}}
.cover .divider {{
    display: inline-block;
    width: 3cm; height: 0.12cm;
    background: {AMBER};
    margin: 1.1cm 0;
    border-radius: 1cm;
}}
.cover .bullets {{
    font-size: 12pt;
    color: #ede9fe;
    line-height: 2.1;
}}
.cover .footer {{
    margin-top: 2.2cm;
    font-size: 10pt;
    color: #a5b4fc;
}}

/* ----- headings ----- */
h1 {{
    page-break-before: always;
    font-size: 22pt;
    font-weight: 900;
    color: {DARK};
    background: linear-gradient(90deg, #ede9fe, #ffffff);
    padding: 0.5cm 0.6cm;
    border-right: 0.22cm solid {PURPLE};
    border-radius: 0.2cm;
    margin: 0 0 0.6cm 0;
    line-height: 1.3;
}}
h2 {{
    font-size: 15pt;
    font-weight: 800;
    color: {PURPLE};
    margin: 0.8cm 0 0.25cm 0;
    padding-bottom: 0.1cm;
    border-bottom: 1.5pt solid #ede9fe;
    page-break-after: avoid;
}}
h3 {{
    font-size: 12.5pt;
    font-weight: 700;
    color: {DARK};
    margin: 0.55cm 0 0.15cm 0;
    page-break-after: avoid;
}}

p {{ margin: 0.22cm 0; }}
strong {{ color: {DARK}; }}
ul, ol {{ margin: 0.2cm 0; padding-right: 0.7cm; padding-left: 0; }}
li {{ margin: 0.1cm 0; }}

/* ----- prompt boxes (code blocks) ----- */
pre {{
    direction: rtl;
    text-align: right;
    font-family: 'Assistant', sans-serif;
    font-size: 10.5pt;
    background: #f5f3ff;
    border: 1.2pt dashed {PURPLE};
    border-radius: 0.25cm;
    padding: 0.45cm 0.5cm;
    margin: 0.3cm 0;
    white-space: pre-wrap;
    page-break-inside: avoid;
}}
pre::before {{
    content: "📋 פרומפט להעתקה";
    display: block;
    font-family: 'Heebo', 'Noto Emoji';
    font-weight: 700;
    font-size: 8.5pt;
    color: {PURPLE};
    margin-bottom: 0.2cm;
}}
code {{ font-family: 'Assistant', sans-serif; background: #f5f3ff; padding: 0 0.1cm; border-radius: 0.08cm; }}
pre code {{ background: none; padding: 0; }}

/* ----- tip boxes (blockquotes) ----- */
blockquote {{
    background: #fffbeb;
    border-right: 0.18cm solid {AMBER};
    border-radius: 0.2cm;
    padding: 0.35cm 0.5cm;
    margin: 0.35cm 0;
    page-break-inside: avoid;
}}
blockquote p {{ margin: 0; }}

/* ----- tables ----- */
table {{
    width: 100%;
    border-collapse: collapse;
    margin: 0.35cm 0;
    font-size: 10.5pt;
    page-break-inside: avoid;
}}
th {{
    background: {DARK};
    color: white;
    font-weight: 700;
    padding: 0.2cm 0.3cm;
    text-align: right;
}}
td {{
    padding: 0.18cm 0.3cm;
    border-bottom: 0.8pt solid #e5e7eb;
    text-align: right;
}}
tr:nth-child(even) td {{ background: #faf5ff; }}

hr {{ display: none; }}

.emoji {{ font-family: 'Noto Emoji', 'DejaVu Sans', sans-serif; }}
.cover .emoji {{ color: white; }}
"""

# weasyprint בוחר בפונט אימוג'י גם עבור ספרות אם הוא ב-font stack,
# לכן עוטפים רק את תווי האימוג'י עצמם ב-span ייעודי
EMOJI_RE = re.compile(
    "([\U0001F000-\U0001FAFF☀-➿⬀-⯿‼⁉™️]+)"
)


def wrap_emoji(html: str) -> str:
    return EMOJI_RE.sub(r'<span class="emoji">\1</span>', html)

COVER_HTML = """
<div class="cover">
  <div class="badge">מהדורת 2026 ✦ מעודכן</div>
  <h1>AI בלי להיות<br>טכנולוגי</h1>
  <div class="subtitle">המדריך הישראלי המלא: איך לחסוך 10+ שעות בשבוע<br>(ולהתחיל להרוויח) עם כלי AI חינמיים</div>
  <div><span class="divider"></span></div>
  <div class="bullets">
    ✓ בלי רקע טכני &nbsp;•&nbsp; ✓ כלים חינמיים &nbsp;•&nbsp; ✓ הכל בעברית<br>
    ✓ 50 פרומפטים מוכנים להעתקה &nbsp;•&nbsp; ✓ 5 מסלולי הכנסה
  </div>
  <div class="footer">מדריך דיגיטלי ✦ 8 פרקים מעשיים</div>
</div>
"""


def main():
    md_text = MD_FILE.read_text(encoding="utf-8")
    # העמוד הראשון במסמך ה-md הוא כותרת — מוחלף בעמוד שער מעוצב
    md_text = re.sub(r"^# AI בלי להיות טכנולוגי\n+## [^\n]+\n+---\n+", "", md_text)

    body = markdown.markdown(md_text, extensions=["tables", "nl2br", "fenced_code", "sane_lists"])
    # weasyprint מתעלם מ-start ברשימות ממוספרות — ממירים ל-counter-reset
    body = re.sub(
        r'<ol start="(\d+)">',
        lambda m: f'<ol style="counter-reset: list-item {int(m.group(1)) - 1}">',
        body,
    )
    html = (
        '<html dir="rtl" lang="he"><head><meta charset="utf-8">'
        f"<style>{CSS}</style></head><body>{wrap_emoji(COVER_HTML)}{wrap_emoji(body)}</body></html>"
    )
    HTML(string=html).write_pdf(OUT_FILE)
    print(f"OK: {OUT_FILE} ({OUT_FILE.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
