"""Converts the Phase 1-2 Markdown document to a print-ready HTML file with Mermaid and PDF styling."""
import os
import markdown

MD_PATH = os.path.join("docs", "Phase1_2_Project_Proposal_and_System_Design.md")
HTML_PATH = os.path.join("docs", "Phase1_2_Project_Proposal_and_System_Design.html")

CSS_STYLES = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    body {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        line-height: 1.6;
        color: #1f2937;
        max-width: 900px;
        margin: 0 auto;
        padding: 40px 20px;
    }
    
    h1 {
        color: #1e3a8a;
        border-bottom: 3px solid #3b82f6;
        padding-bottom: 12px;
        margin-top: 30px;
        font-size: 26px;
    }
    h2 {
        color: #1e40af;
        border-bottom: 1px solid #e5e7eb;
        padding-bottom: 8px;
        margin-top: 25px;
        font-size: 20px;
    }
    h3 {
        color: #374151;
        margin-top: 18px;
        font-size: 16px;
    }
    
    table {
        width: 100%;
        border-collapse: collapse;
        margin: 20px 0;
        font-size: 14px;
    }
    th, td {
        border: 1px solid #d1d5db;
        padding: 10px 14px;
        text-align: left;
    }
    th {
        background-color: #f3f4f6;
        color: #111827;
        font-weight: 600;
    }
    tr:nth-child(even) {
        background-color: #f9fafb;
    }
    
    pre {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 14px;
        overflow-x: auto;
        font-size: 13px;
    }
    code {
        font-family: 'Consolas', 'Courier New', monospace;
    }
    
    blockquote {
        border-left: 4px solid #3b82f6;
        background: #eff6ff;
        padding: 10px 16px;
        margin: 15px 0;
        border-radius: 0 6px 6px 0;
    }
    
    .cover-box {
        text-align: center;
        border: 2px solid #1e40af;
        border-radius: 10px;
        padding: 30px 20px;
        margin-bottom: 40px;
        background: linear-gradient(135deg, #f0f7ff 0%, #ffffff 100%);
    }
    .cover-title {
        font-size: 28px;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 10px;
    }
    .cover-subtitle {
        font-size: 16px;
        color: #4b5563;
        margin-bottom: 20px;
    }
    .print-btn {
        background-color: #2563eb;
        color: white;
        padding: 10px 20px;
        border: none;
        border-radius: 6px;
        font-weight: 600;
        cursor: pointer;
        margin-bottom: 20px;
    }
    
    @media print {
        .print-btn { display: none; }
        body { padding: 0; max-width: 100%; }
        h1, h2 { page-break-after: avoid; }
        table, pre { page-break-inside: avoid; }
    }
</style>
<script type="module">
    import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
    mermaid.initialize({ startOnLoad: true, theme: 'neutral' });
</script>
"""


def convert():
    with open(MD_PATH, "r", encoding="utf-8") as f:
        md_text = f.read()

    # Pre-process mermaid code blocks for rendering
    lines = md_text.splitlines()
    processed_lines = []
    in_mermaid = False
    mermaid_block = []

    for line in lines:
        if line.strip().startswith("```mermaid"):
            in_mermaid = True
            mermaid_block = []
        elif in_mermaid and line.strip() == "```":
            in_mermaid = False
            processed_lines.append('<div class="mermaid">')
            processed_lines.extend(mermaid_block)
            processed_lines.append("</div>")
        elif in_mermaid:
            mermaid_block.append(line)
        else:
            processed_lines.append(line)

    clean_md = "\n".join(processed_lines)
    html_body = markdown.markdown(clean_md, extensions=["tables", "fenced_code"])

    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Mini Project Phase 1–2: Project Proposal & System Design Document</title>
    {CSS_STYLES}
</head>
<body>
    <button class="print-btn" onclick="window.print()">🖨️ Print / Save as PDF (Ctrl + P)</button>
    <div class="cover-box">
        <div class="cover-title">🎓 EduPredict-MLOps</div>
        <div class="cover-subtitle">Assignment Submission Delay Prediction System</div>
        <p><strong>Course:</strong> Machine Learning Operations (MLOps) Mini Project</p>
        <p><strong>Deliverable:</strong> Phase 1–2: Project Proposal & System Design Document</p>
    </div>
    {html_body}
</body>
</html>
"""

    with open(HTML_PATH, "w", encoding="utf-8") as f:
        f.write(full_html)
    print(f"Successfully generated print-ready document at {HTML_PATH}")


if __name__ == "__main__":
    convert()
