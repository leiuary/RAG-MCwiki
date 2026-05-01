"""HTML -> Markdown batch worker used by wiki_manager update."""

from __future__ import annotations

import json
import os
from urllib.parse import quote

try:
    from wiki_cleaner import clean_html, html_to_markdown
except ImportError:  # pragma: no cover - package import fallback
    from scripts.wiki_cleaner import clean_html, html_to_markdown


def process_batch(items: list[tuple[str, str, str]]) -> dict:
    """Process (title, html_path, out_path) tuples and write Markdown files."""
    processed = 0
    errors: list[dict] = []

    for title, html_path, out_path in items:
        if os.path.exists(out_path):
            processed += 1
            continue

        try:
            with open(html_path, "r", encoding="utf-8") as f:
                html_raw = f.read()

            body = clean_html(html_raw)
            markdown = html_to_markdown(body)
            encoded_title = quote(title.replace(" ", "_"), safe="")

            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(f"# {title}\n\n")
                f.write(f"Source: https://zh.minecraft.wiki/w/{encoded_title}\n\n")
                f.write(markdown)

            processed += 1
        except Exception as e:
            errors.append({
                "title": title,
                "error": str(e),
                "type": type(e).__name__,
            })

    return {"processed": processed, "errors": errors}
