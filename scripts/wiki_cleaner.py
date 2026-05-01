"""Shared MediaWiki HTML cleaning helpers for ZIM and API extraction."""

from __future__ import annotations

import re

from bs4 import BeautifulSoup
from markdownify import MarkdownConverter


def sanitize_wiki_text(text: str | None) -> str:
    """Extract a compact item name from MediaWiki alt/title text."""
    if not text:
        return ""

    parts = text.split("描述")
    if len(parts) > 1:
        text = parts[-1]
        text = re.split(r"\s+\d|(?<=\S)\s+(?:在|穿|戴于|[：:])", text)[0]
    else:
        text = re.sub(r"^[a-zA-Z0-9_\-.\s/:：]+", "", text)

    text = re.sub(
        r"的精灵图|链接到Minecraft中\S+|Sprite|Invicon|SlotSprite|BlockSprite",
        "",
        text,
    )
    text = text.strip(" ，。:：,[]()（）{}/\n\t")

    if not text or re.match(r"^[a-zA-Z0-9_\-./]+$", text):
        return ""
    if len(text) > 15:
        text = text[:15]
    return text


def clean_html(html_content: str):
    """Clean a MediaWiki HTML fragment and return the article body soup node."""
    # Fast regex cleanup to avoid building unneeded DOM nodes
    html_content = re.sub(r"\[\[\|\]\]", "", html_content)
    html_content = re.sub(r"<style[^>]*>.*?</style>", "", html_content, flags=re.DOTALL | re.IGNORECASE)
    html_content = re.sub(r"<script[^>]*>.*?</script>", "", html_content, flags=re.DOTALL | re.IGNORECASE)
    html_content = re.sub(r"<link[^>]*>", "", html_content, flags=re.IGNORECASE)

    soup = BeautifulSoup(html_content, "lxml")
    body = soup.find("div", class_="mw-parser-output") or soup

    # Pass 1: Prune massive noise branches (galleries, navboxes, hidden spans, etc.)
    noise_classes = ["searchaux", "gallery", "mw-jump-link", "printfooter", "mw-editsection", "ambox", "navbox"]
    for noise in body.find_all(class_=noise_classes):
        noise.decompose()

    for heading in body.find_all(["h2", "h3"]):
        if heading.get_text().strip() in ("画廊", "Gallery"):
            parent = heading.parent
            if parent:
                parent.decompose()
            else:
                heading.decompose()

    # Pass 2: Process the surviving useful elements in a single sweep
    for el in body.find_all(["img", "a", "span"]):
        if el.parent is None:
            continue
            
        if el.name == "img":
            text = sanitize_wiki_text(el.get("alt", ""))
            if not text:
                parent = el.parent
                for _ in range(3):
                    if parent and parent.get("title"):
                        text = sanitize_wiki_text(parent["title"])
                        if text:
                            break
                    parent = parent.parent if parent else None
            if text:
                el.replace_with(f" {text}。")
            else:
                el.decompose()
        elif el.name == "a":
            el.unwrap()
        elif el.name == "span" and "mcui" in el.get("class", []):
            _rewrite_mcui(el)

    return body


def html_to_markdown(body) -> str:
    """Convert a cleaned BeautifulSoup node to Markdown."""
    # Directly convert the DOM node to avoid double parsing by markdownify
    text = MarkdownConverter(heading_style="ATX").convert_soup(body)
    return text.replace(" ROWBREAK ", "<br>")


def _rewrite_mcui(mcui) -> None:
    mc_input = mcui.find("span", class_="mcui-input")
    if mc_input:
        row_parts = []
        rows = mc_input.find_all("span", class_="mcui-row") or [mc_input]
        for row in rows:
            slots = [_slot_text(slot) for slot in row.find_all("span", class_="invslot")]
            row_parts.append(" ".join(slots))
        mc_input.clear()
        mc_input.string = " ROWBREAK ".join(row_parts)

    mc_output = mcui.find("span", class_="mcui-output")
    if mc_output:
        for slot in mc_output.find_all("span", class_="invslot"):
            slot.string = _slot_text(slot)


def _slot_text(slot) -> str:
    title_val = ""
    item = slot.find(["img", "a", "span"], title=True)
    if item and item.get("title"):
        title_val = sanitize_wiki_text(item["title"])
    if not title_val:
        img_tag = slot.find("img", alt=True)
        title_val = sanitize_wiki_text(img_tag["alt"]) if img_tag else ""
    return f"[{title_val}]" if title_val else "[空]"
