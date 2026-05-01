"""Minecraft Wiki data management tool.

Usage:
  python scripts/wiki_manager.py                         # interactive TUI
  python scripts/wiki_manager.py extract <title>          # extract one page from ZIM and API
  python scripts/wiki_manager.py crawl --title <title>    # fetch one page from API as JSON
  python scripts/wiki_manager.py meta                     # update cloud metadata
  python scripts/wiki_manager.py compare <title>          # compare ZIM and API Markdown
  python scripts/wiki_manager.py export [-w N]            # export ZIM pages to Markdown
  python scripts/wiki_manager.py update [-w N]            # export pages changed online
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from urllib.parse import quote
import requests

try:
    from wiki_cleaner import clean_html, html_to_markdown
except ImportError:  # pragma: no cover - package import fallback
    from test.wiki_cleaner import clean_html, html_to_markdown

try:
    from libzim.reader import Archive as ZimArchive

    HAS_ZIM = True
except ImportError:
    ZimArchive = None
    HAS_ZIM = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ZIM_PATH = "data/zh.minecraft.wiki_zh_all_2026-03.zim"
METADATA_FILE = "data/page_metadata.json"
API_CANDIDATES = [
    "https://zh.minecraft.wiki/api.php",
    "https://zh.minecraft.wiki/w/api.php",
]
USER_AGENT = "mcwiki-tools/0.2 (+https://zh.minecraft.wiki/)"

EXPORT_OUTPUT_DIR = "data/markdown"
EXPORT_REDIRECT_FILE = "data/redirects.json"
EXPORT_ERROR_LOG = "data/extract_errors.jsonl"
HTML_CACHE_DIR = "data/html_cache"
ZIM_HTML_CACHE_DIR = os.path.join(HTML_CACHE_DIR, "zim")
API_HTML_CACHE_DIR = os.path.join(HTML_CACHE_DIR, "api")
CLOUD_PROGRESS_FILE = "test/cloud_update_progress.json"

REQUEST_DELAY_SECONDS = 0.1
REVISION_BATCH_SIZE = 50


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def safe_filename(title: str) -> str:
    safe = re.sub(r'[<>:"/\\|?*]', "_", title).strip(" .")
    return safe or "untitled"


def build_page_url(title: str) -> str:
    return f"https://zh.minecraft.wiki/w/{quote(title.replace(' ', '_'), safe='')}"


def ensure_parent(path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


def read_json(path: str, default: dict) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return default.copy()
    except json.JSONDecodeError:
        return default.copy()


def write_json(path: str, data: dict) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def open_zim_archive(zim_path: str):
    if not HAS_ZIM or ZimArchive is None:
        raise RuntimeError("libzim is not installed")
    if not os.path.exists(zim_path):
        raise FileNotFoundError(f"ZIM file not found: {zim_path}")

    devnull = os.open(os.devnull, os.O_WRONLY)
    old_fd = os.dup(2)
    os.dup2(devnull, 2)
    os.close(devnull)
    try:
        return ZimArchive(zim_path)
    finally:
        os.dup2(old_fd, 2)
        os.close(old_fd)


# ---------------------------------------------------------------------------
# Source layer: API and ZIM
# ---------------------------------------------------------------------------

_api_session = None

def api_request(params: dict, max_retries: int = 3) -> dict:
    global _api_session
    if _api_session is None:
        _api_session = requests.Session()
        _api_session.headers.update({
            "Accept": "application/json",
            "User-Agent": USER_AGENT,
        })

    last_err: Exception | None = None

    for api_url in API_CANDIDATES:
        for attempt in range(max_retries + 1):
            try:
                # `requests` automatically handles urlencoding and sets Content-Type for POST data
                resp = _api_session.post(api_url, data=params, timeout=30)
                resp.raise_for_status()
                return resp.json()
            except Exception as e:
                last_err = e
                if attempt < max_retries:
                    time.sleep(2 ** attempt)

    if last_err:
        raise last_err
    raise RuntimeError("API request failed without an exception")


def fetch_from_zim(title: str, zim_path: str = ZIM_PATH) -> tuple[str, str] | None:
    if not HAS_ZIM or not os.path.exists(zim_path):
        return None

    try:
        archive = open_zim_archive(zim_path)
        entry = archive.get_entry_by_title(title)
        item = entry.get_item()
        return bytes(item.content).decode("utf-8"), entry.title
    except Exception:
        return None


def fetch_from_api(title: str) -> tuple[str, str] | None:
    data = api_request({
        "action": "parse",
        "page": title,
        "prop": "text|displaytitle",
        "redirects": 1,
        "format": "json",
    })
    if "error" in data:
        return None
    parsed = data["parse"]
    return parsed["text"]["*"], parsed.get("title") or title


def fetch_page_html(title: str) -> str | None:
    result = fetch_from_api(title)
    return result[0] if result else None


def extract_page_html(
    title: str,
    source: str = "zim",
    zim_path: str = ZIM_PATH,
) -> tuple[str, str, str] | None:
    result = None
    if source in ("zim", "both"):
        result = fetch_from_zim(title, zim_path=zim_path)
    if not result and source in ("api", "both"):
        result = fetch_from_api(title)
    if not result:
        return None

    html, page_title = result
    markdown = html_to_markdown(clean_html(html))
    return markdown, page_title, build_page_url(page_title)


# ---------------------------------------------------------------------------
# Export layer
# ---------------------------------------------------------------------------

_worker_zim = None


def _worker_init(zim_path: str) -> None:
    global _worker_zim
    _worker_zim = open_zim_archive(zim_path)


def _worker_cache_html_titles(titles: list[str]) -> dict:
    done_titles = existing_markdown_stems()
    cached_html = existing_html_stems(ZIM_HTML_CACHE_DIR)
    html_cached = 0
    html_items: list[tuple[str, str, str]] = []
    redirects: dict[str, str] = {}
    errors: list[dict] = []

    for title in titles:
        try:
            entry = _worker_zim.get_entry_by_title(title)
        except Exception as e:
            errors.append({
                "title": title,
                "error": "未在 ZIM 离线包中找到对应条目",
                "type": type(e).__name__,
            })
            continue

        if entry.is_redirect:
            try:
                redirects[entry.title] = entry.get_redirect_entry().title
            except Exception:
                pass
            continue

        stem = safe_filename(entry.title)

        try:
            item = entry.get_item()
            if "text/html" not in item.mimetype:
                continue

            html_raw = bytes(item.content).decode("utf-8")
            html_path = os.path.join(ZIM_HTML_CACHE_DIR, stem + ".html")
            if stem not in cached_html:
                write_raw_html(html_path, entry.title, html_raw, source="ZIM")
                cached_html.add(stem)
                html_cached += 1

            if stem in done_titles:
                continue

            out_path = os.path.join(EXPORT_OUTPUT_DIR, stem + ".md")
            html_items.append((entry.title, html_path, out_path))
        except Exception as e:
            errors.append({
                "title": entry.title,
                "error": str(e),
                "type": type(e).__name__,
            })

    return {
        "html_cached": html_cached,
        "html_items": html_items,
        "redirects": redirects,
        "errors": errors,
    }


def existing_markdown_stems() -> set[str]:
    if not os.path.exists(EXPORT_OUTPUT_DIR):
        return set()
    return {
        fname[:-3]
        for fname in os.listdir(EXPORT_OUTPUT_DIR)
        if fname.endswith(".md")
    }


def existing_html_stems(cache_dir: str) -> set[str]:
    if not os.path.exists(cache_dir):
        return set()
    return {
        fname[:-5]
        for fname in os.listdir(cache_dir)
        if fname.endswith(".html")
    }


def write_raw_html(out_path: str, title: str, html: str, source: str) -> None:
    ensure_parent(out_path)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"<!-- {title} - {source} -->\n")
        f.write(html)


def write_markdown(out_path: str, title: str, markdown: str) -> None:
    encoded_title = quote(title.replace(" ", "_"), safe="")
    ensure_parent(out_path)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        f.write(f"Source: https://zh.minecraft.wiki/w/{encoded_title}\n\n")
        f.write(markdown)


def process_cached_html_items(
    html_items: list[tuple[str, str, str]],
    workers: int | None,
    label: str,
) -> tuple[int, int]:
    if not html_items:
        print("无需清洗 Markdown")
        return 0, 0

    try:
        import _md_worker
    except ImportError:  # pragma: no cover - package import fallback
        from test import _md_worker

    n_workers = max(1, workers or os.cpu_count() or 4)
    # 平衡多核通信开销与控制台刷新频率，建议 500-1000 左右
    chunk_size = min(1000, max(1, len(html_items) // n_workers // 4))
    chunks = [html_items[i:i + chunk_size] for i in range(0, len(html_items), chunk_size)]

    print(f"{label}: {n_workers} 进程并行清洗 {len(chunks)} 个批次")
    processed = 0
    error_count = 0
    start_time = time.time()
    ensure_parent(EXPORT_ERROR_LOG)

    with open(EXPORT_ERROR_LOG, "a", encoding="utf-8") as error_log:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            for result in executor.map(_md_worker.process_batch, chunks):
                processed += result["processed"]
                error_count += len(result["errors"])

                for err in result["errors"]:
                    error_log.write(json.dumps(err, ensure_ascii=False) + "\n")
                error_log.flush()

                elapsed = time.time() - start_time
                rate = processed / elapsed if elapsed > 0 else 0
                remaining = (len(html_items) - processed) / rate if rate > 0 else 0
                print(
                    f"\r  [{processed:,}/{len(html_items):,}  "
                    f"{rate:.0f} 篇/秒  {error_count} 错误  "
                    f"剩余约 {remaining / 60:.0f} 分钟]    ",
                    end="",
                    flush=True,
                )

    print()
    return processed, error_count


def cmd_export(workers: int | None = None, zim_path: str = ZIM_PATH) -> None:
    """Export ZIM HTML first (based on cloud metadata), then reuse the shared HTML -> Markdown pipeline."""
    if not HAS_ZIM or not os.path.exists(zim_path):
        print(f"错误: ZIM 文件未找到 ({zim_path})")
        return

    try:
        tmp_archive = open_zim_archive(zim_path)
        local_date_raw = tmp_archive.get_metadata("Date")
        zim_date = local_date_raw.decode("utf-8") if isinstance(local_date_raw, bytes) else (local_date_raw or now_utc())
    except Exception:
        zim_date = now_utc()

    meta = load_metadata()
    meta_pages = meta.get("pages", {})
    if not meta_pages:
        print("\n[错误] 元数据为空！")
        print("本地无法准确识别重定向页面，请先运行 `python test/wiki_manager.py meta` 命令从云端获取准确的词条元数据。")
        return

    os.makedirs(EXPORT_OUTPUT_DIR, exist_ok=True)
    os.makedirs(ZIM_HTML_CACHE_DIR, exist_ok=True)

    titles = list(meta_pages.keys())
    total = len(titles)

    redirects = read_json(EXPORT_REDIRECT_FILE, {})
    done_count = len(existing_markdown_stems())
    html_done_count = len(existing_html_stems(ZIM_HTML_CACHE_DIR))

    n_workers = max(1, workers or os.cpu_count() or 4)
    chunk_size = 1000
    chunks = [titles[i:i + chunk_size] for i in range(0, total, chunk_size)]

    print(f"云端元数据词条总数: {total:,}  已导出 Markdown: {done_count:,}  已缓存 HTML: {html_done_count:,}")
    print(f"阶段 1/2: {n_workers} 进程并行根据元数据导出 ZIM HTML，共 {len(chunks)} 个 chunk，每个约 {chunk_size:,} 条目")

    html_cached = 0
    html_items: list[tuple[str, str, str]] = []
    error_count = 0
    start_time = time.time()
    ensure_parent(EXPORT_ERROR_LOG)

    with open(EXPORT_ERROR_LOG, "a", encoding="utf-8") as error_log:
        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_worker_init,
            initargs=(zim_path,),
        ) as executor:
            for result in executor.map(_worker_cache_html_titles, chunks):
                html_cached += result["html_cached"]
                html_items.extend(result["html_items"])
                error_count += len(result["errors"])
                redirects.update(result["redirects"])

                for err in result["errors"]:
                    error_log.write(json.dumps(err, ensure_ascii=False) + "\n")
                error_log.flush()

                elapsed = time.time() - start_time
                rate = html_cached / elapsed if elapsed > 0 else 0
                print(
                    f"\r  [HTML 新增 {html_cached:,}  待清洗 Markdown {len(html_items):,}  "
                    f"{rate:.0f} 页/秒  {error_count} 错误]    ",
                    end="",
                    flush=True,
                )
    print()
    write_json(EXPORT_REDIRECT_FILE, redirects)

    markdown_processed, markdown_errors = process_cached_html_items(
        html_items,
        workers=workers,
        label="阶段 2/2",
    )
    error_count += markdown_errors
    elapsed = time.time() - start_time

    final_done_stems = existing_markdown_stems()
    updated_meta_count = 0
    for title in titles:
        stem = safe_filename(title)
        if stem in final_done_stems:
            cloud_date = meta_pages[title].get("cloud_date", "")
            if not cloud_date or zim_date >= cloud_date:
                meta_pages[title]["local_date"] = cloud_date
            else:
                meta_pages[title]["local_date"] = zim_date
            updated_meta_count += 1
            
    meta["generated_at"] = now_utc()
    save_metadata(meta)

    print(
        f"\n导出完成。本轮 HTML 新增 {html_cached}，Markdown 新增 {markdown_processed}，"
        f"错误/缺失 {error_count}，Markdown 续跑跳过 {done_count}，"
        f"重定向 {len(redirects)}，耗时 {elapsed / 60:.1f} 分钟"
    )
    print(f"已更新 {updated_meta_count} 个页面的 local_date 基准。")
    if error_count:
        print(f"错误日志: {EXPORT_ERROR_LOG}")


# ---------------------------------------------------------------------------
# Single-page commands
# ---------------------------------------------------------------------------

def cmd_extract(title: str, output_dir: str = "test/output", zim_path: str = ZIM_PATH) -> None:
    os.makedirs(output_dir, exist_ok=True)

    zim_result = fetch_from_zim(title, zim_path=zim_path)
    if zim_result:
        html, page_title = zim_result
        stem = safe_filename(page_title)
        raw_path = os.path.join(output_dir, f"{stem}_zim_raw.html")
        md_path = os.path.join(output_dir, f"{stem}_zim.md")
        json_path = os.path.join(output_dir, f"{stem}_zim.json")

        with open(raw_path, "w", encoding="utf-8") as f:
            f.write(f"<!-- {page_title} - ZIM -->\n{html}")
        markdown = html_to_markdown(clean_html(html))
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(f"# {page_title}\n\n{markdown}")
        write_json(json_path, {
            "title": page_title,
            "source_url": build_page_url(page_title),
            "markdown": markdown,
        })
        print(f"ZIM -> {raw_path}, {md_path}, {json_path}")
    else:
        print(f"ZIM: 未找到 '{title}'")

    api_result = fetch_from_api(title)
    if api_result:
        html, page_title = api_result
        stem = safe_filename(page_title)
        raw_path = os.path.join(output_dir, f"{stem}_api_raw.html")
        md_path = os.path.join(output_dir, f"{stem}_api.md")
        json_path = os.path.join(output_dir, f"{stem}_api.json")

        with open(raw_path, "w", encoding="utf-8") as f:
            f.write(f"<!-- {page_title} - API -->\n{html}")
        markdown = html_to_markdown(clean_html(html))
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(f"# {page_title}\n\n{markdown}")
        write_json(json_path, {
            "title": page_title,
            "source_url": build_page_url(page_title),
            "markdown": markdown,
        })
        print(f"API -> {raw_path}, {md_path}, {json_path}")
    else:
        print(f"API: 未找到 '{title}'")


def cmd_crawl(title: str, output_dir: str = "test/output") -> None:
    os.makedirs(output_dir, exist_ok=True)
    result = extract_page_html(title, source="api")
    if not result:
        print(f"API 获取失败: {title}")
        return

    markdown, page_title, source_url = result
    path = os.path.join(output_dir, f"{safe_filename(page_title)}.json")
    write_json(path, {"title": page_title, "source_url": source_url, "markdown": markdown})
    print(f"已保存: {path}")


def cmd_compare(title: str, zim_path: str = ZIM_PATH) -> None:
    print(f"=== 对比页面: {title} ===\n")
    zim = fetch_from_zim(title, zim_path=zim_path)
    api = fetch_from_api(title)

    zim_md = ""
    api_md = ""

    if zim:
        html, _ = zim
        zim_md = html_to_markdown(clean_html(html))
        print(f"ZIM ({len(zim_md):,} 字符):")
        print(zim_md[:500])
        print("...\n")
    else:
        print("ZIM: 未找到\n")

    if api:
        html, _ = api
        api_md = html_to_markdown(clean_html(html))
        print(f"API ({len(api_md):,} 字符):")
        print(api_md[:500])
        print("...\n")
    else:
        print("API: 未找到\n")

    if zim and api:
        same = zim_md == api_md
        detail = "是" if same else f"否 (ZIM={len(zim_md)} 字, API={len(api_md)} 字)"
        print(f"内容一致: {detail}")


# ---------------------------------------------------------------------------
# Metadata layer
# ---------------------------------------------------------------------------

def load_metadata() -> dict:
    return read_json(METADATA_FILE, {"pages": {}})


def save_metadata(data: dict) -> None:
    if "pages" in data:
        data["pages"] = dict(sorted(data["pages"].items(), key=lambda x: x[0].lower()))
    write_json(METADATA_FILE, data)


def load_cloud_progress() -> dict:
    return read_json(CLOUD_PROGRESS_FILE, {})


def save_cloud_progress(data: dict) -> None:
    write_json(CLOUD_PROGRESS_FILE, data)


def list_all_titles(resume: bool = True) -> list[str]:
    progress = load_cloud_progress() if resume else {}
    titles = progress.get("titles", [])
    continue_params = progress.get("continue", {})

    if progress.get("stage") in ("titles_done", "revisions", "revisions_done") and titles:
        print(f"  标题收集已完成: {len(titles)} 个")
        return titles

    if titles:
        print(f"  从断点继续: 已有 {len(titles)} 个标题")

    consecutive_failures = 0
    while True:
        params = {
            "action": "query",
            "list": "allpages",
            "apnamespace": 0,
            "aplimit": 500,
            "apfilterredir": "nonredirects",
            "format": "json",
        }
        params.update(continue_params)

        try:
            data = api_request(params)
            consecutive_failures = 0
        except Exception as e:
            consecutive_failures += 1
            save_cloud_progress({
                "stage": "titles",
                "titles": titles,
                "continue": continue_params,
            })
            print(f"\n  API 请求失败 ({consecutive_failures}/3): {e}")
            if consecutive_failures >= 3:
                print("  已保存进度，稍后可继续")
                break
            time.sleep(5)
            continue

        pages = data.get("query", {}).get("allpages", [])
        titles.extend(p["title"] for p in pages if "title" in p)

        if "continue" not in data:
            break

        continue_params = data["continue"]
        if len(titles) % 2000 == 0:
            save_cloud_progress({
                "stage": "titles",
                "titles": titles,
                "continue": continue_params,
            })
        print(f"\r  已收集 {len(titles)} 个标题...", end="", flush=True)
        time.sleep(REQUEST_DELAY_SECONDS)

    if titles:
        save_cloud_progress({
            "stage": "titles_done",
            "titles": titles,
            "continue": continue_params,
        })
    print()
    return titles


def _fetch_one_batch(batch: list[str]) -> tuple[list[str], dict[str, str]]:
    data = api_request({
        "action": "query",
        "prop": "revisions",
        "rvprop": "timestamp",
        "format": "json",
        "titles": "|".join(batch),
    })

    if "error" in data:
        raise RuntimeError(f"API Error: {data['error']}")
    if "warnings" in data:
        print(f"\n  [警告] API Warning: {data['warnings']}")

    timestamps: dict[str, str] = {}
    for info in data.get("query", {}).get("pages", {}).values():
        revs = info.get("revisions", [])
        if revs:
            timestamps[info["title"]] = revs[0]["timestamp"]
            
    if not timestamps:
        print(f"\n[DEBUG] API 返回了 0 页! Raw data: {json.dumps(data)[:500]}...")
        
    return batch, timestamps


def fetch_revisions(
    titles: list[str],
    verbose: bool = True,
    resume: bool = True,
    workers: int = 8,
) -> dict[str, str]:
    progress = load_cloud_progress() if resume else {}
    result: dict[str, str] = progress.get("timestamps", {})
    completed_batches = set(progress.get("completed_revision_batches", []))

    if progress.get("stage") == "revisions_done" and result:
        if verbose:
            print(f"  修改时间查询已完成: {len(result)} 页")
        return result

    batches = [
        titles[i:i + REVISION_BATCH_SIZE]
        for i in range(0, len(titles), REVISION_BATCH_SIZE)
    ]
    pending = [
        (idx, batch)
        for idx, batch in enumerate(batches)
        if idx not in completed_batches
    ]

    if verbose and completed_batches:
        print(f"  从断点继续: 已完成 {len(completed_batches)} / {len(batches)} 个 batch")

    errors = 0
    done_this_run = 0
    max_workers = max(1, workers)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_fetch_one_batch, batch): idx
            for idx, batch in pending
        }

        for future in as_completed(futures):
            idx = futures[future]
            try:
                _, timestamps = future.result()
                result.update(timestamps)
                completed_batches.add(idx)
            except Exception as e:
                errors += 1
                if verbose and errors <= 3:
                    print(f"\n  batch {idx} 失败: {e}")

            done_this_run += 1
            if done_this_run % 50 == 0 or done_this_run == len(pending):
                save_cloud_progress({
                    "stage": "revisions",
                    "titles": titles,
                    "timestamps": result,
                    "completed_revision_batches": sorted(completed_batches),
                })
                if verbose:
                    print(
                        f"\r  [{len(completed_batches):,}/{len(batches):,} batches, "
                        f"{len(result):,} 页]",
                        end="",
                        flush=True,
                    )

    if len(completed_batches) == len(batches):
        save_cloud_progress({
            "stage": "revisions_done",
            "titles": titles,
            "timestamps": result,
            "completed_revision_batches": sorted(completed_batches),
        })

    if verbose:
        print(f"\n  完成: {len(result)} 页，{errors} 个 batch 失败")
    return result


def cmd_cloud_update(workers: int = 8) -> None:
    cached = load_metadata()
    cached_pages: dict = cached.get("pages", {})

    print("=== 列出线上所有页面 ===")
    titles = list_all_titles(resume=True)
    print(f"共 {len(titles)} 个主命名空间页面")

    print("\n=== 查询最后修改时间 ===")
    timestamps = fetch_revisions(titles, resume=True, workers=workers)
    print(f"获取成功: {len(timestamps)} 页")

    new_count = 0
    updated_count = 0
    unchanged_count = 0

    for title, cloud_ts in timestamps.items():
        if title not in cached_pages:
            cached_pages[title] = {"local_date": "", "cloud_date": cloud_ts}
            new_count += 1
            continue

        old_cloud = cached_pages[title].get("cloud_date", "")
        if not old_cloud or cloud_ts > old_cloud:
            cached_pages[title]["cloud_date"] = cloud_ts
            updated_count += 1
        else:
            unchanged_count += 1

    save_metadata({
        "generated_at": now_utc(),
        "total_pages": len(cached_pages),
        "pages": cached_pages,
    })

    if os.path.exists(CLOUD_PROGRESS_FILE):
        os.remove(CLOUD_PROGRESS_FILE)

    print(f"\n新增: {new_count}  更新: {updated_count}  未变: {unchanged_count}")
    print(f"已保存到 {METADATA_FILE}")


def pages_needing_update(meta_pages: dict) -> list[str]:
    changed: list[str] = []
    for title, info in meta_pages.items():
        local_date = info.get("local_date", "")
        cloud_date = info.get("cloud_date", "")
        if cloud_date and (not local_date or cloud_date > local_date):
            changed.append(title)
    return changed


def cmd_update(workers: int | None = None) -> None:
    """Fetch changed pages as HTML, then reuse the shared Markdown pipeline."""
    meta = load_metadata()
    meta_pages: dict = meta.get("pages", {})
    if not meta_pages:
        print("\n[错误] 元数据为空！")
        print("本地无法准确识别重定向页面，请先运行 `python test/wiki_manager.py meta` 命令从云端获取准确的词条元数据。")
        return

    changed = pages_needing_update(meta_pages)
    if not changed:
        print("所有页面均为最新，无需更新")
        return

    print(f"需要更新 {len(changed):,} 页")
    os.makedirs(EXPORT_OUTPUT_DIR, exist_ok=True)
    os.makedirs(API_HTML_CACHE_DIR, exist_ok=True)
    ensure_parent(EXPORT_ERROR_LOG)

    html_items: list[tuple[str, str, str]] = []
    fetched = 0
    skipped = 0
    t0 = time.time()

    print(f"阶段 1/2: 从 API 获取 HTML ({len(changed):,} 页)")
    for title in changed:
        stem = safe_filename(title)
        html_path = os.path.join(API_HTML_CACHE_DIR, stem + ".html")
        out_path = os.path.join(EXPORT_OUTPUT_DIR, stem + ".md")

        if os.path.exists(html_path):
            html_items.append((title, html_path, out_path))
            skipped += 1
            continue

        try:
            html_raw = fetch_page_html(title)
        except Exception as e:
            print(f"\n  获取失败: {title}: {e}")
            continue

        if not html_raw:
            continue

        write_raw_html(html_path, title, html_raw, source="API")
        html_items.append((title, html_path, out_path))
        fetched += 1

        if (fetched + skipped) % 100 == 0:
            elapsed = time.time() - t0
            rate = (fetched + skipped) / elapsed if elapsed > 0 else 0
            print(
                f"\r  [{fetched + skipped:,}/{len(changed):,}  "
                f"{rate:.0f} 页/秒  获取: {fetched}  跳过: {skipped}]",
                end="",
                flush=True,
            )
        time.sleep(REQUEST_DELAY_SECONDS)

    print(
        f"\nHTML: 获取 {fetched:,}  跳过 {skipped:,}  "
        f"耗时 {(time.time() - t0) / 60:.1f} 分钟"
    )

    processed, error_count = process_cached_html_items(
        html_items,
        workers=workers,
        label="阶段 2/2",
    )

    for title in changed:
        cloud_date = meta_pages[title].get("cloud_date", "")
        if cloud_date:
            meta_pages[title]["local_date"] = cloud_date

    meta["generated_at"] = now_utc()
    save_metadata(meta)

    print(f"\n完成。总耗时 {(time.time() - t0):.0f} 秒")
    print(f"导出: {processed}  错误: {error_count}")


# ---------------------------------------------------------------------------
# TUI and CLI
# ---------------------------------------------------------------------------

def interactive_tui() -> None:
    from rich.console import Console
    from rich.prompt import Prompt

    console = Console()
    banner = "  [bold]Minecraft Wiki 数据管理工具[/]\n"

    while True:
        console.clear()
        console.print(banner)
        zim_ok = HAS_ZIM and os.path.exists(ZIM_PATH)

        meta = load_metadata() if os.path.exists(METADATA_FILE) else {"pages": {}}
        pages = meta.get("pages", {})
        meta_pages = meta.get("total_pages", len(pages))
        has_local = sum(1 for info in pages.values() if info.get("local_date"))
        has_cloud = sum(1 for info in pages.values() if info.get("cloud_date"))
        pending = len(pages_needing_update(pages))

        console.print(f"  ZIM 文件: [dim]{'已找到' if zim_ok else '未找到'}[/]")
        console.print(
            f"  元数据:   [dim]{meta_pages} 页 | "
            f"local_date: {has_local} | cloud_date: {has_cloud} | 待更新: {pending}[/]"
        )
        console.print()
        console.print("  [[bold]E[/]] 提取单页 (ZIM + API)")
        console.print("  [[bold]C[/]] 云端获取/更新元数据 (查询 cloud_date)")
        console.print("  [[bold]X[/]] 全量导出 ZIM (基于云端元数据)")
        console.print("  [[bold]U[/]] 增量更新数据 (从 API 拉取变动页面)")
        console.print("  [[bold]Q[/]] 退出")
        console.print()

        choice = Prompt.ask("  >", choices=["e", "c", "x", "u", "q"]).lower()

        if choice == "q":
            break
        if choice == "e":
            title = Prompt.ask("  页面标题")
            cmd_extract(title)
        elif choice == "c":
            cmd_cloud_update()
        elif choice == "x":
            cmd_export()
        elif choice == "u":
            cmd_update()

        Prompt.ask("  [dim]按回车继续[/]", default="", show_default=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Minecraft Wiki 数据管理工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
子命令:
  extract <title>     ZIM + API 双源提取，输出 raw HTML / Markdown / JSON
  crawl --title <t>   API 爬取单页，输出结构化 JSON
  meta                从云端 API 增量/全量更新元数据
  compare <title>     对比 ZIM 与 API 输出内容一致性
  export              ZIM 全量导出 Markdown (基于元数据)，支持断点续跑
  update              基于元数据增量导出变更页面
""",
    )
    sub = parser.add_subparsers(dest="command")

    extract_p = sub.add_parser("extract", help="ZIM + API 双源提取单页")
    extract_p.add_argument("title", help="页面标题")

    crawl_p = sub.add_parser("crawl", help="API 爬取单页")
    crawl_p.add_argument("--title", required=True, help="页面标题")
    crawl_p.add_argument("--output-dir", default="test/output", help="输出目录")

    meta_p = sub.add_parser("meta", help="从云端 API 增量/全量更新元数据")
    meta_p.add_argument(
        "-w",
        "--workers",
        type=int,
        default=None,
        help="并发 worker 数，默认根据命令选择",
    )

    cmp_p = sub.add_parser("compare", help="对比 ZIM vs API")
    cmp_p.add_argument("title", help="页面标题")

    export_p = sub.add_parser("export", help="ZIM 全量导出 Markdown")
    export_p.add_argument(
        "-w",
        "--workers",
        type=int,
        default=None,
        help="并行进程数，默认 CPU 核心数",
    )

    update_p = sub.add_parser("update", help="基于元数据增量导出变更页面")
    update_p.add_argument(
        "-w",
        "--workers",
        type=int,
        default=None,
        help="并行进程数，默认 CPU 核心数",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        interactive_tui()
    elif args.command == "extract":
        cmd_extract(args.title)
    elif args.command == "crawl":
        cmd_crawl(args.title, args.output_dir)
    elif args.command == "meta":
        cmd_cloud_update(workers=args.workers or 8)
    elif args.command == "compare":
        cmd_compare(args.title)
    elif args.command == "export":
        cmd_export(workers=args.workers)
    elif args.command == "update":
        cmd_update(workers=args.workers)


if __name__ == "__main__":
    main()
