#!/usr/bin/env python
"""知识库独立管理 CLI —— TUI 界面，无需启动 FastAPI 即可构建/清理/查看状态。

用法:
    python scripts/manage_kb.py                                   交互式 TUI
    python scripts/manage_kb.py status                            查看状态
    python scripts/manage_kb.py build [--model xxx] [--priority low|high]  构建（默认最低优先级）
    python scripts/manage_kb.py clean                             清理
"""

import sys
import os
import argparse
import ctypes
import contextlib
import io
import threading
import logging
import time
import msvcrt

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich.prompt import Confirm
from rich.progress import Progress, BarColumn, TextColumn
from rich import box

from backend.app.core.kb_manager import kb_manager, compute_data_signature
from backend.app.core.config import settings
from backend.app.core.embedding_registry import (
    EMBEDDING_MODEL_PRESETS,
    EMBEDDING_MODEL_MAP,
    DEFAULT_MODEL_ID,
    get_model_persist_dir,
)

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

console = Console()

BANNER = "  [bold]MCwiki 知识库管理[/]\n"

# Windows process priority constants
_PRIORITY_CLASSES = {
    "low": 0x00000040,   # IDLE_PRIORITY_CLASS
    "high": 0x00000080,  # HIGH_PRIORITY_CLASS
}


def set_process_priority(priority: str) -> None:
    """设置当前进程的优先级（仅 Windows），静默忽略失败。"""
    if sys.platform != "win32":
        return
    priority_class = _PRIORITY_CLASSES.get(priority)
    if priority_class is None:
        return
    try:
        kernel32 = ctypes.windll.kernel32
        kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), priority_class)
    except Exception:
        pass


PHASE_LABELS = {
    "load":  "加载知识文件",
    "embed": "向量化并写入",
}


def select_option(options: list[str], prompt: str = "  >") -> int:
    """上下方向键选择选项，Enter 确认，Esc 取消返回 -1。"""
    selected = 0
    n = len(options)

    def _render():
        for i, opt in enumerate(options):
            if i == selected:
                console.print(f"  [reverse]{prompt} {opt}[/reverse]     ")
            else:
                console.print(f"    {opt}     ")

    # 首次渲染
    _render()

    while True:
        ch = msvcrt.getwch()
        # 方向键前缀
        if ch in ('\x00', '\xe0'):
            ch2 = msvcrt.getwch()
            if ch2 == 'H':      # Up
                selected = (selected - 1) % n
            elif ch2 == 'P':    # Down
                selected = (selected + 1) % n
            else:
                continue
        elif ch == '\r':        # Enter
            # 清除选项行
            for _ in range(n):
                sys.stdout.write("\033[A\033[2K")
            sys.stdout.flush()
            return selected
        elif ch == '\x1b':      # Esc
            for _ in range(n):
                sys.stdout.write("\033[A\033[2K")
            sys.stdout.flush()
            return -1
        else:
            continue

        # 重绘
        for _ in range(n):
            sys.stdout.write("\033[A\033[2K")
        sys.stdout.flush()
        _render()


# ── 视图 ──

def make_models_table() -> Table:
    """显示所有模型预设及构建状态"""
    table = Table(
        show_header=True, box=box.ROUNDED, border_style="bright_black",
        show_lines=False, header_style="bold dim white",
    )
    table.add_column("预设 ID", style="dim cyan", width=14)
    table.add_column("显示名", style="white", width=26)
    table.add_column("状态", style="white", width=12)
    table.add_column("构建时间", style="dim white", width=12)
    table.add_column("维度", style="dim white", width=7)
    table.add_column("说明", style="dim white", width=36)

    s = kb_manager.get_status()
    models = s.get("available_models", [])

    for m in models:
        if m["is_built"]:
            status_text = "[bold green]● 已构建[/]"
            built_at = m.get("built_at", "") or "-"
        else:
            status_text = "[dim red]○ 未构建[/]"
            built_at = "-"
        table.add_row(
            m["id"],
            m["name"],
            status_text,
            built_at,
            str(m["dim"]),
            m.get("description", ""),
        )

    return table


def show_summary():
    """显示源数据签名（单行）"""
    sig = compute_data_signature(settings.DATA_DIR)
    sig_short = sig[:24] + "..." if sig else "(无数据)"
    console.print(f"  [dim cyan]源数据签名:[/] [dim white]{sig_short}[/]")


# ── 构建 ──

def _run_rebuild(model_id: str, result: dict, progress_cb=None):
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            stats = kb_manager.rebuild(model_id, progress_cb=progress_cb)
        result["ok"] = True
        result["stats"] = stats
    except Exception as e:
        result["ok"] = False
        result["error"] = str(e)
    finally:
        os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)
    if result["ok"]:
        kb_manager.unload_model()


def cmd_build(model_id: str = None, priority: str = "low"):
    if model_id is None:
        model_id = DEFAULT_MODEL_ID
    preset = EMBEDDING_MODEL_MAP[model_id]

    # 检测是首次建立还是更新
    persist_dir = get_model_persist_dir(settings.PERSIST_DIR, model_id)
    state_path = os.path.join(persist_dir, "build_state.json")
    is_update = os.path.exists(state_path)
    action = "更新" if is_update else "建立"

    console.print(BANNER)
    console.print(f"  操作: [cyan]{action} Embedding 向量库[/]")
    console.print(f"  模型: [yellow]{preset.display_name}[/]")
    console.print(f"  后端: [dim]{preset.backend_type}[/]")
    console.print(f"  维度: [dim]{preset.dim}[/]")
    console.print(f"  优先级: [dim]{priority}[/]\n")

    set_process_priority(priority)

    result: dict = {"ok": False}
    progress_state: dict = {"phase": "init", "current": 0, "total": 0, "msg": ""}

    def on_progress(phase: str, current: int, total: int, msg: str = "", final_msg: str = ""):
        d = {"phase": phase, "current": current, "total": total}
        if msg:
            d["msg"] = msg
        if final_msg:
            d["final_msg"] = final_msg
        progress_state.update(d)

    thread = threading.Thread(target=_run_rebuild, args=(model_id, result, on_progress))

    def _fmt_eta(seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.0f}s"
        m, s = divmod(int(seconds), 60)
        if m < 60:
            return f"{m}m{s:02d}s"
        h, m = divmod(m, 60)
        return f"{h}h{m:02d}m"

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("[dim]{task.fields[info]}[/]"),
        console=console,
        transient=True,
    ) as progress_bar:
        task = progress_bar.add_task("[cyan]加载模型...[/]", total=1, completed=0, info="")
        thread.start()

        prev_phase = None
        embed_start_time = None
        embed_start_offset = 0
        last_speed = 0.0

        while thread.is_alive():
            ps = dict(progress_state)
            phase = ps["phase"]
            current = ps["current"]
            total = ps["total"] if ps["total"] > 0 else 1

            # 进入 embed 阶段时记录起始时间
            if phase == "embed" and embed_start_time is None:
                embed_start_time = time.time()
                embed_start_offset = current

            info = ""
            if phase == "embed" and embed_start_time is not None:
                elapsed = time.time() - embed_start_time
                done = current - embed_start_offset
                if elapsed > 0 and done > 0:
                    speed = done / elapsed
                    remaining = total - current
                    eta = remaining / speed if speed > 0 else 0
                    last_speed = speed
                    info = f"{speed:.1f} chunks/s  ETA {_fmt_eta(eta)}"
                elif last_speed > 0:
                    info = f"{last_speed:.1f} chunks/s"
                # 附加最终调整结果
                final_msg = ps.get("final_msg", "")
                if final_msg:
                    info = f"{final_msg} | {info}" if info else final_msg
            else:
                msg = ps.get("msg", "")
                if msg:
                    info = msg

            if total > 1:
                label = f"[cyan]{PHASE_LABELS.get(phase, phase)}[/] [dim]{current}/{total}[/]"
            else:
                label = f"[cyan]{PHASE_LABELS.get(phase, phase)}[/]"

            progress_bar.update(task, description=label, total=total, completed=current, info=info)
            prev_phase = phase
            time.sleep(0.1)

        thread.join()

        ps = progress_state
        total = ps["total"] if ps["total"] > 0 else 1
        progress_bar.update(task, description="[green]✔ 完成[/]", completed=total, total=total)

    if result["ok"]:
        stats = result.get("stats", {})
        console.print(f"  [bold green]✔ Embedding 向量库{action}完成[/]\n")

        # 扫描结果
        console.print(f"  [dim]扫描文件数:[/] {stats.get('files_scanned', '?')}")

        # 变更统计（更新模式下显示）
        if action == "更新":
            console.print(f"\n  [cyan]变更统计:[/]")
            chunks_added = stats.get('chunks_added', 0)
            chunks_removed = stats.get('chunks_removed', 0)
            if chunks_added > 0:
                console.print(f"    [green]+ {chunks_added} 个新 chunks[/]")
            if chunks_removed > 0:
                console.print(f"    [red]- {chunks_removed} 个已删除 chunks[/]")

        # 最终结果
        console.print(f"\n  [cyan]最终结果:[/]")
        console.print(f"    chunks 数: {stats.get('new_chunks', '?')}")
        console.print()
        show_summary()
    else:
        console.print(f"\n[bold red]✘ {action}失败:[/] {result.get('error', '未知错误')}")


def cmd_rebuild_bm25():
    """构建/更新 BM25 索引（独立于 Embedding，直接从文件读取）"""
    console.print(BANNER)
    console.print("  [cyan]正在构建/更新 BM25 索引...[/]\n")

    result: dict = {"ok": False, "stats": None}
    progress_state: dict = {"phase": "init", "current": 0, "total": 0, "msg": ""}

    def on_progress(phase: str, current: int, total: int, msg: str = "", final_msg: str = ""):
        d = {"phase": phase, "current": current, "total": total}
        if msg:
            d["msg"] = msg
        if final_msg:
            d["final_msg"] = final_msg
        progress_state.update(d)

    def _run():
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                stats = kb_manager.rebuild_bm25(progress_cb=on_progress)
            result["ok"] = True
            result["stats"] = stats
        except Exception as e:
            result["ok"] = False
            result["error"] = str(e)

    thread = threading.Thread(target=_run)

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("[dim]{task.fields[info]}[/]"),
        console=console,
        transient=True,
    ) as progress_bar:
        task = progress_bar.add_task("[cyan]初始化...[/]", total=1, completed=0, info="")
        thread.start()

        while thread.is_alive():
            ps = dict(progress_state)
            phase = ps["phase"]
            current = ps["current"]
            total = ps["total"] if ps["total"] > 0 else 1
            msg = ps.get("msg", "")

            phase_labels = {
                "init": "初始化",
                "index": "索引文档",
                "bm25": "构建 BM25",
                "done": "完成",
            }
            label = f"[cyan]{phase_labels.get(phase, phase)}[/]"
            if phase == "index" and total > 1:
                label += f" [dim]{current}/{total}[/]"

            progress_bar.update(task, description=label, total=total, completed=current, info=msg)
            time.sleep(0.1)

        thread.join()

    if result["ok"]:
        stats = result["stats"]
        action = stats.get("action", "构建")
        console.print(f"  [bold green]✔ BM25 索引{action}完成[/]\n")

        # 扫描结果
        console.print(f"  [dim]扫描文件数:[/] {stats['files_scanned']}")

        # 变更统计（更新模式下显示）
        if action == "更新":
            console.print(f"\n  [cyan]变更统计:[/]")
            if stats['pages_added'] > 0:
                console.print(f"    [green]+ {stats['pages_added']} 个新页面[/]")
            if stats['pages_removed'] > 0:
                console.print(f"    [red]- {stats['pages_removed']} 个已删除页面[/]")
            if stats['docs_added'] > 0:
                console.print(f"    [green]+ {stats['docs_added']} 个新文档[/]")
            if stats['docs_removed'] > 0:
                console.print(f"    [red]- {stats['docs_removed']} 个已删除文档[/]")

        # 最终结果
        console.print(f"\n  [cyan]最终结果:[/]")
        console.print(f"    页面数: {stats['pages']}")
        console.print(f"    n-gram 数: {stats['ngrams']}")
        console.print(f"    BM25 文档数: {stats['bm25_docs']}")
    else:
        console.print(f"\n[bold red]✘ 操作失败:[/] {result.get('error', '未知错误')}")


# ── 子命令 ──

def cmd_status():
    console.print(BANNER)
    console.print(make_models_table())
    console.print()
    show_summary()
    console.print()


def cmd_clean(model_id: str = None):
    console.print(BANNER)
    target = model_id or DEFAULT_MODEL_ID
    preset = EMBEDDING_MODEL_MAP[target]
    console.print(f"  [yellow]正在清理向量库: {preset.display_name}...[/]")
    kb_manager.clean(target)
    console.print(f"  [bold green]✔ 向量库已清理 ({preset.persist_subdir})[/]\n")


# ── 交互式 TUI ──

def _pick_model() -> str:
    """交互式选择模型预设（方向键 + 回车）"""
    console.print()
    console.print(make_models_table())
    console.print()

    presets = list(EMBEDDING_MODEL_PRESETS)
    default_id = kb_manager.get_status().get("active_model_id", DEFAULT_MODEL_ID)
    # 把当前使用的模型排到第一项
    presets.sort(key=lambda p: (0 if p.id == default_id else 1))
    labels = [f"{p.id} — {p.display_name}" for p in presets]

    idx = select_option(labels, prompt="选择模型")
    if idx < 0:
        return default_id
    return presets[idx].id


def interactive_tui():
    MENU = [
        ("构建/更新 Embedding", "b"),
        ("构建/更新 BM25", "m"),
        ("清理", "c"),
        ("卸载 (释放显存)", "u"),
        ("刷新", "r"),
        ("退出", "q"),
    ]

    while True:
        console.clear()
        console.print(BANNER)
        console.print(make_models_table())
        console.print()
        show_summary()
        console.print()

        idx = select_option([label for label, _ in MENU], prompt=">")
        action = MENU[idx][1] if idx >= 0 else "r"

        if action == "q":
            console.print("  [dim]再见~[/]")
            break
        elif action == "b":
            model_id = _pick_model()
            console.print()
            priority_idx = select_option(["low — 低优先级（后台运行）", "high — 高优先级"], prompt="优先级")
            priority = "high" if priority_idx == 1 else "low"
            cmd_build(model_id, priority=priority)
            console.print("\n  [dim]按任意键继续...[/]")
            msvcrt.getwch()
        elif action == "m":
            cmd_rebuild_bm25()
            console.print("\n  [dim]按任意键继续...[/]")
            msvcrt.getwch()
        elif action == "c":
            model_id = _pick_model()
            console.print()
            confirm_idx = select_option(["取消", f"确认清理 ({model_id})"], prompt="选择")
            if confirm_idx == 1:
                cmd_clean(model_id)
            console.print("\n  [dim]按任意键继续...[/]")
            msvcrt.getwch()
        elif action == "u":
            console.print()
            confirm_idx = select_option(["取消", "确认卸载并释放显存"], prompt="选择")
            if confirm_idx == 1:
                kb_manager.unload_model()
                console.print("  [bold green]✔ 模型已卸载，显存已释放[/]\n")
            console.print("\n  [dim]按任意键继续...[/]")
            msvcrt.getwch()
        # "r" → loop again


# ── 入口 ──

def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="MCwiki 知识库管理工具")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("status", help="查看知识库状态")
    build_parser = sub.add_parser("build", help="构建/更新 Embedding 向量库")
    build_parser.add_argument(
        "--model-id", type=str, default=None,
        help=f"Embedding 模型预设 ID (可选: {', '.join(p.id for p in EMBEDDING_MODEL_PRESETS)})",
    )
    build_parser.add_argument(
        "--priority", type=str, choices=("low", "high"), default="low",
        help="进程优先级: low=最低（默认，不影响日常使用）, high=最高（加快构建）",
    )
    sub.add_parser("bm25", help="构建/更新 BM25 索引")
    sub.add_parser("clean", help="清理向量库")
    sub.add_parser("sync", help="同步向量库（等价于 build）")

    args = parser.parse_args()

    if args.command is None:
        interactive_tui()
    elif args.command == "status":
        cmd_status()
    elif args.command in ("build", "sync"):
        cmd_build(model_id=getattr(args, "model_id", None), priority=getattr(args, "priority", "low"))
    elif args.command == "bm25":
        cmd_rebuild_bm25()
    elif args.command == "clean":
        cmd_clean()


if __name__ == "__main__":
    main()
