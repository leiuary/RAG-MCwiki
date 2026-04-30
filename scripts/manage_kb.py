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

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich.prompt import Prompt, Confirm
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
            kb_manager.rebuild(model_id, progress_cb=progress_cb)
        result["ok"] = True
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

    console.print(BANNER)
    console.print(f"  模型: [yellow]{preset.display_name}[/]")
    console.print(f"  后端: [dim]{preset.backend_type}[/]")
    console.print(f"  维度: [dim]{preset.dim}[/]")
    console.print(f"  优先级: [dim]{priority}[/]\n")

    set_process_priority(priority)

    result: dict = {"ok": False}
    progress_state: dict = {"phase": "init", "current": 0, "total": 0}

    def on_progress(phase: str, current: int, total: int):
        progress_state.update(phase=phase, current=current, total=total)

    thread = threading.Thread(target=_run_rebuild, args=(model_id, result, on_progress))

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
        transient=True,
    ) as progress_bar:
        task = progress_bar.add_task("[cyan]加载模型...[/]", total=1, completed=0)
        thread.start()

        prev_phase = None
        while thread.is_alive():
            ps = dict(progress_state)
            phase = ps["phase"]
            current = ps["current"]
            total = ps["total"] if ps["total"] > 0 else 1

            if total > 1:
                label = f"[cyan]{PHASE_LABELS.get(phase, phase)}[/] [dim]{current}/{total}[/]"
            else:
                label = f"[cyan]{PHASE_LABELS.get(phase, phase)}[/]"

            progress_bar.update(task, description=label, total=total, completed=current)
            prev_phase = phase
            time.sleep(0.1)

        thread.join()

        ps = progress_state
        total = ps["total"] if ps["total"] > 0 else 1
        progress_bar.update(task, description="[green]✔ 完成[/]", completed=total, total=total)

    if result["ok"]:
        console.print("  [bold green]✔ 构建完成[/]\n")
        show_summary()
    else:
        console.print(f"\n[bold red]✘ 构建失败:[/] {result.get('error', '未知错误')}")


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
    """交互式选择模型预设"""
    console.print()
    console.print(make_models_table())
    console.print()

    model_ids = [p.id for p in EMBEDDING_MODEL_PRESETS]
    choices = model_ids + [""]
    default_id = kb_manager.get_status().get("active_model_id", DEFAULT_MODEL_ID)

    choice = Prompt.ask(
        "  选择模型 ID",
        choices=choices,
        default=default_id,
        show_choices=False,
    )
    return choice.strip() or default_id


def interactive_tui():
    while True:
        console.clear()
        console.print(BANNER)
        console.print(make_models_table())
        console.print()
        show_summary()
        console.print()
        console.print("  [[bold]B[/]] 构建  [[bold]C[/]] 清理  [[bold]U[/]] 卸载(释放显存)  [[bold]R[/]] 刷新  [[bold]Q[/]] 退出")
        console.print()

        choice = Prompt.ask(
            "  >", choices=["b", "c", "u", "r", "q", ""],
            default="r", show_choices=False,
        ).lower()

        if choice == "q":
            console.print("  [dim]再见~[/]")
            break
        elif choice == "b":
            model_id = _pick_model()
            priority = Prompt.ask(
                "  构建优先级", choices=["low", "high"], default="low", show_choices=True
            )
            cmd_build(model_id, priority=priority)
            Prompt.ask("  [[dim]按回车继续[/]]", default="", show_default=False, show_choices=False)
        elif choice == "c":
            model_id = _pick_model()
            if Confirm.ask(f"  [yellow]确认清理向量库 ({model_id})？[/]", default=False):
                cmd_clean(model_id)
            Prompt.ask("  [[dim]按回车继续[/]]", default="", show_default=False, show_choices=False)
        elif choice == "u":
            if Confirm.ask("  [yellow]确认卸载当前 Embedding 模型并释放显存？[/]", default=True):
                kb_manager.unload_model()
                console.print("  [bold green]✔ 模型已卸载，显存已释放[/]\n")
            Prompt.ask("  [[dim]按回车继续[/]]", default="", show_default=False, show_choices=False)
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
    build_parser = sub.add_parser("build", help="构建/重建向量库")
    build_parser.add_argument(
        "--model-id", type=str, default=None,
        help=f"Embedding 模型预设 ID (可选: {', '.join(p.id for p in EMBEDDING_MODEL_PRESETS)})",
    )
    build_parser.add_argument(
        "--priority", type=str, choices=("low", "high"), default="low",
        help="进程优先级: low=最低（默认，不影响日常使用）, high=最高（加快构建）",
    )
    sub.add_parser("clean", help="清理向量库")
    sub.add_parser("sync", help="同步向量库（等价于 build）")

    args = parser.parse_args()

    if args.command is None:
        interactive_tui()
    elif args.command == "status":
        cmd_status()
    elif args.command in ("build", "sync"):
        cmd_build(model_id=getattr(args, "model_id", None), priority=getattr(args, "priority", "low"))
    elif args.command == "clean":
        cmd_clean()


if __name__ == "__main__":
    main()
