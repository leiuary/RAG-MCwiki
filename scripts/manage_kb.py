#!/usr/bin/env python
"""知识库独立管理 CLI —— TUI 界面，无需启动 FastAPI 即可构建/清理/查看状态。

用法:
    python scripts/manage_kb.py                        交互式 TUI
    python scripts/manage_kb.py status                 查看状态
    python scripts/manage_kb.py build [--model xxx]    构建
    python scripts/manage_kb.py clean                  清理
"""

import sys
import os
import argparse
import threading
import logging
import time

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.text import Text
from rich.align import Align
from rich.prompt import Prompt, Confirm
from rich import box

from backend.app.core.kb_manager import kb_manager, compute_data_signature
from backend.app.core.config import settings

# 构建期间抑制 logger 输出，避免破坏 TUI 画面
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

console = Console()

BANNER = "  [bold]MCwiki 知识库管理[/]\n"


def make_status_table() -> Table:
    s = kb_manager.get_status()
    table = Table(show_header=False, box=box.ROUNDED, border_style="bright_black")
    table.add_column("attr", style="dim cyan", width=16)
    table.add_column("value", style="white")

    ready = s["ready"]
    status_text = "[bold green]● 已就绪[/]" if ready else "[dim red]○ 未初始化[/]"
    table.add_row("运行状态", status_text)
    table.add_row("Embedding 模型", Text(s["embedding_model"], style="yellow"))
    table.add_row("数据版本", Text(s["version"], style="white"))

    if ready:
        sig = compute_data_signature(settings.DATA_DIR)
        table.add_row("源数据签名", Text(sig[:24] + "...", style="dim white"))

    return table


# ── 子命令模式 ──

def cmd_status():
    console.print(BANNER)
    console.print(make_status_table())
    console.print()


def _run_rebuild(model: str | None, result: dict):
    try:
        kb_manager.rebuild(model)
        result["ok"] = True
    except Exception as e:
        result["ok"] = False
        result["error"] = str(e)


def cmd_build(model: str | None = None):
    console.print(BANNER)
    model_label = model or settings.EMBEDDING_MODEL_NAME
    console.print(f"  Embedding: [yellow]{model_label}[/]\n")

    result: dict = {"ok": False}
    thread = threading.Thread(target=_run_rebuild, args=(model, result))

    with Live(console=console, refresh_per_second=10, transient=True) as live:
        thread.start()
        while thread.is_alive():
            count = int(time.time() * 3) % 4
            dots = "⏳ 正在构建向量库" + "." * count + " " * (3 - count)
            live.update(Align.center(Text(dots, style="bold cyan")))
            time.sleep(0.25)
        thread.join()

    if result["ok"]:
        console.print(Align.center("[bold green]✔ 构建完成[/]\n"))
        console.print(make_status_table())
    else:
        console.print(f"\n[bold red]✘ 构建失败:[/] {result.get('error', '未知错误')}")


def cmd_clean():
    console.print(BANNER)
    console.print("  [yellow]正在清理向量库...[/]")
    kb_manager.clean()
    console.print("  [bold green]✔ 向量库已清理[/]\n")


# ── 交互式 TUI ──

def interactive_tui():
    """无子命令时进入交互式 TUI 循环。"""
    while True:
        console.clear()
        console.print(BANNER)
        console.print(make_status_table())
        console.print()
        console.print("  [[bold]B[/]] 构建  [[bold]C[/]] 清理  [[bold]R[/]] 刷新  [[bold]Q[/]] 退出")
        console.print()

        choice = Prompt.ask("  >", choices=["b", "c", "r", "q", ""], default="r", show_choices=False).lower()

        if choice == "q":
            console.print("  [dim]再见~[/]")
            break
        elif choice == "b":
            console.print()
            model = Prompt.ask(
                "  Embedding 模型",
                default=settings.EMBEDDING_MODEL_NAME,
                show_default=False,
            )
            if not model.strip():
                model = settings.EMBEDDING_MODEL_NAME
            cmd_build(model if model != settings.EMBEDDING_MODEL_NAME else None)
            Prompt.ask("  [[dim]按回车继续[/]]", default="", show_default=False, show_choices=False)
        elif choice == "c":
            if Confirm.ask("  [yellow]确认清理向量库？[/]", default=False):
                cmd_clean()
            Prompt.ask("  [[dim]按回车继续[/]]", default="", show_default=False, show_choices=False)
        # "r" falls through to loop again


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
    build_parser.add_argument("--model", type=str, default=None, help="指定 Embedding 模型名称")
    sub.add_parser("clean", help="清理向量库")
    sub.add_parser("sync", help="同步向量库（等价于 build）")

    args = parser.parse_args()

    if args.command is None:
        interactive_tui()
    elif args.command == "status":
        cmd_status()
    elif args.command in ("build", "sync"):
        cmd_build(model=getattr(args, "model", None))
    elif args.command == "clean":
        cmd_clean()


if __name__ == "__main__":
    main()
