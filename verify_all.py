"""一键验证脚本：把所有课程和实战项目跑一遍，确认一切正常。

用法：
    python verify_all.py

它会做什么：
    1. 用「离线模拟模型」依次运行 lessons/ 下的每一课和 capstone/assistant.py
    2. 记录每项的退出码和耗时
    3. 最后打印一张汇总表，有任何一项失败就以非零退出码结束

为什么要写这个？
    学习过程中你可能会改代码。改完之后跑一下这个脚本，
    就能立刻知道有没有把哪一课改坏 —— 这就是"回归测试"的思想。
    （真实项目里通常用 pytest，这里为了零依赖，用标准库 subprocess 手写了一个。）
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# 要运行的脚本清单：(显示名, 相对路径)
TARGETS: list[tuple[str, str]] = [
    ("第 01 课 · 基础聊天机器人", "lessons/01_basic_chatbot.py"),
    ("第 02 课 · 工具调用", "lessons/02_tools.py"),
    ("第 03 课 · 记忆持久化", "lessons/03_memory.py"),
    ("第 04 课 · 人机协同", "lessons/04_human_in_the_loop.py"),
    ("第 05 课 · 自定义状态", "lessons/05_custom_state.py"),
    ("第 06 课 · 流式输出", "lessons/06_streaming.py"),
    ("第 07 课 · 控制流", "lessons/07_control_flow.py"),
    ("第 08 课 · 子图与时间旅行", "lessons/08_subgraph.py"),
    ("第 09 课 · 多智能体", "lessons/09_multi_agent.py"),
    ("第 10 课 · 手写 Agent", "lessons/10_handmade_agent.py"),
    ("实战项目 · 命令行智能助理", "capstone/assistant.py"),
]

# 关键产出物检查：跑完之后这些文件/内容应该存在
ARTIFACTS = [
    ("data/memory_demo.sqlite", "第 03 课生成的记忆数据库"),
    ("data/assistant.sqlite", "实战项目生成的记忆数据库"),
]


def build_env() -> dict:
    """构造子进程环境变量。

    OFFLINE_STREAM_DELAY=0  关掉离线模型的"打字延迟"，否则会跑得很慢
    NO_COLOR=1              关掉颜色，避免输出里全是转义码
    PYTHONIOENCODING=utf-8  防止 Windows 控制台编码报错
    """
    env = os.environ.copy()
    env["OFFLINE_STREAM_DELAY"] = "0"
    env["NO_COLOR"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return env


def run_one(name: str, rel_path: str, env: dict) -> tuple[bool, float, str]:
    """运行一个脚本，返回 (是否成功, 耗时秒数, 失败时的错误摘要)。"""
    script = ROOT / rel_path
    if not script.exists():
        return False, 0.0, "文件不存在：%s" % rel_path

    start = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, "-X", "utf8", str(script)],
            cwd=str(ROOT),
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=300,
        )
    except subprocess.TimeoutExpired:
        return False, time.time() - start, "运行超时（超过 300 秒）"

    elapsed = time.time() - start
    if proc.returncode == 0:
        return True, elapsed, ""

    # 失败时把 traceback 的最后几行抓出来，方便定位
    output = (proc.stderr or "") + (proc.stdout or "")
    tail = [line for line in output.strip().splitlines() if line.strip()][-6:]
    return False, elapsed, "\n".join(tail)


def check_artifacts() -> list[tuple[str, bool, str]]:
    results = []
    for rel_path, desc in ARTIFACTS:
        path = ROOT / rel_path
        ok = path.exists() and path.stat().st_size > 0
        size = "%d 字节" % path.stat().st_size if path.exists() else "不存在"
        results.append((desc, ok, size))
    return results


def main() -> int:
    print("=" * 78)
    print("  LangGraph 学习项目 · 全量验证")
    print("=" * 78)
    print()
    print("  解释器：%s" % sys.executable)
    print("  模型：离线模拟模型（不需要 API Key）")
    print()

    env = build_env()
    results: list[tuple[str, bool, float, str]] = []

    for name, rel_path in TARGETS:
        print("  ▶ 正在运行 %s …" % name, end="", flush=True)
        ok, elapsed, error = run_one(name, rel_path, env)
        results.append((name, ok, elapsed, error))
        print("\r  %s %-40s %6.2fs" % ("✔" if ok else "✘", name, elapsed))
        if not ok:
            print("      ┌─ 失败原因 ─────────────────────────────────────────")
            for line in error.splitlines():
                print("      │ %s" % line[:100])
            print("      └────────────────────────────────────────────────────")

    # ---- 产出物检查 ----
    print()
    print("  检查产出文件：")
    artifact_results = check_artifacts()
    for desc, ok, detail in artifact_results:
        print("    %s %-32s %s" % ("✔" if ok else "✘", desc, detail))

    # ---- 汇总 ----
    passed = sum(1 for _, ok, _, _ in results if ok)
    total = len(results)
    total_time = sum(e for _, _, e, _ in results)

    print()
    print("=" * 78)
    if passed == total and all(ok for _, ok, _ in artifact_results):
        print("  全部通过！%d/%d 个脚本正常运行，耗时 %.1f 秒。" % (passed, total, total_time))
        print()
        print("  下一步建议：")
        print("    1. 想真正对话？运行  python lessons/01_basic_chatbot.py --interactive")
        print("    2. 想看完整项目？运行 python capstone/assistant.py --interactive")
        print("    3. 想接真实大模型？把 .env.example 复制成 .env 并填上 API Key")
        print("    4. 想深入原理？读 docs/ 目录下的文档")
        return 0

    print("  有 %d/%d 个脚本失败了，详见上面的失败原因。" % (total - passed, total))
    return 1


if __name__ == "__main__":
    sys.exit(main())
