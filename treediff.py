#!/usr/bin/env python3
"""treediff.py — 嵌套结构树逐节点比较工具（纯 Python 标准库，单文件）

== 输入格式：缩进大纲 ==
每行一个节点：

    <缩进><名称>            # 中间节点（或无值叶子）
    <缩进><名称>: <值>      # 带值节点（第一个冒号之后整体为值）

- 层级完全由缩进表达：子节点比父节点恰好多一级缩进；
- 全文缩进单位必须一致（如统一 2 个空格，或统一 1 个制表符），不得混用；
- 空行、以 # 开头的行视为注释，忽略；
- 允许多个顶层节点（森林），顶层之间也按名称对齐比较。

== 用法 ==
    python treediff.py 甲.txt 乙.txt     # 比较两个结构文件
    python treediff.py --demo            # 运行内置示例（覆盖全部差异/错误类型）

== 退出码 ==
    0  无差异且无错误
    1  有差异（无结构错误）
    2  存在结构错误（仍输出已解析部分的差异，供参考）
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass
class Node:
    """结构树节点。value 为 None 表示无值节点；children 为空表示叶子。"""

    name: str
    value: Optional[str]
    line: int  # 源文件行号（1 起），用于错误定位
    children: List["Node"] = field(default_factory=list)


@dataclass
class StructureError:
    """结构错误（缩进不匹配、路径重复等），line 为 1 起行号。"""

    source: str  # 来自哪份结构（甲/乙）
    kind: str  # 错误类型
    line: int
    message: str

    def __str__(self) -> str:
        return f"[{self.source}] {self.kind}（第 {self.line} 行）：{self.message}"


DIFF_LABELS = {
    "ADDED": "新增",
    "REMOVED": "删除",
    "MODIFIED": "修改",
    "CHILDREN_MISMATCH": "子结构不同",
}


@dataclass
class Diff:
    """一条差异。path 为根到节点的路径（如 root/server/host）。"""

    kind: str  # ADDED / REMOVED / MODIFIED / CHILDREN_MISMATCH
    path: str
    left: Optional[str] = None  # 甲侧值（无则 None）
    right: Optional[str] = None  # 乙侧值（无则 None）
    detail: str = ""

    def __str__(self) -> str:
        label = DIFF_LABELS[self.kind]
        if self.kind == "MODIFIED":
            return f"[{label}] {self.path}: 甲={fmt_value(self.left)} 乙={fmt_value(self.right)}"
        if self.kind == "ADDED":
            return f"[{label}] {self.path}（仅存在于乙：{self.right}）"
        if self.kind == "REMOVED":
            return f"[{label}] {self.path}（仅存在于甲：{self.left}）"
        return f"[{label}] {self.path}：{self.detail}"


def fmt_value(value: Optional[str]) -> str:
    return "(无值)" if value is None else value


def node_summary(node: Node) -> str:
    """新增/删除节点的简要描述：名称、值与直接子节点数。"""
    text = node.name if node.value is None else f"{node.name}: {node.value}"
    if node.children:
        text += f"（含 {len(node.children)} 个直接子节点）"
    return text


# ---------------------------------------------------------------------------
# 解析：缩进大纲 -> 节点森林
# ---------------------------------------------------------------------------


def parse(text: str, source: str) -> Tuple[List[Node], List[StructureError]]:
    """解析缩进大纲文本，返回 (根节点列表, 结构错误列表)。

    可恢复的错误（如某行缩进非法）跳过该行继续解析，以便一次性报告尽量多的问题。
    """
    roots: List[Node] = []
    errors: List[StructureError] = []
    stack: List[Node] = []  # 当前路径上的祖先链，stack[i] 的深度为 i
    indent_unit: Optional[int] = None  # 空格局下的缩进单位（由首个缩进行确定）
    indent_style: Optional[str] = None  # "space" 或 "tab"

    for lineno, raw in enumerate(text.splitlines(), 1):
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue

        # --- 计算本行缩进层级 ---
        indent_str = raw[: len(raw) - len(raw.lstrip(" \t"))]
        if " " in indent_str and "\t" in indent_str:
            errors.append(
                StructureError(source, "INDENT_MIXED", lineno, "同一行内混用空格与制表符缩进")
            )
            continue
        if indent_str:
            style = "tab" if indent_str.startswith("\t") else "space"
            if indent_style is None:
                indent_style = style
            elif style != indent_style:
                errors.append(
                    StructureError(
                        source, "INDENT_MIXED", lineno, "与全文缩进风格不一致（空格/制表符混用）"
                    )
                )
                continue
            width = len(indent_str)
            if indent_style == "tab":
                unit = 1
            else:
                if indent_unit is None:
                    indent_unit = width
                unit = indent_unit
            if width % unit != 0:
                errors.append(
                    StructureError(
                        source,
                        "INDENT_INVALID",
                        lineno,
                        f"缩进宽度 {width} 不是缩进单位 {unit} 的整数倍",
                    )
                )
                continue
            level = width // unit
        else:
            level = 0

        # --- 拆分名称与值 ---
        if ":" in stripped:
            name, _, value = stripped.partition(":")
            name, value = name.strip(), value.strip()
        else:
            name, value = stripped, None
        if not name:
            errors.append(StructureError(source, "SYNTAX", lineno, "缺少节点名称"))
            continue

        # --- 缩进跳跃检查：子节点最多比上一层深一级 ---
        if level > len(stack):
            errors.append(
                StructureError(
                    source,
                    "INDENT_JUMP",
                    lineno,
                    f"缩进跳跃：当前路径深度为 {len(stack)}，本行直接跳到第 {level} 级",
                )
            )
            continue

        node = Node(name=name, value=value, line=lineno)
        del stack[level:]  # 弹出同级及更深的节点
        if stack:
            stack[-1].children.append(node)
        else:
            roots.append(node)
        stack.append(node)

    return roots, errors


# ---------------------------------------------------------------------------
# 重复路径检测（单份结构内部）
# ---------------------------------------------------------------------------


def find_duplicates(roots: List[Node], source: str) -> List[StructureError]:
    """同一路径在同一结构中出现多次即报告（重复会使对齐产生歧义）。"""
    errors: List[StructureError] = []

    def walk(nodes: List[Node], path: str) -> None:
        first_seen = {}
        for node in nodes:
            node_path = f"{path}/{node.name}" if path else node.name
            if node.name in first_seen:
                errors.append(
                    StructureError(
                        source,
                        "DUPLICATE",
                        node.line,
                        f"路径重复：'{node_path}' 首次出现于第 {first_seen[node.name].line} 行",
                    )
                )
            else:
                first_seen[node.name] = node
        for node in nodes:
            node_path = f"{path}/{node.name}" if path else node.name
            walk(node.children, node_path)

    walk(roots, "")
    return errors


# ---------------------------------------------------------------------------
# 对齐与比较
#
# 对齐算法：同名稳定配对（stable name-based pairing）。
# 同一父节点下，子节点按名称分组并保持出现顺序，甲的第 k 个同名节点与乙的
# 第 k 个配对；配不上对的即为新增/删除。理由：
#   1. 结构树（配置、目录、文档大纲）中节点身份由"路径=名称序列"天然确定，
#      同名即同一逻辑节点，比对结果语义正确、路径稳定；
#   2. 算法确定性、O(n)，不依赖编辑距离阈值，结果可解释；
#   3. 同名节点数量不一致时按出现顺序配对，多余者自然落入新增/删除，
#      与重复路径报告互补。
# ---------------------------------------------------------------------------


def _group_by_name(nodes: List[Node]) -> Tuple[List[str], dict]:
    order: List[str] = []
    groups = {}
    for node in nodes:
        if node.name not in groups:
            groups[node.name] = []
            order.append(node.name)
        groups[node.name].append(node)
    return order, groups


def compare_nodes(left: Node, right: Node, path: str, diffs: List[Diff]) -> None:
    """比较已配对的两个节点：值、子结构有无、递归比较子节点。"""
    if left.value != right.value:
        diffs.append(Diff("MODIFIED", path, left=left.value, right=right.value))
    if bool(left.children) != bool(right.children):
        side = "甲有子节点而乙没有" if left.children else "乙有子节点而甲没有"
        diffs.append(Diff("CHILDREN_MISMATCH", path, detail=side))
    compare_forests(left.children, right.children, path, diffs)


def compare_forests(
    left_nodes: List[Node], right_nodes: List[Node], path: str, diffs: List[Diff]
) -> None:
    """对同一层级的两个节点序列做同名稳定配对并逐对比较。"""
    left_order, left_groups = _group_by_name(left_nodes)
    right_order, right_groups = _group_by_name(right_nodes)
    names = left_order + [n for n in right_order if n not in left_groups]

    for name in names:
        lefts = left_groups.get(name, [])
        rights = right_groups.get(name, [])
        node_path = f"{path}/{name}" if path else name
        for index in range(max(len(lefts), len(rights))):
            # 重复路径用 #k 区分第几次出现，保证差异路径唯一
            occurrence_path = node_path if index == 0 else f"{node_path}#{index + 1}"
            left = lefts[index] if index < len(lefts) else None
            right = rights[index] if index < len(rights) else None
            if left is not None and right is not None:
                compare_nodes(left, right, occurrence_path, diffs)
            elif left is not None:
                diffs.append(
                    Diff(
                        "REMOVED",
                        occurrence_path,
                        left=node_summary(left),
                        detail=f"甲第 {left.line} 行",
                    )
                )
            else:
                diffs.append(
                    Diff(
                        "ADDED",
                        occurrence_path,
                        right=node_summary(right),
                        detail=f"乙第 {right.line} 行",
                    )
                )


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------


def print_report(diffs: List[Diff], errors: List[StructureError], out) -> None:
    print("=" * 24 + " 差异清单 " + "=" * 24, file=out)
    if diffs:
        for diff in diffs:
            print(diff, file=out)
    else:
        print("（无差异）", file=out)
    print(f"共 {len(diffs)} 处差异", file=out)

    print("=" * 24 + " 错误清单 " + "=" * 24, file=out)
    if errors:
        for error in errors:
            print(error, file=out)
    else:
        print("（无结构错误）", file=out)
    print(f"共 {len(errors)} 个结构错误", file=out)


def diff_texts(text_a: str, text_b: str, out=sys.stdout) -> int:
    """比较两份结构文本，打印报告，返回退出码。"""
    roots_a, errors_a = parse(text_a, "结构甲")
    roots_b, errors_b = parse(text_b, "结构乙")
    errors = errors_a + errors_b
    errors += find_duplicates(roots_a, "结构甲")
    errors += find_duplicates(roots_b, "结构乙")

    diffs: List[Diff] = []
    compare_forests(roots_a, roots_b, "", diffs)

    print_report(diffs, errors, out)
    if errors:
        return 2
    return 1 if diffs else 0


# ---------------------------------------------------------------------------
# 内置示例
# ---------------------------------------------------------------------------

DEMO_A = """\
root
  server
    host: 127.0.0.1
    port: 8080
  cache
    enabled: true
  log
    level: info
"""

DEMO_B = """\
root
  server
    host: 127.0.0.2
    port: 8080
  cache: true
  log
    level: debug
    output: stdout
"""

DEMO_BAD_A = """\
root
  db
    host: a
   port: 5432
  db
    host: b
"""

DEMO_BAD_B = """\
root
  db
    host: a
"""


def run_demo() -> int:
    print(">>> 示例一：新增 / 删除 / 修改 / 子结构不同")
    code = diff_texts(DEMO_A, DEMO_B)
    print()
    print(">>> 示例二：结构错误（缩进不匹配、路径重复）")
    diff_texts(DEMO_BAD_A, DEMO_BAD_B)
    return code


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="嵌套结构树逐节点比较工具（缩进大纲格式，详见文件头注释）"
    )
    parser.add_argument("file_a", nargs="?", help="结构甲文件路径")
    parser.add_argument("file_b", nargs="?", help="结构乙文件路径")
    parser.add_argument("--demo", action="store_true", help="运行内置示例")
    args = parser.parse_args(argv)

    if args.demo:
        return run_demo()
    if not args.file_a or not args.file_b:
        parser.error("需要提供两个结构文件路径，或使用 --demo 运行示例")

    try:
        with open(args.file_a, encoding="utf-8") as f:
            text_a = f.read()
        with open(args.file_b, encoding="utf-8") as f:
            text_b = f.read()
    except OSError as exc:
        print(f"读取文件失败：{exc}", file=sys.stderr)
        return 2
    return diff_texts(text_a, text_b)


if __name__ == "__main__":
    sys.exit(main())
