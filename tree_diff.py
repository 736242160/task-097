#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tree_diff.py — 逐节点比较两棵嵌套结构树，输出差异清单与错误清单。

仅使用 Python 标准库，单文件。

层级表示方式（自定义，理由见下）
--------------------------------
采用「缩进文本」格式，一行一个节点，子节点用空格缩进表示::

    根节点
      子节点A: 值1
      子节点B
        孙节点: 值2

- 行格式: ``名字`` 或 ``名字: 值``（第一个冒号分隔名与值，值可含冒号）。
- 缩进只能使用空格；同级节点缩进必须一致；回退缩进必须落在某个祖先层级上。
- 空行与 ``#`` 开头的行被忽略。
- 允许有多个顶层节点（虚拟根）。

选择理由: 缩进格式对人类最直观、手写不易错、与 YAML/Python 习惯一致；
解析无需引入括号配对状态机，且「缩进不匹配」这类结构错误可以被精确定位到行。

对齐算法（自定义，理由见下）
----------------------------
按「节点名字」在每一层做对齐，与书写顺序无关:

1. 同一父节点下，名字相同的节点两两配对（同名出现多次时按出现次序配对）。
2. 只在甲出现的名字 => 删除；只在乙出现的名字 => 新增。
3. 配对的节点递归比较: 值不同 => 修改；一方有子节点另一方没有 => 子结构不同。

理由: 结构树中节点的身份由「路径（名字序列）」决定而非书写顺序，
按名对齐可避免因节点重排产生大量伪差异；按出现次序配对重名节点，
则保证重复路径也能被稳定对齐并单独报告。

用法
----
    python3 tree_diff.py 结构甲.txt 结构乙.txt
    python3 tree_diff.py 甲.txt - < 乙.txt      # 用 - 表示标准输入

退出码: 0 = 一致且无错误; 1 = 存在差异; 2 = 存在结构/重复错误。
"""

import argparse
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ---------------------------------------------------------------- 数据模型

@dataclass
class Node:
    name: str
    value: Optional[str] = None
    line: int = 0
    children: List["Node"] = field(default_factory=list)


@dataclass
class Issue:
    """错误清单中的一项（结构错误或重复路径）。"""
    source: str   # 来自哪份结构（文件名/标签）
    line: int     # 1 起始行号，0 表示不适用
    message: str

    def __str__(self) -> str:
        where = f"第{self.line}行" if self.line else "（整体）"
        return f"[{self.source}] {where}: {self.message}"


@dataclass
class Diff:
    """差异清单中的一项。"""
    kind: str                 # added / removed / modified / children_mismatch
    path: str                 # 节点路径，如 /根/子节点A
    detail: str = ""

    def __str__(self) -> str:
        label = {
            "added": "新增",
            "removed": "删除",
            "modified": "修改",
            "children_mismatch": "子结构不同",
        }[self.kind]
        return f"[{label}] {self.path}" + (f" — {self.detail}" if self.detail else "")


# ---------------------------------------------------------------- 解析

def parse(text: str, source: str) -> Tuple[Node, List[Issue]]:
    """把缩进文本解析为树。出错时尽量恢复，保证比较能继续。"""
    issues: List[Issue] = []
    root = Node(name="", line=0)
    # 栈元素: (缩进宽度, 节点)。虚拟根缩进为 -1，永不出栈。
    stack: List[Tuple[int, Node]] = [(-1, root)]

    for lineno, raw in enumerate(text.splitlines(), 1):
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue

        indent_text = raw[: len(raw) - len(raw.lstrip(" \t"))]
        if "\t" in indent_text:
            issues.append(Issue(source, lineno,
                                "缩进中混入了制表符（Tab），请只使用空格"))
            indent_text = indent_text.replace("\t", "    ")
        indent = len(indent_text)

        name, sep, value = stripped.partition(":")
        name = name.strip()
        node_value: Optional[str] = value.strip() if sep else None
        if not name:
            issues.append(Issue(source, lineno, "节点名为空，该行被忽略"))
            continue

        node = Node(name=name, value=node_value, line=lineno)

        popped = False
        level_matched = False
        while indent <= stack[-1][0]:
            if stack[-1][0] == indent:
                level_matched = True
            stack.pop()
            popped = True
        if popped and not level_matched:
            # 回退后没有落在任何已有层级上：缩进不匹配
            issues.append(Issue(
                source, lineno,
                f"缩进回退到未出现过的层级（缩进 {indent} 格，"
                f"最近的祖先层级为 {stack[-1][0]} 格），已按该祖先的子节点恢复"))

        stack[-1][1].children.append(node)
        stack.append((indent, node))

    return root, issues


# ------------------------------------------------------- 重复路径检测

def check_duplicates(node: Node, path: str, source: str,
                     issues: List[Issue]) -> None:
    """同一父节点下出现同名节点 => 路径重复，逐份报告。"""
    seen = {}
    for child in node.children:
        child_path = f"{path}/{child.name}"
        if child.name in seen:
            first = seen[child.name]
            issues.append(Issue(
                source, child.line,
                f"路径重复: {child_path}（首次出现于第{first.line}行）"))
        else:
            seen[child.name] = child
        check_duplicates(child, child_path, source, issues)


# ---------------------------------------------------------------- 对齐比较

def _count_descendants(node: Node) -> int:
    return sum(1 + _count_descendants(c) for c in node.children)


def _mark_subtree(node: Node, path: str, kind: str,
                  diffs: List[Diff], side_word: str) -> None:
    """把整棵子树标记为新增或删除。"""
    n = _count_descendants(node)
    extra = f"（仅存在于{side_word}，含 {n} 个后代节点）" if n else f"（仅存在于{side_word}）"
    diffs.append(Diff(kind, path, extra))


def diff_nodes(a: Node, b: Node, path: str, diffs: List[Diff]) -> None:
    """比较已对齐的一对节点 a（甲）与 b（乙）。"""
    # 1) 值比较
    if a.value != b.value:
        fmt = lambda v: "（无值）" if v is None else repr(v)
        diffs.append(Diff("modified", path,
                          f"甲={fmt(a.value)}，乙={fmt(b.value)}"))

    # 2) 子节点结构比较：一方有子节点另一方没有
    if bool(a.children) != bool(b.children):
        who = "甲有子节点而乙没有" if a.children else "乙有子节点而甲没有"
        diffs.append(Diff("children_mismatch", path, who))
        # 有子节点的一侧，其子节点整体视为删除/新增
        for child in a.children:
            _mark_subtree(child, f"{path}/{child.name}", "removed", diffs, "甲")
        for child in b.children:
            _mark_subtree(child, f"{path}/{child.name}", "added", diffs, "乙")
        return

    # 3) 按名字对齐子节点（与顺序无关；重名按出现次序配对）
    def index_by_name(node: Node):
        table = {}
        for child in node.children:
            table.setdefault(child.name, []).append(child)
        return table

    a_map, b_map = index_by_name(a), index_by_name(b)

    for name, nodes in a_map.items():
        if name not in b_map:
            for child in nodes:
                _mark_subtree(child, f"{path}/{name}", "removed", diffs, "甲")
    for name, nodes in b_map.items():
        if name not in a_map:
            for child in nodes:
                _mark_subtree(child, f"{path}/{name}", "added", diffs, "乙")

    for name in a_map.keys() & b_map.keys():
        a_list, b_list = a_map[name], b_map[name]
        for i in range(min(len(a_list), len(b_list))):
            diff_nodes(a_list[i], b_list[i], f"{path}/{name}", diffs)
        # 重名数量不一致时，多出的按删除/新增处理（重复本身已在错误清单报告）
        for child in a_list[len(b_list):]:
            _mark_subtree(child, f"{path}/{name}", "removed", diffs, "甲")
        for child in b_list[len(a_list):]:
            _mark_subtree(child, f"{path}/{name}", "added", diffs, "乙")


# ---------------------------------------------------------------- 主流程

def compare(text_a: str, text_b: str,
            source_a: str = "结构甲", source_b: str = "结构乙"
            ) -> Tuple[List[Diff], List[Issue]]:
    root_a, err_a = parse(text_a, source_a)
    root_b, err_b = parse(text_b, source_b)

    issues = err_a + err_b
    check_duplicates(root_a, "", source_a, issues)
    check_duplicates(root_b, "", source_b, issues)

    diffs: List[Diff] = []
    diff_nodes(root_a, root_b, "", diffs)
    # 虚拟根路径为空，修正顶层路径显示
    for d in diffs:
        if not d.path:
            d.path = "/"
    return diffs, issues


def render(diffs: List[Diff], issues: List[Issue]) -> str:
    out = []
    out.append("=== 差异清单 ===")
    if diffs:
        out.extend(str(d) for d in diffs)
    else:
        out.append("（无差异，两棵结构树一致）")
    out.append("")
    out.append("=== 错误清单 ===")
    if issues:
        out.extend(str(e) for e in issues)
    else:
        out.append("（无结构错误）")
    return "\n".join(out)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="逐节点比较两棵缩进文本表示的结构树，输出差异与错误清单。")
    parser.add_argument("file_a", help="结构甲文件路径（- 表示标准输入）")
    parser.add_argument("file_b", help="结构乙文件路径（- 表示标准输入）")
    parser.add_argument("--name-a", default="结构甲", help="报告中甲的名称")
    parser.add_argument("--name-b", default="结构乙", help="报告中乙的名称")
    args = parser.parse_args(argv)

    def read(path: str) -> str:
        if path == "-":
            return sys.stdin.read()
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    try:
        text_a = read(args.file_a)
        text_b = read(args.file_b)
    except OSError as exc:
        print(f"读取文件失败: {exc}", file=sys.stderr)
        return 2

    diffs, issues = compare(text_a, text_b, args.name_a, args.name_b)
    print(render(diffs, issues))

    if issues:
        return 2
    return 1 if diffs else 0


if __name__ == "__main__":
    sys.exit(main())
