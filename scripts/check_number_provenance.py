#!/usr/bin/env python3
"""
check_number_provenance.py -- enforce the consistency contract across all four
manuscripts.

The rule (revision plan section 03): every number in every manuscript comes
from a generator reading a named artefact, and no number is ever typed into a
.tex file by hand.

This script finds the violations. It classifies every .tex file in the four
manuscripts as GENERATED (written by a known generator) or HAND, then reports
numeric literals that appear:

  * inside a tabular/table environment in a HAND file  -- hard violation, these
    are the leaderboard numbers a referee will check against the artefacts;
  * inside prose in any file                           -- soft, reported
    separately, since some numbers (year, page counts, hyperparameters) are
    legitimately prose.

A literal is "numeric" if it looks like a measurement: a decimal, or an integer
with a unit or percent attached. Bare small integers (column counts, citation
numbers, \\multicolumn{4}) are ignored -- they are structure, not data.

Usage:
    python scripts/check_number_provenance.py
    python scripts/check_number_provenance.py --strict     # exit 1 on violations
    python scripts/check_number_provenance.py --papers paper paper3
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Which manuscripts, and which of their table files a generator owns.
#
# Sources:
#   scripts/fill_tables.py            -> Paper A T1-T7, Paper D T2-T5,T7-T9
#   scripts/yolo/aggregate_results.py -> Y1, Y2, Y3, Y4
#   scripts/yolo/make_journal_tables.py -> Y1, Y2, Y3, Y5, Y6
# ---------------------------------------------------------------------------
PAPERS: Dict[str, Path] = {
    "A": ROOT / "paper",
    "B": ROOT / "paper2",
    "C": ROOT / "paper3",
    "D": ROOT.parent / "Paper2" / "paper",
}

GENERATED: Dict[str, set] = {
    "A": {
        "T0_archs.tex",
        "T1_main_dice.tex", "T2_boundary.tex", "T3_edes.tex", "T4_ef.tex",
        "T5_quality.tex", "T6_efficiency.tex", "T7_wilcoxon.tex",
        "T8_iou.tex",
    },
    "B": {"Y1_main.tex", "Y2_ablation.tex", "Y3_ceiling.tex", "Y4_stats.tex"},
    "C": {
        "Y1_main.tex", "Y2_ablation.tex", "Y3_ceiling.tex", "Y4_stats.tex",
        "Y5_latency.tex", "Y6_ef.tex", "Y7_seedfloor.tex",
    },
    "D": {
        "T1_architectures.tex",
        "T2_main_leaderboard.tex", "T3_perclass.tex", "T4_edes.tex",
        "T5_variants.tex", "T6_failures.tex", "T7_param_matched.tex",
        "T8_efficiency.tex", "T9_wilcoxon.tex", "T10_shared_mem.tex",
    },
}

GENERATOR_OF = {
    "A": "scripts/fill_tables.py --paper1_tables",
    "B": "scripts/yolo/aggregate_results.py",
    "C": "scripts/yolo/make_journal_tables.py",
    "D": "scripts/fill_tables.py --paper2_tables",
}

# A measurement-looking literal: a decimal number, or an integer immediately
# followed by a unit / percent. Deliberately does NOT match bare integers.
NUM = re.compile(
    r"(?<![\w.])"
    r"(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?"      # 1,234  101,376
    r"|\d+\.\d+"                             # 0.9151  3.736
    r"|\d+\s*(?:\\,)?\s*(?:%|\\%|mm|ms|GB|MB|KB|M\b|B\b|px|FPS))"
    r"(?![\w.])"
)

# Structural contexts where a number is layout, not data.
STRUCTURAL = re.compile(
    r"\\(?:multicolumn|multirow|cmidrule|cline|setlength|tabcolsep|arraystretch"
    r"|includegraphics|label|ref|cite[a-z]*|begin|end|hspace|vspace|columnwidth"
    r"|textwidth|linewidth|scalebox|resizebox|renewcommand|newcommand)\b"
)

TABLE_ENV = re.compile(
    r"\\begin\{(tabular\*?|tabularx|table\*?|longtable|threeparttable)\}"
    r".*?\\end\{\1\}",
    re.S,
)


def strip_comments(text: str) -> str:
    """Remove TeX line comments, keeping line count stable."""
    return "\n".join(re.sub(r"(?<!\\)%.*$", "", ln) for ln in text.splitlines())


def line_of(text: str, pos: int) -> int:
    return text.count("\n", 0, pos) + 1


def scan_file(path: Path) -> Tuple[List[Tuple[int, str, str]], List[Tuple[int, str, str]]]:
    """Return (in_table, in_prose) hits as (lineno, literal, context)."""
    raw = path.read_text(encoding="utf-8", errors="replace")
    text = strip_comments(raw)

    table_spans = [m.span() for m in TABLE_ENV.finditer(text)]

    def in_table(pos: int) -> bool:
        return any(a <= pos < b for a, b in table_spans)

    in_tab: List[Tuple[int, str, str]] = []
    in_pro: List[Tuple[int, str, str]] = []

    for m in NUM.finditer(text):
        ls = text.rfind("\n", 0, m.start()) + 1
        le = text.find("\n", m.end())
        line = text[ls: le if le != -1 else len(text)].strip()
        if STRUCTURAL.search(line):
            continue
        hit = (line_of(text, m.start()), m.group(0), line[:120])
        (in_tab if in_table(m.start()) else in_pro).append(hit)

    return in_tab, in_pro


def tex_files(base: Path) -> List[Path]:
    out: List[Path] = []
    for sub in ("tables", "sections"):
        d = base / sub
        if d.is_dir():
            out += sorted(d.glob("*.tex"))
    if (base / "main.tex").exists():
        out.append(base / "main.tex")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--papers", nargs="*", default=list(PAPERS),
                    help="Subset of A B C D to check (default: all).")
    ap.add_argument("--strict", action="store_true",
                    help="Exit non-zero if any hard violation is found.")
    ap.add_argument("--show-prose", action="store_true",
                    help="Also list every numeric literal found in prose.")
    args = ap.parse_args()

    hard_total = 0
    prose_total = 0

    for key in args.papers:
        base = PAPERS.get(key)
        if base is None or not base.is_dir():
            print(f"[{key}] SKIP -- {base} not found")
            continue

        print(f"\n{'=' * 78}\nPaper {key}  ({base})\n"
              f"  generator: {GENERATOR_OF[key]}\n{'=' * 78}")

        gen = GENERATED[key]
        files = tex_files(base)
        tables = [p for p in files if p.parent.name == "tables"]

        covered = {p.name for p in tables if p.name in gen}
        uncovered = [p for p in tables if p.name not in gen]

        print(f"  tables: {len(tables)} total, {len(covered)} generated, "
              f"{len(uncovered)} hand-maintained")
        for p in uncovered:
            print(f"    HAND  {p.name}")

        # Hard violations: numeric literals inside a table env of a HAND file.
        for p in uncovered:
            hits, _ = scan_file(p)
            if not hits:
                continue
            hard_total += len(hits)
            print(f"\n  !! {p.relative_to(base)}: {len(hits)} hand-typed "
                  f"literals inside a table environment")
            for ln, lit, ctx in hits[:12]:
                print(f"       L{ln:<4} {lit:>10}   {ctx}")
            if len(hits) > 12:
                print(f"       ... and {len(hits) - 12} more")

        # Any table environment living in sections/ or main.tex is also HAND.
        for p in files:
            if p.parent.name == "tables":
                continue
            hits, prose = scan_file(p)
            if hits:
                hard_total += len(hits)
                print(f"\n  !! {p.relative_to(base)}: {len(hits)} literals in an "
                      f"inline table environment (should live in tables/)")
                for ln, lit, ctx in hits[:8]:
                    print(f"       L{ln:<4} {lit:>10}   {ctx}")
            prose_total += len(prose)
            if args.show_prose and prose:
                print(f"\n  ~  {p.relative_to(base)}: {len(prose)} prose literals")
                for ln, lit, ctx in prose:
                    print(f"       L{ln:<4} {lit:>10}   {ctx}")

    print(f"\n{'=' * 78}")
    print(f"HARD violations (hand-typed numbers in table environments): {hard_total}")
    print(f"Prose literals (review manually, or lint with fill_tables --check_prose): "
          f"{prose_total}")
    print("=" * 78)

    if args.strict and hard_total:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
