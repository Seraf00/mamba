"""Syntax-check every code cell of the generated notebook.

IPython shell escapes (`!cmd`) and magics (`%cd`) are not Python, so they are
blanked before parsing -- along with any lines they continue onto via a
trailing backslash.
"""
import ast
import json
import sys

BACKSLASH = chr(92)


def strip_magics(src: str) -> str:
    out, skip, held = [], False, ''
    for ln in src.split('\n'):
        if skip:
            # A continuation of a magic: keep the MAGIC's indent, not the
            # continuation line's own (which is deeper and would look like an
            # unexpected indent once the magic became a bare `pass`).
            out.append(held + 'pass')
            skip = ln.rstrip().endswith(BACKSLASH)
            continue
        s = ln.lstrip()
        if s.startswith('!') or s.startswith('%'):
            held = ln[:len(ln) - len(s)]
            out.append(held + 'pass')
            skip = ln.rstrip().endswith(BACKSLASH)
        else:
            out.append(ln)
    return '\n'.join(out)


nb = json.load(open(sys.argv[1], encoding='utf-8'))
bad = 0
for i, c in enumerate(nb['cells']):
    if c['cell_type'] != 'code':
        continue
    cleaned = strip_magics('\n'.join(c['source']))
    try:
        ast.parse(cleaned)
    except SyntaxError as e:
        bad += 1
        print(f'--- cell {i}: {e}')
        for n, ln in enumerate(cleaned.split('\n'), 1):
            mark = '>>' if n == e.lineno else '  '
            print(f'{mark} {n:3d} {ln}')
print(f'code cells checked: {sum(c["cell_type"] == "code" for c in nb["cells"])}')
print(f'syntax errors: {bad}')
sys.exit(1 if bad else 0)
