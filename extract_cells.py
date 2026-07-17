import re
from pathlib import Path

path = Path('/Users/mingrun/reflexion_repro/create_notebook.py')
text = path.read_text(encoding='utf-8')
lines = text.splitlines()

cells_dir = Path('/Users/mingrun/reflexion_repro/cells')
cells_dir.mkdir(exist_ok=True)

i = 0
code_idx = 0
md_idx = 0
while i < len(lines):
    line = lines[i]
    code_match = re.match(r"^(code\d+) = '''", line)
    md_match = re.match(r"^(md\d+) = '''", line)
    if code_match or md_match:
        var = (code_match or md_match).group(1)
        is_code = var.startswith('code')
        # find closing ''' at depth 0
        depth = 1
        content_lines = []
        rest = line.split("'''", 1)[1]
        if rest:
            content_lines.append(rest)
        j = i + 1
        while j < len(lines) and depth > 0:
            cur = lines[j]
            if cur.strip() == "'''":
                # standalone closing delimiter
                depth -= 1
                if depth == 0:
                    break
                content_lines.append(cur)
            else:
                content_lines.append(cur)
            j += 1
        content = '\n'.join(content_lines)
        if is_code:
            code_idx += 1
            out = cells_dir / f'code_{code_idx:02d}.py'
        else:
            md_idx += 1
            out = cells_dir / f'md_{md_idx:02d}.md'
        out.write_text(content, encoding='utf-8')
        print(f'Wrote {out} ({len(content)} chars)')
        i = j + 1
    else:
        i += 1

print('Done')
