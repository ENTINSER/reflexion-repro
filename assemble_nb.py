import nbformat as nbf
from pathlib import Path

cells_dir = Path('/Users/mingrun/reflexion_repro/cells')
out_path = Path('/Users/mingrun/reflexion_repro/reflexion_reproduction.ipynb')

nb = nbf.v4.new_notebook()
nb.cells = []

# papermill parameters cell (must be first code cell and tagged 'parameters')
params_cell = nbf.v4.new_code_cell("""# Parameters (overridden by papermill)
N_SAMPLES = 50
""")
params_cell.metadata['tags'] = ['parameters']
nb.cells.append(params_cell)

for i in range(1, 10):
    md_file = cells_dir / f'md_{i:02d}.md'
    code_file = cells_dir / f'code_{i:02d}.py'
    md_src = md_file.read_text(encoding='utf-8')
    code_src = code_file.read_text(encoding='utf-8')
    nb.cells.append(nbf.v4.new_markdown_cell(md_src))
    nb.cells.append(nbf.v4.new_code_cell(code_src))

nb.metadata['kernelspec'] = {
    'display_name': 'Python 3',
    'language': 'python',
    'name': 'python3'
}

with open(out_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f'Notebook written to {out_path}')
print(f'Total cells: {len(nb.cells)}')
