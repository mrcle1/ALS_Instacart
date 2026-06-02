import ast, os, sys
errs = []
for root, dirs, files in os.walk('.'):
    # skip artifacts and venvs
    dirs[:] = [d for d in dirs if d not in ('artifacts', '__pycache__', '.git', 'venv', '.venv')]
    for f in files:
        if f.endswith('.py'):
            p = os.path.join(root, f)
            try:
                ast.parse(open(p, encoding='utf-8').read(), filename=p)
            except SyntaxError as e:
                errs.append((p, e))
if errs:
    for p, e in errs:
        print(f'SYNTAX ERROR in {p}: {e}')
    sys.exit(1)
print('All Python files parse OK')