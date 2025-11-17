from pathlib import Path
lines = Path('app.py').read_text(encoding='utf-8').splitlines()
for i, line in enumerate(lines[:60]):
    print(i+1, repr(line))
