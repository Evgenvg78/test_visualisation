from pathlib import Path
import re
text = Path('app.py').read_text(encoding='utf-8')
m = re.search(r'DEFAULT_MODE = \"([^\"]*)\"', text)
print(m.group(1))
print([ord(c) for c in m.group(1)])
