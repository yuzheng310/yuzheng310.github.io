"""Check generated pages and internal destinations after the production build."""
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlsplit, unquote
import sys

root = Path(__file__).resolve().parents[1] / 'dist'
class Page(HTMLParser):
    def __init__(self, source):
        super().__init__()
        self.ids, self.links, self.errors = set(), [], []
        self.h1 = 0
        self.feed(source)
    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if 'id' in a:
            if a['id'] in self.ids: self.errors.append('duplicate id: ' + a['id'])
            self.ids.add(a['id'])
        if tag == 'h1': self.h1 += 1
        if tag == 'img' and a.get('src', '').startswith('/') and not (a.get('width') and a.get('height')):
            self.errors.append('image missing reserved dimensions: ' + a['src'])
        if 'katex-error' in a.get('class', '').split(): self.errors.append('KaTeX parse error')
        for attr in ('href', 'src'):
            if a.get(attr): self.links.append(a[attr])

pages = {p: Page(p.read_text()) for p in root.rglob('*.html')}
errors = []
checks = 0
for path, page in pages.items():
    route = '/' + path.relative_to(root).as_posix()
    if page.h1 != 1: errors.append(f'{route}: expected one h1, got {page.h1}')
    errors.extend(f'{route}: {error}' for error in page.errors)
    for link in page.links:
        url = urlsplit(urljoin('https://yuzheng310.github.io' + route, link))
        if url.scheme not in ('https', 'http') or url.netloc != 'yuzheng310.github.io': continue
        target = root / unquote(url.path).lstrip('/')
        if target.is_dir(): target /= 'index.html'
        checks += 1
        if not target.exists(): errors.append(f'{route}: missing {link}')
        elif url.fragment and target in pages and unquote(url.fragment) not in pages[target].ids:
            errors.append(f'{route}: missing anchor {link}')
for asset in ['pagefind/pagefind.js', 'pagefind/pagefind-ui.js', 'pagefind/pagefind-ui.css']:
    if not (root / asset).is_file(): errors.append(f'missing search asset: {asset}')
if errors:
    print('\n'.join(errors)); sys.exit(1)
print(f'PASS: {len(pages)} pages, {checks} internal links/assets; headings, anchors, math and search assets valid.')
