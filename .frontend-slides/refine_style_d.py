"""Recreate the four quantum/empty choices in the selected D layout."""
from pathlib import Path
import re

root = Path(__file__).parent / 'slide-previews'
source = (root / 'style-d.html').read_text(encoding='utf-8')
source = re.sub(r'<div class="tiles".*?</div>', '', source, flags=re.S)
source = source.replace('</style>', '''
/* === ENLARGED TEAM CREDIT === */
.team{font-size:34px;max-width:1300px;letter-spacing:-.35px}
.team-label{font-size:18px;letter-spacing:2px;margin-bottom:10px}
/* === QUANTUM FIGURES IN THE BLUE SIDEBAR === */
.diagram{position:absolute;right:25px;top:280px;width:380px;height:470px;color:var(--accent)}
.diagram text{fill:currentColor;font-family:'DM Sans',sans-serif;font-size:21px;letter-spacing:0}
.diagram .figure-heading{font-size:25px;font-weight:500}
.diagram .figure-note{font-size:19px;fill:var(--muted)}
.diagram .math{font-family:'DM Serif Display',serif;font-size:28px}
</style>''')

variants = [
    ('d1', 'Grover Search', 'An ideal two-qubit Grover-search result, placed within the blue sidebar.'),
    ('d2', 'Bloch Sphere', 'A qubit-state illustration in the clean split layout.'),
    ('d3', 'Quantum Circuit', 'A valid Bell-state preparation circuit, with gate fills matched to the blue panel.'),
    ('d4', 'Open Space', 'The original geometric tiles are removed, leaving the blue sidebar empty.'),
]
for index, (key, name, description) in enumerate(variants, start=1):
    graphic = ''
    if index < 4:
        previous = (root / f'style-e{index}.html').read_text(encoding='utf-8')
        match = re.search(r'<svg class="diagram".*?</svg>', previous, re.S)
        assert match, f'Quantum figure missing from E{index}'
        graphic = match.group().replace('fill="#f1eee5"', 'fill="#dce6ed"')
    html = source.replace('<footer>', graphic + '<footer>', 1)
    (root / f'style-{key}.html').write_text(html, encoding='utf-8')

gallery = (root / 'compare.html').read_text(encoding='utf-8')
gallery = gallery.replace('qlang — Five visual directions', 'qlang — D refinements')
gallery = gallery.replace('Five ways to introduce qlang.', 'Porcelain &amp; Ink. Four refinements.')
gallery = gallery.replace('Restrained blues, warm neutrals, and room to explain.', 'D’s clean layout, with a larger team credit and quantum graphics or an empty blue sidebar.')
cards = ''.join(f'<article><div class="label"><h2>{key.upper()} / {name}</h2><a href="style-{key}.html" target="_blank" rel="noopener">Open full slide ↗</a></div><iframe title="{name} title slide" src="style-{key}.html" loading="eager"></iframe><p>{description}</p></article>' for key,name,description in variants)
gallery = re.sub(r'<div class="grid">.*?</div></main>', '<div class="grid">' + cards + '</div></main>', gallery, flags=re.S)
(root / 'compare-d.html').write_text(gallery, encoding='utf-8')

for key, _, _ in variants:
    html = (root / f'style-{key}.html').read_text(encoding='utf-8')
    assert '<div class="tiles"' not in html
    assert 'font-size:34px;max-width:1300px' in html
    assert 'Quadruply Quick Quantum Quad Squad' in html
    assert html.count('<svg class="diagram"') == (0 if key == 'd4' else 1)
print('Created D1–D4 and compare-d.html; checked graphics, removed tiles, and enlarged full team credit.')
