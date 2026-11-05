"""Refine the selected editorial direction with quantum graphics or whitespace."""
from pathlib import Path
import re

root = Path(__file__).parent / 'slide-previews'
source = (root / 'style-e.html').read_text(encoding='utf-8')
source = re.sub(r'<svg class="diagram".*?</svg>', '', source, flags=re.S)
source = source.replace('</style>', '''
/* === ENLARGED TEAM CREDIT === */
.team{font-size:34px;max-width:1300px;letter-spacing:-.35px}
.team-label{font-size:18px;letter-spacing:2px;margin-bottom:10px}
/* === QUANTUM FIGURES === */
.diagram{right:100px;top:280px;width:380px;height:470px}
.diagram text{fill:currentColor;font-family:'DM Sans',sans-serif;font-size:21px;letter-spacing:0}
.diagram .figure-heading{font-size:25px;font-weight:500}
.diagram .figure-note{font-size:19px;fill:var(--muted)}
.diagram .math{font-family:'DM Serif Display',serif;font-size:28px}
</style>''')

def figure(body, label):
    return f'<svg class="diagram" viewBox="0 0 380 470" fill="none" role="img" aria-label="{label}">{body}</svg>'

grover = figure('''
<!-- Ideal Grover search: four equiprobable states, one marked state,
     one Grover iteration. Bars depict exact ideal final probabilities. -->
<text x="190" y="58" text-anchor="middle" class="figure-heading">Grover search</text>
<path d="M56 120V356H342" stroke="currentColor" stroke-opacity=".45" stroke-width="1.5"/>
<text x="42" y="134" text-anchor="end" class="figure-note">1</text>
<text x="42" y="361" text-anchor="end" class="figure-note">0</text>
<path d="M56 300H342" stroke="currentColor" stroke-opacity=".5" stroke-width="1.5" stroke-dasharray="5 6"/>
<text x="64" y="288" class="figure-note">initial probability: ¼</text>
<path d="M94 350V356M162 350V356M230 350V356" stroke="currentColor" stroke-width="7"/>
<rect x="281" y="132" width="36" height="224" rx="3" fill="currentColor"/>
<circle cx="299" cy="116" r="5" fill="#a97850"/>
<text x="94" y="389" text-anchor="middle">00</text>
<text x="162" y="389" text-anchor="middle">01</text>
<text x="230" y="389" text-anchor="middle">10</text>
<text x="299" y="389" text-anchor="middle">11</text>
<text x="190" y="430" text-anchor="middle" class="figure-note">Ideal two-qubit example</text>
''', 'Ideal two-qubit Grover search with state 11 marked: one iteration raises its probability from one quarter to one.')

bloch = figure('''
<!-- Bloch sphere with the |+> state along positive x. Dashed segments
     show the far half of the equator and the negative axis directions. -->
<text x="190" y="58" text-anchor="middle" class="figure-heading">A qubit, made visible</text>
<circle cx="190" cy="244" r="126" stroke="currentColor" stroke-opacity=".5" stroke-width="1.5"/>
<path d="M64 244A126 43 0 0 1 316 244" stroke="currentColor" stroke-opacity=".4" stroke-width="1.5" stroke-dasharray="5 6"/>
<path d="M64 244A126 43 0 0 0 316 244" stroke="currentColor" stroke-opacity=".65" stroke-width="1.5"/>
<ellipse cx="190" cy="244" rx="44" ry="126" stroke="currentColor" stroke-opacity=".2"/>
<path d="M190 244V93M190 244L338 244M190 244L95 313" stroke="currentColor" stroke-opacity=".65" stroke-width="1.5"/>
<path d="M190 244V389M190 244H45M190 244L270 186" stroke="currentColor" stroke-opacity=".25" stroke-dasharray="4 6"/>
<path d="M190 244L95 313M95 313L100 298M95 313L111 313" stroke="currentColor" stroke-width="4"/>
<circle cx="95" cy="313" r="7" fill="#a97850"/>
<circle cx="190" cy="244" r="4" fill="currentColor"/>
<text x="185" y="81">z</text><text x="343" y="250">y</text><text x="73" y="338">x</text>
<text x="42" y="308" class="math">|+⟩</text>
<text x="190" y="430" text-anchor="middle" class="math">|+⟩ = (|0⟩ + |1⟩) / √2</text>
''', 'Bloch sphere showing the plus state on the positive x axis, an equal superposition of zero and one.')

circuit = figure('''
<!-- This is a valid two-qubit Bell-state circuit, not an assertion
     that this exact circuit was emitted by qlang. -->
<text x="190" y="58" text-anchor="middle" class="figure-heading">From logic to gates</text>
<text x="20" y="202" class="math">|0⟩</text>
<text x="20" y="302" class="math">|0⟩</text>
<path d="M70 194H346M70 294H346" stroke="currentColor" stroke-width="2"/>
<rect x="108" y="166" width="56" height="56" fill="#f1eee5" stroke="currentColor" stroke-width="2"/>
<text x="136" y="205" text-anchor="middle" class="math">H</text>
<path d="M250 194V294" stroke="currentColor" stroke-width="2"/>
<circle cx="250" cy="194" r="7" fill="currentColor"/>
<circle cx="250" cy="294" r="21" fill="#f1eee5" stroke="currentColor" stroke-width="2"/>
<path d="M229 294H271M250 273V315" stroke="currentColor" stroke-width="2"/>
<text x="190" y="382" text-anchor="middle" class="math">(|00⟩ + |11⟩) / √2</text>
<text x="190" y="430" text-anchor="middle" class="figure-note">Bell-state preparation</text>
''', 'A Bell-state preparation circuit: Hadamard on the first zero-state qubit followed by a controlled X on the second qubit.')

variants = [
    ('e1', 'Grover Search', 'A meaningful link to qlang: an ideal two-qubit amplitude-amplification result.', grover),
    ('e2', 'Bloch Sphere', 'An accurate qubit-state diagram with a warm accent on the plus state.', bloch),
    ('e3', 'Quantum Circuit', 'A small, valid Bell-state circuit in the same quiet editorial line style.', circuit),
    ('e4', 'Open Space', 'No graphic. The title, warm paper, and enlarged team credit carry the slide.', ''),
]

for key, name, description, graphic in variants:
    html = source.replace('<footer>', graphic + '<footer>', 1)
    (root / f'style-{key}.html').write_text(html, encoding='utf-8')

gallery = (root / 'compare.html').read_text(encoding='utf-8')
gallery = gallery.replace('qlang — Five visual directions', 'qlang — Editorial refinements')
gallery = gallery.replace('Five ways to introduce qlang.', 'One direction. Four refinements.')
gallery = gallery.replace('Restrained blues, warm neutrals, and room to explain.', 'Blue Marginalia, with a larger team credit and a choice of quantum graphic or open space.')
cards = ''.join(f'<article><div class="label"><h2>{key.upper()} / {name}</h2><a href="style-{key}.html" target="_blank" rel="noopener">Open full slide ↗</a></div><iframe title="{name} title slide" src="style-{key}.html" loading="eager"></iframe><p>{description}</p></article>' for key,name,description,_ in variants)
gallery = re.sub(r'<div class="grid">.*?</div></main>', '<div class="grid">' + cards + '</div></main>', gallery, flags=re.S)
(root / 'compare-e.html').write_text(gallery, encoding='utf-8')
print('Created four E refinements and compare-e.html; team credit increased from 25px to 34px.')
