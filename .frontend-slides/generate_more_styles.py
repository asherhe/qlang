"""Generate five single-slide directions and a local comparison gallery."""
from pathlib import Path

ROOT = Path(__file__).parent / 'slide-previews'
ROOT.mkdir(parents=True, exist_ok=True)
BASE = Path('C:/Users/franc/.agents/skills/frontend-slides/viewport-base.css').read_text(encoding='utf-8')
TEAM = 'Quadruply Quick Quantum Quad Squad'
FONTS = 'https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;800&family=DM+Sans:wght@400;500;600;700&family=DM+Serif+Display:ital@0;1&family=Libre+Caslon+Display&family=Outfit:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap'
COMMON = '''
/* === SHARED TYPOGRAPHY AND SPACING === */
*{box-sizing:border-box} :root{--ink:#202c3b;--accent:#315c91;--muted:#526173;--line:#bfc8cf;--slide-bg:#f5f3ed;--stage-bg:#1d2630}
.slide{color:var(--ink);font-family:'DM Sans',sans-serif;padding:80px 104px}
header{display:flex;align-items:center;justify-content:space-between;height:56px}
.brand{font:800 42px Manrope,sans-serif;letter-spacing:-2px}
.eyebrow{font:500 22px 'IBM Plex Mono',monospace;letter-spacing:2px;text-transform:uppercase}
h1{margin:0;font-weight:400;line-height:1.04;letter-spacing:-4px}
p{margin:0}.subtitle{font-size:34px;line-height:1.5;color:var(--muted)}
footer{position:absolute;left:104px;right:104px;bottom:64px;display:flex;justify-content:space-between;align-items:center;gap:32px;font-size:25px;line-height:1.4}
.team{max-width:1000px}.team-label{display:block;font:400 17px 'IBM Plex Mono',monospace;letter-spacing:2px;text-transform:uppercase;margin-bottom:10px;color:var(--muted)}
.page{font:400 21px 'IBM Plex Mono',monospace}.accent{color:var(--accent)}
.quad{display:grid;grid-template-columns:repeat(2,1fr);gap:12px;width:60px;height:60px}.quad i{background:var(--accent)}
/* === ENTRANCE MOTION === */
.reveal{animation:enter .65s cubic-bezier(.16,1,.3,1) both}.subtitle.reveal{animation-delay:.12s}
@keyframes enter{from{opacity:0;transform:translateY(18px)}to{opacity:1;transform:translateY(0)}}
@media(prefers-reduced-motion:reduce){.reveal{animation:none}}
'''
SCRIPT = '''
/* === UNIFORM 16:9 STAGE SCALING === */
function fit(){const s=Math.min(innerWidth/1920,innerHeight/1080);document.querySelector('.deck-stage').style.transform=`translate(${(innerWidth-1920*s)/2}px,${(innerHeight-1080*s)/2}px) scale(${s})`;}
addEventListener('resize',fit);fit();
'''
QUAD = '<div class="quad" aria-hidden="true"><i></i><i></i><i></i><i></i></div>'
FOOTER = f'<footer><div class="team"><span class="team-label">Presented by</span>{TEAM}</div><span class="page">QISKIT · 01</span></footer>'
HEADER = '<header><span class="brand">qlang<span class="accent">.</span></span><span class="eyebrow">Expressions → quantum circuits</span></header>'
SUBTITLE = '<p class="subtitle reveal">A familiar language. A visible bridge to Qiskit.</p>'

styles = [
('d', 'Porcelain & Ink', 'A clean pitch layout with a narrow blue side panel and four geometric tiles.', '''
/* === PORCELAIN AND INK === */
:root{--slide-bg:#f8f7f2;--accent:#315c91}
.rail{position:absolute;top:0;bottom:0;right:0;width:430px;background:#dce6ed}
header{position:relative}.eyebrow{margin-right:390px;font-size:20px}
h1{font:800 138px/1.08 Manrope,sans-serif;letter-spacing:-7px;position:absolute;top:274px;left:104px}
.subtitle{position:absolute;top:646px;left:112px;width:1080px;font-size:35px}
.tiles{position:absolute;right:84px;top:342px;display:grid;grid-template-columns:repeat(2,124px);gap:20px;transform:rotate(-9deg)}
.tiles i{width:124px;height:124px;background:#315c91}.tiles i:nth-child(2){background:#8ca9bf}.tiles i:nth-child(3){background:#b89c7a}.tiles i:nth-child(4){background:#f8f7f2;border:2px solid #315c91}
footer{right:70px}.page{color:#223d5b}footer:before{content:'';position:absolute;left:0;top:-32px;width:1170px;border-top:1px solid var(--line)}
''', f'<div class="rail" aria-hidden="true"></div>{HEADER}<h1 class="reveal">Think in problems.<br><span class="accent">Learn the quantum.</span></h1>{SUBTITLE}<div class="tiles" aria-hidden="true"><i></i><i></i><i></i><i></i></div>{FOOTER}'),
('e', 'Blue Marginalia', 'Warm paper, expressive serif typography, and a quiet blue orbital drawing.', '''
/* === BLUE MARGINALIA === */
:root{--slide-bg:#f1eee5;--accent:#365d85;--muted:#566270}
header{border-bottom:1px solid var(--line);padding-bottom:36px;height:92px}.brand{font-size:35px}.eyebrow{font-size:19px}
h1{position:absolute;left:104px;top:286px;font-family:'DM Serif Display',serif;font-size:146px;line-height:1.06;letter-spacing:-3px}
h1 em{color:var(--accent)}.subtitle{position:absolute;left:112px;top:660px;width:1000px;font-size:33px}
.diagram{position:absolute;right:110px;top:300px;width:360px;height:420px;color:var(--accent)}
footer{border-top:1px solid var(--line);padding-top:28px}.team-label{font-size:16px}.page{color:var(--accent)}
''', f'{HEADER}<h1 class="reveal">Think in problems.<br><em>Learn the quantum.</em></h1>{SUBTITLE}<svg class="diagram" viewBox="0 0 360 420" fill="none" aria-hidden="true"><ellipse cx="180" cy="210" rx="154" ry="66" stroke="currentColor" stroke-width="2" transform="rotate(-40 180 210)"/><ellipse cx="180" cy="210" rx="154" ry="66" stroke="currentColor" stroke-width="2" transform="rotate(40 180 210)"/><circle cx="180" cy="210" r="145" stroke="currentColor" stroke-opacity=".25"/><circle cx="180" cy="210" r="11" fill="#b0784f"/><circle cx="290" cy="113" r="9" fill="currentColor"/><circle cx="64" cy="303" r="6" fill="currentColor"/></svg>{FOOTER}'),
('f', 'The Working Notebook', 'A structured academic notebook with a blue spine and a readable expression-to-circuit diagram.', '''
/* === THE WORKING NOTEBOOK === */
:root{--slide-bg:#f6f5ef;--accent:#2c587f}
.spine{position:absolute;left:0;top:0;bottom:0;width:28px;background:var(--accent)}
.slide{background-image:linear-gradient(#2c587f09 1px,transparent 1px);background-size:100% 48px}
.eyebrow{font-size:21px}.brand{font-family:Outfit,sans-serif;font-weight:600}
h1{position:absolute;left:104px;top:268px;font:500 128px/1.09 Outfit,sans-serif;letter-spacing:-5px;max-width:1250px}
.subtitle{position:absolute;left:110px;top:589px;font-size:32px}
.flow{position:absolute;top:738px;left:110px;display:flex;align-items:center;gap:30px;font:24px 'IBM Plex Mono',monospace}
.flow .node{border-bottom:3px solid var(--accent);padding:16px 0;color:var(--accent)}.flow .arrow{color:var(--muted)}
.margin{position:absolute;right:106px;top:300px;border-left:1px solid #bcc8cf;padding-left:36px;width:240px}
.margin .quad{width:96px;height:96px;margin-bottom:34px;gap:8px}.margin p{font:21px/1.6 'IBM Plex Mono',monospace;color:var(--muted)}
footer{bottom:46px}.team-label{font-size:15px;margin-bottom:4px}
''', f'<div class="spine" aria-hidden="true"></div>{HEADER}<h1 class="reveal">Think in problems.<br><span class="accent">Learn the quantum.</span></h1>{SUBTITLE}<div class="flow"><span class="node">expression</span><span class="arrow">→</span><span class="node">oracle</span><span class="arrow">→</span><span class="node">Grover search</span></div><aside class="margin">{QUAD}<p>Write.<br>Translate.<br>Understand.</p></aside>{FOOTER}'),
('g', 'Slate & Sand', 'A spacious editorial cover with oversized serif type and a blue architectural bridge.', '''
/* === SLATE AND SAND === */
:root{--slide-bg:#ece7dd;--accent:#476784;--muted:#505d69}
.eyebrow{font-size:19px}.brand{font-size:36px}
h1{position:absolute;left:104px;top:250px;font:400 164px/1.02 'Libre Caslon Display',serif;letter-spacing:-4px}
.subtitle{position:absolute;left:112px;top:635px;width:1000px;font-size:32px}
.bridge{position:absolute;right:140px;top:352px;width:300px;height:340px;border-radius:160px 160px 0 0;background:var(--accent)}
.bridge:before{content:'';position:absolute;left:83px;right:83px;bottom:0;height:246px;border-radius:90px 90px 0 0;background:var(--slide-bg)}
.bridge:after{content:'';position:absolute;left:-40px;right:-40px;height:2px;background:#a28b70;bottom:-32px}
.caption{position:absolute;right:102px;top:748px;width:375px;text-align:center;font:18px 'IBM Plex Mono',monospace;color:var(--muted)}
footer{border-top:1px solid #aab2b9;padding-top:30px}
''', f'{HEADER}<h1 class="reveal">Think in problems.<br><span class="accent">Learn the quantum.</span></h1>{SUBTITLE}<div class="bridge" aria-hidden="true"></div><p class="caption">A BRIDGE TO QISKIT</p>{FOOTER}'),
('h', 'Fourfold', 'A precise modern layout with a small blue band and four interlocking rounded squares.', '''
/* === FOURFOLD === */
:root{--slide-bg:#faf9f5;--accent:#3c648d;--muted:#536170}
.band{position:absolute;left:0;right:0;bottom:0;height:208px;background:#e1e9ef}
.brand{font-family:Outfit,sans-serif;font-weight:600;font-size:44px}.eyebrow{font-size:20px}
h1{position:absolute;left:104px;top:278px;font:600 134px/1.08 Outfit,sans-serif;letter-spacing:-5px;max-width:1260px}
.subtitle{position:absolute;left:110px;top:626px;font-size:33px}
.symbol{position:absolute;right:110px;top:337px;width:310px;height:310px}
.symbol i{position:absolute;width:184px;height:184px;border:3px solid var(--accent);border-radius:50px}
.symbol i:nth-child(2){right:0;background:#c3d4e1}.symbol i:nth-child(3){bottom:0;background:#e9e2d4}.symbol i:nth-child(4){bottom:0;right:0;background:var(--accent)}
footer{bottom:54px}.team-label{color:#4c5c6b}.page{color:var(--accent)}
''', f'<div class="band" aria-hidden="true"></div>{HEADER}<h1 class="reveal">Think in problems.<br><span class="accent">Learn the quantum.</span></h1>{SUBTITLE}<div class="symbol" aria-hidden="true"><i></i><i></i><i></i><i></i></div>{FOOTER}')
]

for key, name, desc, css, html in styles:
    output = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>qlang — {TEAM}</title><link rel="stylesheet" href="{FONTS}">
<style>/* === MANDATORY FIXED STAGE === */\n{BASE}\n{COMMON}\n{css}</style></head>
<body><div class="deck-viewport"><main class="deck-stage"><section class="slide active" aria-label="qlang title slide">{html}</section></main></div>
<script>{SCRIPT}</script></body></html>'''
    (ROOT / f'style-{key}.html').write_text(output, encoding='utf-8')

# This responsive page is outside the fixed slide stage. Its embedded slides
# retain their 1920 x 1080 authored layout at every comparison-card size.
cards = ''.join(f'''<article><div class="label"><h2>{key.upper()} / {name}</h2><a href="style-{key}.html" target="_blank" rel="noopener">Open full slide ↗</a></div><iframe title="{name} title slide" src="style-{key}.html" loading="lazy"></iframe><p>{desc}</p></article>''' for key,name,desc,_,_ in styles)
gallery = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>qlang — Five visual directions</title><link rel="stylesheet" href="{FONTS}"><style>
/* === COMPARISON PAGE === */
*{{box-sizing:border-box}}body{{margin:0;background:#eeede8;color:#202c3b;font-family:'DM Sans',sans-serif;padding:48px 5vw 72px}}main{{max-width:1560px;margin:auto}}h1{{font:500 clamp(32px,4vw,58px) Outfit,sans-serif;margin:0 0 12px;letter-spacing:-1px}}.intro{{font-size:18px;line-height:1.6;margin-bottom:40px;color:#526173}}.grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:40px 32px}}article{{min-width:0}}.label{{display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap;margin-bottom:12px}}h2{{font-size:18px;margin:0}}a{{color:#315c91;min-height:44px;display:inline-flex;align-items:center;font-size:14px;text-underline-offset:4px}}a:hover{{color:#173651}}a:focus-visible{{outline:3px solid #315c91;outline-offset:5px}}iframe{{display:block;width:100%;aspect-ratio:16/9;border:1px solid #c1c9cd;background:#f5f3ed;pointer-events:none}}article p{{font-size:16px;line-height:1.6;color:#526173;margin-top:12px;max-width:65ch}}@media(max-width:850px){{.grid{{grid-template-columns:1fr}}body{{padding:30px 20px}}}}
</style></head><body><main><h1>Five ways to introduce qlang.</h1><p class="intro">Restrained blues, warm neutrals, and room to explain.<br>{TEAM}</p><div class="grid">{cards}</div></main></body></html>'''
(ROOT / 'compare.html').write_text(gallery, encoding='utf-8')
print('Created style-d.html through style-h.html and compare.html')

