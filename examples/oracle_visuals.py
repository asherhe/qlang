"""Presentation helpers for qlang_demo.ipynb; reuse the real AST and circuit.

These helpers add presentation details to a circuit copy. They never alter the
compiled oracle or introduce a second expression parser or graph renderer.
"""

from io import StringIO
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt

from qlang.demo import oracle_phase_rows
from qlang.expression import Bit
from live_ast_widgets import expression_text, render_graph

# Match the existing expression explorer and the reference slides.
BACKGROUND = '#f8f7f2'
TEXT = '#202c3b'
MUTED = '#526173'
MARKED = '#6d28d9'
UNCHANGED = '#526173'


def annotated_circuit(oracle):
    """Keep all real gates, adding display boundaries around the marking Z."""
    marking = [i for i, item in enumerate(oracle.circuit.data) if item.operation.name == 'z']
    if len(marking) != 1:
        raise ValueError('This presentation helper expects exactly one marking Z gate.')
    result = oracle.circuit.copy_empty_like()
    # copy_empty_like preserves registers/qubits, but may carry global phase;
    # the original phase is retained exactly once in the resulting circuit.
    for index, item in enumerate(oracle.circuit.data):
        if index == marking[0]:
            result.barrier(label='Mark')
        result.append(item.operation, item.qubits, item.clbits)
        if index == marking[0]:
            result.barrier(label='Uncompute')
    return result


def oracle_figure(oracle, *, label=''):
    """Use the notebook's Qiskit drawer, with boundaries and slide colors."""
    circuit = annotated_circuit(oracle)
    figure = circuit.draw('mpl', fold=-1, style={
        'backgroundcolor': BACKGROUND,
        'displaycolor': {'z': (MARKED, '#ffffff')},
    })
    # Match the AST's blue variable and purple operation colors. Read the input
    # map and the real Z target rather than assuming particular wire numbers.
    input_labels = {qubit: f'{name}[{bit}]'
                    for name, qubits in oracle.inputs.items()
                    for bit, qubit in enumerate(qubits)}
    z_item = next(item for item in oracle.circuit.data if item.operation.name == 'z')
    predicate_wire = oracle.circuit.find_bit(z_item.qubits[0]).index
    for text in figure.axes[0].texts:
        match = re.fullmatch(r'\$\{(\d+)\}\$', text.get_text())
        if match is None:
            continue
        wire = int(match.group(1))
        text.set_clip_on(False)
        if wire in input_labels:
            text.set_text(input_labels[wire])
            text.set_color('#1d4ed8')
        else:
            text.set_text(f'work[{wire}]')
            text.set_color(MUTED)
        if wire == predicate_wire:
            text.set_text(text.get_text() + ' / predicate' if wire in input_labels else 'predicate')
            text.set_color(MARKED)
    figure.set_facecolor(BACKGROUND)
    title = 'Compute → Mark (Z) → Uncompute'
    if label:
        title = f'{label}   |   {title}'
    figure.axes[0].set_title(title, loc='left', color=TEXT, fontsize=14, pad=20)
    return figure


def _graph_svg(predicate):
    # The existing renderer returns notebook HTML containing its SVG.
    markup, *_ = render_graph(predicate)
    return re.search(r'<svg\b.*?</svg>', markup, flags=re.S).group(0)


def _overview(predicate, figure):
    """Put the existing AST and real circuit on one connected SVG canvas.

    A single canvas keeps the arrow attached to both figures in narrow notebook
    panes. Neither renderer is duplicated, and both parts remain vector artwork.
    """
    namespace = 'http://www.w3.org/2000/svg'
    ET.register_namespace('', namespace)
    ET.register_namespace('xlink', 'http://www.w3.org/1999/xlink')
    def tag(name):
        return f'{{{namespace}}}{name}'

    graph = ET.fromstring(_graph_svg(predicate))
    gx, gy, gw, gh = map(float, graph.attrib['viewBox'].split())
    # The AST renderer reserves extra bottom space for its scrollable viewer.
    # Trim only that empty margin in the combined diagram; keep all graph nodes.
    node_bottoms = [float(rect.get('y')) + float(rect.get('height'))
                    for rect in graph.iter(tag('rect')) if rect.get('y') is not None]
    if node_bottoms:
        gh = min(gh, max(node_bottoms) - gy + 28)
        graph.set('viewBox', f'{gx} {gy} {gw} {gh}')
    graph_scale = min(720 / gw, 300 / gh)
    graph_width, graph_height = gw * graph_scale, gh * graph_scale

    circuit_svg = StringIO()
    figure.savefig(circuit_svg, format='svg', bbox_inches='tight', facecolor=BACKGROUND)
    circuit = ET.fromstring(circuit_svg.getvalue())
    _, _, cw, ch = map(float, circuit.attrib['viewBox'].split())
    width = 1600
    circuit_width = width - 40
    circuit_height = ch * circuit_width / cw
    arrow_top = 75 + graph_height + 8
    circuit_top = arrow_top + 125
    height = circuit_top + circuit_height + 55
    root = ET.Element(tag('svg'), {
        'width': str(width), 'height': str(round(height)),
        'viewBox': f'0 0 {width} {height}', 'role': 'img',
        'aria-label': f'Expression graph connected to the compiled oracle for {expression_text(predicate)}',
        'style': 'max-width:100%;height:auto;display:block',
    })
    ET.SubElement(root, tag('rect'), {'width': '100%', 'height': '100%', 'fill': BACKGROUND})
    def text(x, y, value, *, size=24, color=TEXT, anchor='middle', bold=False):
        element = ET.SubElement(root, tag('text'), {
            'x': str(x), 'y': str(y), 'fill': color, 'text-anchor': anchor,
            'font-family': 'system-ui,sans-serif', 'font-size': str(size),
            'font-weight': '700' if bold else '400',
        })
        element.text = value
    text(30, 40, f'Compiling {expression_text(predicate)}', size=34, anchor='start', bold=True)
    graph.set('x', str((width - graph_width) / 2))
    graph.set('y', '65')
    graph.set('width', str(graph_width)); graph.set('height', str(graph_height))
    root.append(graph)

    # This arrow connects the complete expression to the complete circuit:
    # a comparison expands into a block of gates, rather than one EQ gate.
    defs = ET.SubElement(root, tag('defs'))
    marker = ET.SubElement(defs, tag('marker'), {
        'id': 'oracle-compile-arrow', 'viewBox': '0 0 10 10', 'refX': '9', 'refY': '5',
        'markerWidth': '7', 'markerHeight': '7', 'orient': 'auto',
    })
    ET.SubElement(marker, tag('path'), {'d': 'M 0 0 L 10 5 L 0 10 Z', 'fill': MUTED})
    ET.SubElement(root, tag('line'), {
        'x1': str(width / 2), 'y1': str(arrow_top),
        'x2': str(width / 2), 'y2': str(arrow_top + 48),
        'stroke': MUTED, 'stroke-width': '3', 'marker-end': 'url(#oracle-compile-arrow)',
    })
    text(width / 2 + 25, arrow_top + 30, 'compile_oracle', size=23, anchor='start', color=MUTED)
    text(width / 2 - 20, arrow_top + 88, 'Variables → blue input wires', color='#1d4ed8', anchor='end', size=23)
    text(width / 2 + 20, arrow_top + 88, 'predicate → purple wire and Z', color=MARKED, anchor='start', size=23)
    circuit.set('x', '20'); circuit.set('y', str(circuit_top))
    circuit.set('width', str(circuit_width)); circuit.set('height', str(circuit_height))
    root.append(circuit)
    text(width / 2, height - 16, 'Inputs are restored and temporary qubits return to zero.', size=21, color=MUTED)
    return ET.tostring(root, encoding='unicode')


def show_oracle(predicate: Bit, oracle):
    """Display one connected expression-to-circuit diagram."""
    from IPython.display import SVG, display

    figure = oracle_figure(oracle, label=expression_text(predicate))
    try:
        display(SVG(_overview(predicate, figure)))
    finally:
        plt.close(figure)


def plot_oracle_phases(predicate: Bit, oracle):
    """Verify every x, then plot signed phases, not measurement probabilities.

    This compact view intentionally covers the notebook's single-input example.
    The existing truth-table section covers x + y == 5 with all 32 assignments.
    """
    if set(oracle.inputs) != {'x'}:
        raise ValueError('Use this compact plot for x == 5; use print_truth_table for x + y == 5.')
    width = len(oracle.inputs['x'])
    if width > 5:
        raise ValueError('Keep this presentation plot to at most 32 inputs.')
    assignments = [{'x': value} for value in range(1 << width)]
    rows = oracle_phase_rows(predicate, assignments, oracle=oracle)
    signs = [sign for _, _, sign in rows]
    colors = [MARKED if expected else UNCHANGED for _, expected, _ in rows]
    figure, axis = plt.subplots(figsize=(11, 3.5), facecolor=BACKGROUND)
    axis.set_facecolor(BACKGROUND)
    axis.bar(range(len(rows)), signs, color=colors, width=0.55)
    axis.axhline(0, color=MUTED, linewidth=1)
    axis.set_xticks(range(len(rows)), [str(row[0]['x']) for row in rows])
    axis.set_yticks([-1, 0, 1], ['−1', '0', '+1'])
    axis.set_ylim(-1.45, 1.45)
    axis.set_xlabel('Input x', color=MUTED)
    axis.set_ylabel('Phase sign', color=MUTED)
    axis.set_title(f'Phase oracle: {expression_text(predicate)}', loc='left', color=TEXT, fontsize=16, pad=16)
    axis.tick_params(colors=MUTED, length=0)
    for side in ('top', 'right', 'left'):
        axis.spines[side].set_visible(False)
    axis.spines['bottom'].set_color(MUTED)
    for value, sign in enumerate(signs):
        axis.text(value, sign + (0.13 if sign > 0 else -0.13), f'{sign:+d}',
                  ha='center', va='bottom' if sign > 0 else 'top', color=colors[value], fontsize=13)
    figure.text(0.5, 0.015, f'All {len(rows)} inputs verified; inputs restored and workspace cleared. Signs are relative phases, not probabilities.',
                ha='center', color=MUTED, fontsize=10)
    figure.tight_layout(rect=(0, 0.07, 1, 1))
    return figure


def export_oracle_visuals(predicate: Bit, oracle, directory, *, stem='oracle'):
    """Export vector graphs and circuit/phase images; compile once in the caller.

    The SVG expression graph is the original renderer's output. HTML can display
    it directly without introducing a separate SVG rendering dependency.
    """
    if Path(stem).name != stem or stem in ('', '.', '..'):
        raise ValueError('stem must be a filename, not a directory path')
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    paths = {}
    graph = directory / f'{stem}-expression.svg'
    graph.write_text(_graph_svg(predicate), encoding='utf-8')
    paths['expression_svg'] = graph
    figure = oracle_figure(oracle, label=expression_text(predicate))
    try:
        for extension in ('png', 'svg'):
            path = directory / f'{stem}-circuit.{extension}'
            figure.savefig(path, dpi=220, bbox_inches='tight', facecolor=BACKGROUND)
            paths[f'circuit_{extension}'] = path
        overview = _overview(predicate, figure)
        overview_svg = directory / f'{stem}-overview.svg'
        overview_svg.write_text(overview, encoding='utf-8')
        paths['overview_svg'] = overview_svg
        page = directory / f'{stem}-overview.html'
        page.write_text('<!doctype html><html><head><meta charset="utf-8"><title>Compiled oracle</title></head><body>'
                        + overview + '</body></html>', encoding='utf-8')
        paths['overview_html'] = page
    finally:
        plt.close(figure)
    if set(oracle.inputs) == {'x'}:
        figure = plot_oracle_phases(predicate, oracle)
        try:
            for extension in ('png', 'svg'):
                path = directory / f'{stem}-phases.{extension}'
                figure.savefig(path, dpi=220, bbox_inches='tight', facecolor=BACKGROUND)
                paths[f'phases_{extension}'] = path
        finally:
            plt.close(figure)
    return paths
