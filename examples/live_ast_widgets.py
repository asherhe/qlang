"""Live AST notebook UI. Reloaded from disk by the notebook launch cell."""

import ast
import html
import time
from collections import defaultdict

import qlang
from qlang.expression import BitType, UIntType


# Parse a small expression language explicitly; never eval user text.
BINARY = {
    ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b, ast.LShift: lambda a, b: a << b,
    ast.RShift: lambda a, b: a >> b, ast.BitAnd: lambda a, b: a & b,
    ast.BitOr: lambda a, b: a | b, ast.BitXor: lambda a, b: a ^ b,
}
COMPARE = {
    ast.Eq: lambda a, b: a == b, ast.NotEq: lambda a, b: a != b,
    ast.Lt: lambda a, b: a < b, ast.LtE: lambda a, b: a <= b,
    ast.Gt: lambda a, b: a > b, ast.GtE: lambda a, b: a >= b,
}

import keyword

DEFAULT_DECLARATIONS = [
    {"name": "x", "type": "UInt", "width": 3},
    {"name": "y", "type": "UInt", "width": 3},
    {"name": "p", "type": "Bit", "width": 1},
    {"name": "q", "type": "Bit", "width": 1},
]

def validate_declarations(declarations):
    if len(declarations) > 32:
        raise ValueError("Use at most 32 variables in this demo.")
    seen = set()
    for row in declarations:
        name = row["name"]
        if not isinstance(name, str) or not name.isidentifier() or keyword.iskeyword(name):
            raise ValueError(f"{name!r} is not a valid variable name. Use a Python identifier, not a keyword.")
        if name in seen:
            raise ValueError(f"Variable {name!r} is declared twice. Names must be unique.")
        seen.add(name)
        if row["type"] not in ("UInt", "Bit"):
            raise ValueError(f"Unknown variable type for {name!r}.")
        if type(row["width"]) is not int or not 1 <= row["width"] <= 32:
            raise ValueError(f"Width of {name!r} must be between 1 and 32.")
        if row["type"] == "Bit" and row["width"] != 1:
            raise ValueError(f"Boolean variable {name!r} must have width 1.")
    return declarations

def parse_expression(source, width=3, declarations=None):
    if not source.strip():
        raise ValueError("Type an expression to begin.")
    if len(source) > 500:
        raise ValueError("Keep this live demo below 500 characters.")
    tree = ast.parse(source, mode="eval")
    if sum(1 for _ in ast.walk(tree)) > 150:
        raise ValueError("Expression is too large for the live demo.")
    if declarations is None:
        declarations = [dict(row, width=width if row["type"] == "UInt" else 1)
                        for row in DEFAULT_DECLARATIONS]
    validate_declarations(declarations)
    qlang.ASTNode.reset()
    names = {row["name"]: (qlang.uint(row["name"], row["width"]) if row["type"] == "UInt"
                           else qlang.var_bit(row["name"])) for row in declarations}

    def build(node):
        if isinstance(node, ast.Name):
            if node.id not in names:
                raise ValueError(f"Unknown variable {node.id!r}. Add it to the variable table on the right.")
            return names[node.id]
        if isinstance(node, ast.Constant):
            if type(node.value) not in (int, bool) or not 0 <= node.value <= 255:
                raise ValueError("Constants must be integers from 0 to 255 or booleans.")
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in BINARY:
            left, right = build(node.left), build(node.right)
            # Bound constant arithmetic as well as symbolic widths.
            if isinstance(node.op, (ast.LShift, ast.RShift)):
                if type(right) is not int or not 0 <= right <= 8:
                    raise ValueError("Shifts need a literal integer amount from 0 to 8.")
            return BINARY[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Invert):
            value = build(node.operand)
            if not isinstance(value, qlang.Bit):
                raise ValueError("Use ~ only on Boolean expressions (for example ~p).")
            return ~value
        if isinstance(node, ast.Compare):
            if len(node.ops) != 1 or type(node.ops[0]) not in COMPARE:
                raise ValueError("Use separate comparisons joined by &: (x < y) & (y < 4).")
            return COMPARE[type(node.ops[0])](build(node.left), build(node.comparators[0]))
        raise ValueError("Use arithmetic, bitwise logic, and single comparisons; no calls or Python and/or/not.")

    root = build(tree.body)
    if type(root) is bool:
        root = qlang.const_bit(root)
    elif type(root) is int:
        if not 0 <= root <= 65535:
            raise ValueError("Constant result is outside the demo range.")
        root = qlang.uint(root)
    if not isinstance(root, qlang.ASTNode):
        raise ValueError("Expected a symbolic expression.")
    return root

# Pure SVG rendering: no Graphviz executable, JavaScript, or external assets.
SYMBOLS = {"ADD": "+", "SUB": "−", "MUL": "×", "LSHIFT": "<<", "RSHIFT": ">>",
           "AND": "&", "OR": "|", "XOR": "^", "NOT": "~", "EQ": "==",
           "NE": "!=", "LT": "<", "LE": "<=", "GT": ">", "GE": ">="}

def graph_data(root):
    nodes, edges, seen = [], [], set()
    def visit(node):
        key = id(node)
        if key in seen:
            return
        seen.add(key)
        nodes.append(node)
        for child in node.children:
            edges.append((key, id(child)))
            visit(child)
    visit(root)
    return nodes, edges

def node_text(node):
    kind = node.bit_type if isinstance(node, qlang.Bit) else node.uint_type
    if kind.name == "VAR":
        title, category = node.name, "variable"
    elif kind.name == "CONST":
        title = str(int(node.value)) if isinstance(node, qlang.Bit) else str(node.const_value)
        category = "constant"
    elif isinstance(node, qlang.Bit) and kind == BitType.UINT:
        title, category = f"bit {node.index}", "operation"
    else:
        title, category = SYMBOLS.get(kind.name, kind.name), "operation"
    subtitle = "Boolean" if isinstance(node, qlang.Bit) else f"UInt · {node.width} bits"
    return title, subtitle, category

def expression_text(root):
    """Readable, unambiguous expression represented by a semantic AST node."""
    cache = {}
    operators = dict(SYMBOLS, SUB="-", MUL="*")
    def format_node(node):
        key = id(node)
        if key in cache:
            return cache[key]
        kind = node.bit_type if isinstance(node, qlang.Bit) else node.uint_type
        if kind.name == "VAR":
            text = node.name
        elif kind.name == "CONST":
            text = str(int(node.value)) if isinstance(node, qlang.Bit) else str(node.const_value)
        elif isinstance(node, qlang.Bit) and kind == BitType.UINT:
            text = f"{format_node(node.parent_uint)}[{node.index}]"
        elif isinstance(node, qlang.UInt) and kind == UIntType.BITS:
            text = "bits(" + ", ".join(format_node(bit) for bit in node.bits) + ")"
        elif kind == BitType.NOT and isinstance(node, qlang.Bit):
            text = f"(~{format_node(node.operands[0])})"
        elif len(node.operands) == 2:
            left, right = (format_node(child) for child in node.operands)
            text = f"({left} {operators.get(kind.name, kind.name)} {right})"
        else:
            text = kind.name + "(" + ", ".join(format_node(child) for child in node.children) + ")"
        cache[key] = text
        return text
    result = format_node(root)
    return result[1:-1] if result.startswith("(") and result.endswith(")") else result


def abbreviated_label(text, limit=30):
    if len(text) <= limit:
        return text
    # Retain both ends; the complete expression is always in the SVG tooltip.
    left = (limit - 1) // 2
    return text[:left] + "…" + text[-(limit - 1 - left):]


def layout_graph(root, node_width=130):
    """Layered DAG layout with a reserved slot for every skipped-layer edge."""
    nodes, edges = graph_data(root)
    by_id = {id(n): n for n in nodes}
    depths = {id(root): 0}
    pending = [id(root)]
    while pending:
        parent = pending.pop()
        for child in by_id[parent].children:
            key = id(child)
            proposed = depths[parent] + 1
            if proposed > depths.get(key, -1):
                depths[key] = proposed
                pending.append(key)
    layers = defaultdict(list)
    for node in nodes:
        layers[depths[id(node)]].append(id(node))
    chains = []
    predecessors, successors = defaultdict(list), defaultdict(list)
    for index, (parent, child) in enumerate(edges):
        chain = [parent]
        for depth in range(depths[parent] + 1, depths[child]):
            dummy = ("route", index, depth)
            layers[depth].append(dummy)
            chain.append(dummy)
        chain.append(child)
        chains.append(chain)
        for a, b in zip(chain, chain[1:]):
            successors[a].append(b)
            predecessors[b].append(a)
    # Barycenter sweeps reduce edge crossings, while invisible routing slots
    # keep long connections clear of actual node boxes.
    for _ in range(4):
        for descending in (True, False):
            ranks = range(1, max(layers)+1) if descending else range(max(layers)-1, -1, -1)
            neighbors = predecessors if descending else successors
            for depth in ranks:
                order = {key: (i + .5) / len(layer)
                         for layer in layers.values() for i, key in enumerate(layer)}
                layers[depth].sort(key=lambda key: sum(order[n] for n in neighbors[key]) / len(neighbors[key])
                                   if neighbors[key] else order[key])
    width = max(640, max(map(len, layers.values())) * (node_width + 50) + 40)
    height = (max(layers) + 1) * 150 + 35
    positions = {key: ((i + .5) * width / len(layer), 45 + depth * 150)
                 for depth, layer in layers.items() for i, key in enumerate(layer)}
    outgoing_edges, incoming_edges = defaultdict(list), defaultdict(list)
    gap_segments = defaultdict(list)
    for index, (parent, child) in enumerate(edges):
        outgoing_edges[parent].append(index)
        incoming_edges[child].append(index)
        for a, b in zip(chains[index], chains[index][1:]):
            gap_segments[positions[a][1]].append((index, a, b))

    def port(index, indices):
        # Distinct attachment points: no shared vertical stems at either end.
        return 88 * ((indices.index(index) + .5) / len(indices) - .5)

    routes = []
    for index, chain in enumerate(chains):
        points = []
        for i, (a, b) in enumerate(zip(chain, chain[1:])):
            x1, y1 = positions[a]
            x2, y2 = positions[b]
            if i == 0:
                x1 += port(index, outgoing_edges[a])
            if i == len(chain)-2:
                x2 += port(index, incoming_edges[b])
            start_y = y1 + 28 if i == 0 else y1
            end_y = y2 - 28 if i == len(chain)-2 else y2
            segments = gap_segments[y1]
            lane = segments.index((index, a, b))
            # Every connection has its own horizontal track in this gap.
            mid_y = y1 + 51 + 48 * (lane + .5) / len(segments)
            if not points:
                points.append((x1, start_y))
            points.extend([(x1, mid_y), (x2, mid_y), (x2, end_y)])
        routes.append(points)
    return nodes, edges, positions, routes, width, height


def render_graph(root, full_expressions=False):
    node_width = 240 if full_expressions else 130
    nodes, edges, positions, routes, width, height = layout_graph(root, node_width=node_width)
    incoming = defaultdict(int)
    for _, child in edges:
        incoming[child] += 1
    palette = ["#2563eb", "#c2410c", "#15803d", "#9333ea", "#be185d", "#0e7490", "#a16207", "#475569"]
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" role="img" aria-label="qlang expression graph" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="#f8f7f2"/>', '<defs>']
    for index in range(len(edges)):
        color = palette[index % len(palette)]
        parts.append(f'<marker id="arrow-{index}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M 0 0 L 10 5 L 0 10 Z" fill="{color}"/></marker>')
    parts.append('</defs>')
    child_index = defaultdict(int)
    edge_labels = []
    by_id = {id(node): node for node in nodes}
    for index, ((parent, child), points) in enumerate(zip(edges, routes)):
        operand = child_index[parent]
        child_index[parent] += 1
        role = ("L" if operand == 0 else "R") if len(by_id[parent].children) == 2 else str(operand + 1)
        color = palette[index % len(palette)]
        description = html.escape(f'{node_text(by_id[parent])[0]}: operand {operand+1} → {node_text(by_id[child])[0]}')
        path = "M" + " L".join(f"{x},{y}" for x, y in points)
        # A background halo makes crossings look like crossings, not junctions.
        parts.append(f'<g><title>{description}</title><path d="{path}" fill="none" stroke="#f8f7f2" stroke-width="7" stroke-linejoin="round"/>')
        parts.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2.5" stroke-linejoin="round" marker-end="url(#arrow-{index})"/></g>')
        x, y = points[0]
        edge_labels.append(f'<g><title>{description}</title><circle cx="{x}" cy="{y+11}" r="9" fill="#f8f7f2" stroke="{color}"/><text x="{x}" y="{y+15}" text-anchor="middle" font-family="system-ui,sans-serif" font-size="10" font-weight="700" fill="{color}">{role}</text></g>')
    parts.extend(edge_labels)
    colors = {"variable": ("#dbeafe", "#1d4ed8"), "constant": ("#e2e8f0", "#334155"),
              "operation": ("#ede9fe", "#6d28d9")}
    for node in nodes:
        key = id(node)
        x, y = positions[key]
        title, subtitle, category = node_text(node)
        complete = expression_text(node)
        if full_expressions:
            title = abbreviated_label(complete)
        fill, stroke = colors[category]
        shared = incoming[key] > 1
        tooltip = html.escape(f"{complete}: {subtitle}" + (f" · shared by {incoming[key]} edges" if shared else ""))
        parts.append(f'<g><title>{tooltip}</title><rect x="{x-node_width/2}" y="{y-28}" width="{node_width}" height="56" rx="12" fill="{fill}" stroke="{stroke}" stroke-width="{3 if shared else 1}"/>')
        font_size = 12 if full_expressions else 18
        font_family = "ui-monospace,monospace" if full_expressions else "system-ui,sans-serif"
        parts.append(f'<text x="{x}" y="{y-3}" text-anchor="middle" font-family="{font_family}" font-size="{font_size}" font-weight="650" fill="{stroke}">{html.escape(str(title))}</text>')
        parts.append(f'<text x="{x}" y="{y+17}" text-anchor="middle" font-family="system-ui,sans-serif" font-size="11" fill="#475569">{html.escape(subtitle)}</text></g>')
    parts.append('</svg>')
    markup = '<style>.qlang-explorer, .qlang-explorer div, .qlang-explorer input, .qlang-explorer select, .qlang-explorer button {background-color:#f8f7f2;}</style><p style="color:#475569;font-size:12px">Each colored arrow connects one parent to one child. L = left operand; R = right operand. Crossings are not junctions.</p><div style="overflow:auto;max-height:760px;border:1px solid #e2e8f0;border-radius:12px">' + ''.join(parts) + '</div>'
    return markup, len(nodes), len(edges), sum(v > 1 for v in incoming.values())

def make_explorer():
    # Static graph exports do not need a notebook or widget installation.
    import ipywidgets as widgets

    expression = widgets.Text(value="(x + y == 7) & (x > y)", placeholder="Type an expression…",
                              continuous_update=True, layout=widgets.Layout(width="100%"))
    optimized = widgets.Checkbox(value=False, description="Show optimized graph", indent=False)
    full_labels = widgets.Checkbox(value=True, description="Full expressions", indent=False,
                                  tooltip="Switch between operator symbols and complete expressions. Hover abbreviated labels to read the full expression.")
    presets = widgets.Dropdown(description="Examples", options=[
        ("Arithmetic constraint", "(x + y == 7) & (x > y)"),
        ("Shared subexpression", "((x + y) == 7) | ((x + y) == 5)"),
        ("Simplification", "((x + 0) == y) & (p | ~p)"),
        ("Boolean logic", "(p & q) | (~p & ~q)"),
        ("Unsigned subtraction", "(x - y) >= 2"),
    ], layout=widgets.Layout(width="100%"))
    rows = []
    changing_table = False
    table_body = widgets.VBox(layout=widgets.Layout(gap="6px"))
    table_message = widgets.HTML()
    status = widgets.HTML()
    graph = widgets.HTML()
    def refresh(change=None):
        started = time.perf_counter()
        try:
            declarations = [dict(name=row["name"].value, type=row["type"].value,
                                 width=row["width"].value) for row in rows]
            root = parse_expression(expression.value, declarations=declarations)
            if optimized.value:
                root = qlang.optimize_expr(root)
            svg, n, e, shared = render_graph(root, full_expressions=full_labels.value)
            graph.value = svg
            elapsed = (time.perf_counter() - started) * 1000
            status.value = (f'<p style="color:#475569">{n} nodes · {e} edges · {shared} shared nodes · '
                            f'{elapsed:.1f} ms parse/render · ' + ('optimized' if optimized.value else 'original') + '</p>')
        except Exception as exc:
            message = f"Incomplete expression: {exc.msg}" if isinstance(exc, SyntaxError) else str(exc)
            status.value = '<p style="color:#b45309">' + html.escape(message) + ' · Showing the last valid graph.</p>'
    def table_changed(change=None):
        if not changing_table:
            refresh()

    def add_row(name, kind="UInt", bits=3):
        name_field = widgets.Text(value=name, placeholder="name", continuous_update=True,
                                  layout=widgets.Layout(width="110px"))
        kind_field = widgets.Dropdown(options=[("UInt", "UInt"), ("Boolean", "Bit")], value=kind,
                                     layout=widgets.Layout(width="95px"))
        bit_field = widgets.BoundedIntText(value=bits, min=1, max=32, disabled=kind == "Bit",
                                          layout=widgets.Layout(width="65px"))
        remove = widgets.Button(description="×", tooltip="Remove variable",
                                layout=widgets.Layout(width="34px"))
        row = {"name": name_field, "type": kind_field, "width": bit_field}
        view = widgets.HBox([name_field, kind_field, bit_field, remove])
        row["view"] = view
        rows.append(row)
        table_body.children = tuple(r["view"] for r in rows)
        def type_changed(change):
            # Suspend refresh until type and width form one consistent declaration.
            nonlocal changing_table
            changing_table = True
            try:
                bit_field.disabled = change["new"] == "Bit"
                if bit_field.disabled:
                    bit_field.value = 1
            finally:
                changing_table = False
            refresh()
        def remove_row(_):
            rows.remove(row)
            table_body.children = tuple(r["view"] for r in rows)
            for control in (name_field, kind_field, bit_field, remove, view):
                control.close()
            refresh()
        name_field.observe(table_changed, names="value")
        bit_field.observe(table_changed, names="value")
        kind_field.observe(type_changed, names="value")
        remove.on_click(remove_row)

    def reset_variables(_=None):
        nonlocal changing_table
        changing_table = True
        try:
            for row in rows:
                for control in row.values():
                    control.close()
            rows.clear()
            for declaration in DEFAULT_DECLARATIONS:
                add_row(declaration["name"], declaration["type"], declaration["width"])
            table_message.value = ""
        finally:
            changing_table = False
        refresh()

    def add_variable(_):
        if len(rows) >= 32:
            table_message.value = "At most 32 variables are supported."
            return
        existing = {row["name"].value for row in rows}
        index = 1
        while f"v{index}" in existing:
            index += 1
        add_row(f"v{index}")
        table_message.value = ""
        refresh()

    add_button = widgets.Button(description="+ Add variable", layout=widgets.Layout(width="150px"))
    reset_button = widgets.Button(description="Reset defaults", layout=widgets.Layout(width="150px"))
    add_button.on_click(add_variable)
    reset_button.on_click(reset_variables)
    reset_variables()
    def choose(change):
        # Presets reference x/y/p/q; restore their declarations when selected.
        reset_variables()
        expression.value = change["new"]
    for control in (expression, optimized, full_labels):
        control.observe(refresh, names="value")
    presets.observe(choose, names="value")
    connection = widgets.Button(description="Test widget connection", layout=widgets.Layout(width="210px"))
    connection_status = widgets.HTML(value="Click the button to confirm frontend ↔ kernel communication.")
    def confirm_connection(_):
        connection_status.value = "<b style='color:#15803d'>Connected: browser events reach Python.</b>"
    connection.on_click(confirm_connection)
    refresh()
    headers = widgets.HBox([
        widgets.HTML("<b>Name</b>", layout=widgets.Layout(width="110px")),
        widgets.HTML("<b>Type</b>", layout=widgets.Layout(width="95px")),
        widgets.HTML("<b>Bits</b>", layout=widgets.Layout(width="65px")),
    ])
    variable_panel = widgets.VBox([
        widgets.HTML("<h3 style='margin:0'>Variables</h3><p>Edit names, types, and individual widths.</p>"),
        headers, table_body, widgets.HBox([add_button, reset_button]), table_message,
        widgets.HTML("<small>Names must be unique Python identifiers. Renaming does not rewrite the expression. "
                     "Selecting an example restores x, y, p, q.</small>"),
    ], layout=widgets.Layout(width="350px", min_width="350px", padding="12px", border="1px solid #cbd5e1"))
    canvas = widgets.VBox([presets, expression, widgets.HBox([optimized, full_labels]), status, graph, connection, connection_status],
                         layout=widgets.Layout(flex="1 1 380px", min_width="300px"))
    return widgets.HBox([canvas, variable_panel],
                        layout=widgets.Layout(width="100%", align_items="flex-start", gap="16px", flex_flow="row wrap")).add_class("qlang-explorer")



def show_explorer():
    """Build a fresh view, plus a widget-independent initial graph preview."""
    from IPython.display import display, HTML

    explorer = make_explorer()
    print("Live AST v2: editable variable table and routed edges. Controls appear below.")
    display(explorer)
    preview = render_graph(parse_expression("(x + y == 7) & (x > y)"))[0]
    display(HTML("<details><summary>Static graph preview (no widget renderer needed)</summary>"
                 + preview + "</details>"))
    return explorer
