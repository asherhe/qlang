from .expression import ASTNode, BitType, UIntType, Bit, UInt, bits_uint, const_bit, const_uint, node_type
from collections import defaultdict
from operator import add, attrgetter, mul
from typing import Any, Callable, cast

# ASTNode -> ASTNode transformation
NodeTransform = Callable[[ASTNode], ASTNode]
# some node type
NodeType = BitType | UIntType

# a rule receives the node and its operands unpacked, and returns a replacement or None if it does not apply
RuleFn = Callable[..., ASTNode | None]

class RulePass:
    """
    a named group of rewrite rules, indexed by the node type they apply to

    register rules with the .rule() decorator. rules for a node type are tried in registration order

    the table is keyed by (enum class, member) rather than the member alone, because members of
    different IntEnums with the same value compare and hash equal and would share a table entry
    """
    def __init__(self, name: str):
        self.name = name
        self.table: dict[tuple[type[NodeType], NodeType], list[RuleFn]] = defaultdict(list)

    def rule(self, *types: NodeType) -> Callable[[RuleFn], RuleFn]:
        """register the decorated function as a rule for the given node types"""
        def deco(fn: RuleFn) -> RuleFn:
            for t in types:
                self.table[(type(t), t)].append(fn)
            return fn
        return deco

    def rules_for(self, t: NodeType) -> list[RuleFn]:
        """the rules registered for a node type, in registration order"""
        return self.table.get((type(t), t), [])

def rebuild(expr: ASTNode, operands: tuple[ASTNode, ...]) -> ASTNode:
    """reconstruct an expression with new operands"""
    # nodes are hash-consed, so unchanged operands mean the node is already canonical
    if all(old is new for old, new in zip(expr.operands, operands)):
        return expr
    if isinstance(expr, Bit):
        return Bit._create(bit_type=expr.bit_type, operands=operands)
    if isinstance(expr, UInt):
        return UInt._create(uint_type=expr.uint_type, width=expr.width, operands=operands)
    raise TypeError("only Bit and UInt nodes are supported")

def traverse_transform(
    expr: ASTNode,
    rule: NodeTransform,
    memo: dict[int, ASTNode] | None = None,
) -> ASTNode:
    """
    apply transformation rule to every ASTNode of the expression DAG, traversing it bottom-up

    the rule receives a non-leaf node whose operands have already been processed by the rule and returns its replacement.

    execution results are memoized to avoid reprocessing repeat ASTNodes
    """
    if expr.is_leaf:
        return expr

    if memo is None:
        memo = {}
    key = id(expr)
    if key not in memo:
        operands = tuple(traverse_transform(op, rule, memo) for op in expr.operands)
        memo[key] = rule(rebuild(expr, operands))
    return memo[key]

def run_rule_pass(expr: ASTNode, rule_pass: RulePass) -> ASTNode:
    """apply a rule pass to every node of the expression, bottom-up. the first rule that applies to a node wins"""
    def apply_pass(node: ASTNode) -> ASTNode:
        for rule in rule_pass.rules_for(node_type(node)):
            res = rule(node, *node.operands)
            if res is not None:
                return res
        return node
    return traverse_transform(expr, apply_pass)


# decompose operations that cannot be directly built with quantum gates into primitive operators
decompose = RulePass("decompose")

@decompose.rule(BitType.EQ)
def deco_EQ(expr: Bit, a: UInt, b: UInt):
    """!= is simpler to implement in gates"""
    return ~(a != b)

# collapse all comparison operators to LT to keep compilation clean
@decompose.rule(BitType.GT)
def deco_GT(expr: Bit, a: UInt, b: UInt):
    return b < a
@decompose.rule(BitType.LE)
def deco_LE(expr: Bit, a: UInt, b: UInt):
    return ~(b < a)
@decompose.rule(BitType.GE)
def deco_GE(expr: Bit, a: UInt, b: UInt):
    return ~(a < b)


# shifting by a constant is only a rewiring of bits, so it needs no gates
# a shift of a constant is left to the collapse pass, which folds it directly

@decompose.rule(UIntType.LSHIFT)
def deco_LSHIFT_const(expr: UInt, a: UInt, b: UInt):
    """a << c  ->  the bits of a moved up by c, filled with 0 and truncated to the shift's width"""
    if b.uint_type == UIntType.CONST and a.uint_type != UIntType.CONST:
        c = b.const_value
        if c == 0:
            return a
        if c >= expr.width:
            return const_uint(0, expr.width)
        bits = (const_bit(0),) * c + a.bits
        return bits_uint(bits[:expr.width])

@decompose.rule(UIntType.RSHIFT)
def deco_RSHIFT_const(expr: UInt, a: UInt, b: UInt):
    """a >> c  ->  the bits of a moved down by c, dropping the low bits and filling the top with 0"""
    if b.uint_type == UIntType.CONST and a.uint_type != UIntType.CONST:
        c = b.const_value
        if c == 0:
            return a
        if c >= a.width:
            return const_uint(0, expr.width)
        return bits_uint(a.bits[c:] + (const_bit(0),) * c)


# eliminate superfluous expression nodes that are no-ops or can be simplified
collapse = RulePass("collapse")

def _sorted_bits(a: Bit, b: Bit) -> tuple[Bit, Bit]:
    """order the operands of a commutative operator by type, placing CONST and VAR at the front"""
    first, second = sorted((a, b), key=attrgetter("bit_type"))
    return first, second

def _sorted_uints(a: UInt, b: UInt) -> tuple[UInt, UInt]:
    """order the operands of a commutative operator by type, placing CONST and VAR at the front"""
    first, second = sorted((a, b), key=attrgetter("uint_type"))
    return first, second

def _is_negation_pair(a: Bit, b: Bit) -> bool:
    """whether one of a, b is the NOT of the other"""
    return (a.bit_type == BitType.NOT and a.operands[0] is b
            or b.bit_type == BitType.NOT and b.operands[0] is a)

# ---- NOT ----

@collapse.rule(BitType.NOT)
def coll_NOT_const(expr: Bit, a: Bit):
    """fixed values"""
    if a.bit_type == BitType.CONST:
        return const_bit(not a.value)

@collapse.rule(BitType.NOT)
def coll_NOT_NOT(expr: Bit, a: Bit):
    """~~a  ->  a"""
    if a.bit_type == BitType.NOT:
        return a.operands[0]

# ---- comparisons ----

@collapse.rule(BitType.EQ)
def coll_EQ(expr: Bit, a: UInt, b: UInt):
    # skip - assume all EQ have been decomposed
    pass

@collapse.rule(BitType.NE)
def coll_NE_self(expr: Bit, a: UInt, b: UInt):
    """a != a  ->  false"""
    if a is b:
        return const_bit(False)

@collapse.rule(BitType.NE)
def coll_NE_const_fold(expr: Bit, a: UInt, b: UInt):
    if a.uint_type == b.uint_type == UIntType.CONST:
        return const_bit(a.const_value != b.const_value)
    if isinstance(a, Bit) and isinstance(b, Bit): 
        return a ^ b

@collapse.rule(BitType.NE)
def coll_NE_const_bounds(expr: Bit, a: UInt, b: UInt):
    """a != c  ->  1 when c is too large for a to ever equal it, in either operand order"""
    if b.uint_type == UIntType.CONST and b.const_value >= (1 << a.width):
        return const_bit(True)
    if a.uint_type == UIntType.CONST and a.const_value >= (1 << b.width):
        return const_bit(True)

@collapse.rule(BitType.LT)
def coll_LT_self(expr: Bit, a: UInt, b: UInt):
    """a < a  ->  0"""
    if a is b:
        return const_bit(False)

@collapse.rule(BitType.LT)
def coll_LT_const_fold(expr: Bit, a: UInt, b: UInt):
    """direct evaluation of const"""
    if a.uint_type == b.uint_type == UIntType.CONST:
        return const_bit(a.const_value < b.const_value)
    if isinstance(a, Bit) and isinstance(b, Bit): 
        return (~a) & b

@collapse.rule(BitType.LT)
def coll_LT_const_bounds(expr: Bit, a: UInt, b: UInt):
    """
    comparison against a constant that is out of range of the other operand
    a < b  ->  0       if a >= max b size
    0 < b  ->  b != 0  cheaper than a comparison
    (max-1) < b  ->  b == max  only max is above max-1, written as ~(b != max)
    a < 0  ->  0
    a < b  ->  1       if a > max b size
    a < max  ->  a != max  only max is not below max
    a < 1    ->  a == 0    only 0 is below 1, written as ~(a != 0)
    """
    if a.uint_type == UIntType.CONST:
        aval = a.const_value
        bmax = (1 << b.width) - 1
        if aval >= bmax:
            return const_bit(False)
        if aval == 0:
            return b != 0
        if aval == bmax - 1:
            return ~(b != bmax)
    if b.uint_type == UIntType.CONST:
        bval = b.const_value
        amax = (1 << a.width) - 1
        if bval == 0:
            return const_bit(False)
        if bval > amax:
            return const_bit(True)
        # checked before the a < 1 case so that a 1-bit a (where max is 1) gets the NOT-free form
        if bval == amax:
            return a != bval
        if bval == 1:
            return ~(a != 0)

@collapse.rule(BitType.GT)
def coll_GT(expr: Bit, a: UInt, b: UInt):
    # skip - assume all GT have been decomposed
    pass

@collapse.rule(BitType.LE)
def coll_LE(expr: Bit, a: UInt, b: UInt):
    # skip - assume all LE have been decomposed
    pass

@collapse.rule(BitType.GE)
def coll_GE(expr: Bit, a: UInt, b: UInt):
    # skip - assume all GE have been decomposed
    pass

# ---- commutative (Bit, Bit) -> Bit operators ----

@collapse.rule(BitType.XOR)
def coll_XOR_const_fold(expr: Bit, a: Bit, b: Bit):
    """direct evaluation of const"""
    a, b = _sorted_bits(a, b)
    if a.bit_type == b.bit_type == BitType.CONST:
        return const_bit(cast(bool, a.value) ^ cast(bool, b.value))

@collapse.rule(BitType.XOR)
def coll_XOR_const_identity(expr: Bit, a: Bit, b: Bit):
    """
    0 ^ b  ->  b
    1 ^ b  ->  ~b
    """
    a, b = _sorted_bits(a, b)
    if a.bit_type == BitType.CONST:
        return ~b if a.value else b

@collapse.rule(BitType.XOR)
def coll_XOR_self(expr: Bit, a: Bit, b: Bit):
    """a ^ a  ->  0"""
    if a is b:
        return const_bit(False)

@collapse.rule(BitType.XOR)
def coll_XOR_negation(expr: Bit, a: Bit, b: Bit):
    """a ^ ~a  ->  1"""
    if _is_negation_pair(a, b):
        return const_bit(True)

@collapse.rule(BitType.AND)
def coll_AND_const_fold(expr: Bit, a: Bit, b: Bit):
    """direct evaluation of const"""
    a, b = _sorted_bits(a, b)
    if a.bit_type == b.bit_type == BitType.CONST:
        return const_bit(cast(bool, a.value) & cast(bool, b.value))

@collapse.rule(BitType.AND)
def coll_AND_const_identity(expr: Bit, a: Bit, b: Bit):
    """
    0 & b  ->  0
    1 & b  ->  b
    """
    a, b = _sorted_bits(a, b)
    if a.bit_type == BitType.CONST:
        return b if a.value else const_bit(False)

@collapse.rule(BitType.AND)
def coll_AND_self(expr: Bit, a: Bit, b: Bit):
    """a & a  ->  a"""
    if a is b:
        return a

@collapse.rule(BitType.AND)
def coll_AND_negation(expr: Bit, a: Bit, b: Bit):
    """a & ~a  ->  0"""
    if _is_negation_pair(a, b):
        return const_bit(False)

@collapse.rule(BitType.OR)
def coll_OR(expr: Bit, a: Bit, b: Bit):
    # skip - assume all OR have been decomposed
    pass

# ---- UInt operators ----

def _const_reassoc(kind: UIntType, op: Callable[[Any, Any], Any]) -> RuleFn:
    """
    build a rule that merges the constants of a chain of an associative, commutative UInt operator
    (x op c1) op c2  ->  x op (c1 op c2)

    op is used both to combine the constant values (ints) and to build the new node (UInts), so the
    new node gets the width the operator would normally give it
    """
    def rule(expr: UInt, a: UInt, b: UInt):
        c2, inner = _sorted_uints(a, b)
        if c2.uint_type == UIntType.CONST and inner.uint_type == kind:
            c1, x = _sorted_uints(*cast(tuple[UInt, UInt], inner.operands))
            if c1.uint_type == UIntType.CONST:
                # size the merged constant so the rebuilt node keeps the width the parent relies on
                if kind == UIntType.ADD:
                    width = max(c1.width, c2.width) + 1
                else:  # MUL
                    width = c1.width + c2.width
                return op(x, const_uint(op(c1.const_value, c2.const_value), width))
    rule.__name__ = f"coll_{kind.name}_const_reassoc"
    return rule

@collapse.rule(UIntType.ADD)
def coll_ADD_const_fold(expr: UInt, a: UInt, b: UInt):
    a, b = _sorted_uints(a, b)
    if a.uint_type == b.uint_type == UIntType.CONST:
        # the ADD node's width carries the carry bit; a parent may depend on it
        return const_uint(a.const_value + b.const_value, max(a.width, b.width) + 1)

@collapse.rule(UIntType.ADD)
def coll_ADD_const_identity(expr: UInt, a: UInt, b: UInt):
    a, b = _sorted_uints(a, b)
    if a.uint_type == UIntType.CONST and a.const_value == 0:
        return b

@collapse.rule(UIntType.ADD)
def coll_ADD_self(expr: UInt, a: UInt, b: UInt):
    if a is b:
        return bits_uint((const_bit(0),) + a.bits)

# (x + c1) + c2  ->  x + (c1 + c2)
coll_ADD_const_reassoc = collapse.rule(UIntType.ADD)(_const_reassoc(UIntType.ADD, add))

@collapse.rule(UIntType.SUB)
def coll_SUB_const_fold(expr: UInt, a: UInt, b: UInt):
    if a.uint_type == b.uint_type == UIntType.CONST:
        diff = a.const_value - b.const_value
        width = expr.width
        diff = (a.const_value - b.const_value) & ((1 << width) - 1)
        return const_uint(diff, width)

@collapse.rule(UIntType.SUB)
def coll_SUB_const_identity(expr: UInt, a: UInt, b: UInt):
    # TODO: 0 - b identity if we implement negation operator
    if b.uint_type == UIntType.CONST and b.const_value == 0:
        return a

@collapse.rule(UIntType.SUB)
def coll_SUB_self(expr: UInt, a: UInt, b: UInt):
    if a is b:
        return const_uint(0, expr.width)

@collapse.rule(UIntType.SUB)
def coll_SUB_add_cancel(expr: UInt, a: UInt, b: UInt):
    """
    (x + y) - y  ->  x
    (x + y) - x  ->  y
    the sum is wide enough to never wrap, so subtracting an addend gives the other one back exactly
    """
    if a.uint_type == UIntType.ADD:
        x, y = cast(tuple[UInt, UInt], a.operands)
        if y is b:
            return x
        if x is b:
            return y

@collapse.rule(UIntType.MUL)
def coll_MUL_const_fold(expr: UInt, a: UInt, b: UInt):
    a, b = _sorted_uints(a, b)
    if a.uint_type == b.uint_type == UIntType.CONST:
        # the MUL node's width is the full product width; a parent may depend on it
        return const_uint(a.const_value * b.const_value, a.width + b.width)

@collapse.rule(UIntType.MUL)
def coll_MUL_const_identity(expr: UInt, a: UInt, b: UInt):
    """
    0 * b  ->  0
    1 * b  ->  b
    """
    a, b = _sorted_uints(a, b)
    if a.uint_type == UIntType.CONST:
        if a.const_value == 0:
            return const_uint(0, expr.width)
        if a.const_value == 1:
            return b

# (x * c1) * c2  ->  x * (c1 * c2)
# registered before coll_MUL_pow2 so that (x * 3) * 4 becomes x * 12 instead of a multiply and a shift
coll_MUL_const_reassoc = collapse.rule(UIntType.MUL)(_const_reassoc(UIntType.MUL, mul))

@collapse.rule(UIntType.MUL)
def coll_MUL_pow2(expr: UInt, a: UInt, b: UInt):
    """
    2^k * b  ->  b << k
    """
    a, b = _sorted_uints(a, b)
    if a.uint_type == UIntType.CONST:
        c = a.const_value
        # 0 and 1 already handled by coll_MUL_const_identity
        if c > 1 and c & (c - 1) == 0:
            k = c.bit_length() - 1
            bits = (const_bit(0),) * k + b.bits
            # pad to size
            bits += (const_bit(0),) * (expr.width - len(bits))
            return bits_uint(bits)

@collapse.rule(UIntType.LSHIFT)
def coll_LSHIFT_const_fold(expr: UInt, a: UInt, b: UInt):
    if a.uint_type == b.uint_type == UIntType.CONST:
        mask = (1 << a.width) - 1
        return const_uint((a.const_value << b.const_value) & mask, a.width)

@collapse.rule(UIntType.LSHIFT)
def coll_LSHIFT_const_identity(expr: UInt, a: UInt, b: UInt):
    """
    a << 0  ->  a
    0 << b  ->  0
    """
    if b.uint_type == UIntType.CONST and b.const_value == 0:
        return a
    if a.uint_type == UIntType.CONST and a.const_value == 0:
        return const_uint(0, expr.width)

@collapse.rule(UIntType.LSHIFT)
def coll_LSHIFT_overshift(expr: UInt, a: UInt, b: UInt):
    """a << c  ->  0 when c >= width"""
    if b.uint_type == UIntType.CONST and b.const_value >= expr.width:
        return const_uint(0, expr.width)

@collapse.rule(UIntType.LSHIFT)
def coll_LSHIFT_merge(expr: UInt, a: UInt, b: UInt):
    """(x << c1) << c2  ->  x << (c1 + c2)"""
    # expr.width <= a.width: bits the inner shift truncated would also be truncated by the outer width
    if a.uint_type == UIntType.LSHIFT and b.uint_type == UIntType.CONST and expr.width <= a.width:
        x, c1 = cast(tuple[UInt, UInt], a.operands)
        if c1.uint_type == UIntType.CONST:
            total = c1.const_value + b.const_value
            if total >= expr.width:
                return const_uint(0, expr.width)
            return x << total

@collapse.rule(UIntType.RSHIFT)
def coll_RSHIFT_const_fold(expr: UInt, a: UInt, b: UInt):
    if a.uint_type == b.uint_type == UIntType.CONST:
        # the RSHIFT node keeps the shifted value's width; a parent may depend on it
        return const_uint(a.const_value >> b.const_value, a.width)

@collapse.rule(UIntType.RSHIFT)
def coll_RSHIFT_const_identity(expr: UInt, a: UInt, b: UInt):
    """
    a >> 0  ->  a
    0 >> b  ->  0
    """
    if b.uint_type == UIntType.CONST and b.const_value == 0:
        return a
    if a.uint_type == UIntType.CONST and a.const_value == 0:
        return const_uint(0, expr.width)

@collapse.rule(UIntType.RSHIFT)
def coll_RSHIFT_overshift(expr: UInt, a: UInt, b: UInt):
    """a >> c  ->  0 when c >= width"""
    if b.uint_type == UIntType.CONST and b.const_value >= a.width:
        return const_uint(0, expr.width)

@collapse.rule(UIntType.RSHIFT)
def coll_RSHIFT_merge(expr: UInt, a: UInt, b: UInt):
    """(x >> c1) >> c2  ->  x >> (c1 + c2)"""
    if a.uint_type == UIntType.RSHIFT and b.uint_type == UIntType.CONST:
        x, c1 = cast(tuple[UInt, UInt], a.operands)
        if c1.uint_type == UIntType.CONST:
            total = c1.const_value + b.const_value
            if total >= x.width:
                return const_uint(0, expr.width)
            return x >> total


# ---- BITS ----

@collapse.rule(UIntType.BITS)
def coll_BITS_nested(expr: UInt):
    """
    route the bits that are taken from a BITS back to the bits it was built from
    bits_uint(..., y[i], ...)  ->  bits_uint(..., y.bits[i], ...)  when y is a BITS
    """
    routed = tuple(
        cast(UInt, bit.parent_uint).bits[cast(int, bit.index)]
        if bit.bit_type == BitType.UINT and cast(UInt, bit.parent_uint).uint_type == UIntType.BITS
        else bit
        for bit in expr.bits
    )
    if any(old is not new for old, new in zip(expr.bits, routed)):
        return bits_uint(routed)


def optimize_expr(expr: ASTNode) -> ASTNode:
    """
    apply simplifications to minimize the complexity and gate requirements of the expression tree
    """
    key = id(expr)
    while True:
        expr = run_rule_pass(expr, decompose)
        expr = run_rule_pass(expr, collapse)
        new_key = id(expr)
        if key == new_key:
            return expr
        key = new_key

