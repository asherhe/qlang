from .expression import ASTNode, BitType, UIntType, Bit, UInt, const_bit, const_uint
from collections import defaultdict
from operator import attrgetter
from typing import Callable, cast

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

def node_type(expr: ASTNode) -> NodeType:
    if isinstance(expr, Bit):
        return expr.bit_type
    elif isinstance(expr, UInt):
        return expr.uint_type
    raise TypeError("only ASTNodes of type Bit and UInt are supported")

def rebuild(expr: ASTNode, operands: tuple[ASTNode, ...]) -> ASTNode:
    """reconstruct an expression with new operands"""
    # nodes are hash-consed, so unchanged operands mean the node is already canonical
    if all(old is new for old, new in zip(expr.operands, operands)):
        return expr
    if isinstance(expr, Bit):
        return Bit.cache_bit(expr.bit_type, operands)
    if isinstance(expr, UInt):
        return UInt.cache_int(expr.uint_type, expr.width, operands)
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

@decompose.rule(BitType.OR)
def deco_OR(expr: Bit, a: Bit, b: Bit):
    """
    decompose into AND, NOT by de morgan's laws

    assuming only binary OR operation
    maybe we will do n-ary OR eventually but not now
    """
    return ~(~a & ~b)

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

@collapse.rule(BitType.LT)
def coll_LT_const_bounds(expr: Bit, a: UInt, b: UInt):
    """comparison against a constant that is out of range of the other operand"""
    if b.uint_type == UIntType.CONST:
        bval = b.const_value
        if bval == 0:
            # a < 0  ->  0
            return const_bit(False)
        if bval >= (1 << a.width):
            # a < (anything above the max of a)  ->  1
            return const_bit(True)

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

@collapse.rule(UIntType.ADD)
def coll_ADD_const_fold(expr: UInt, a: UInt, b: UInt):
    a, b = _sorted_uints(a, b)
    if a.uint_type == b.uint_type == UIntType.CONST:
        return const_uint(a.const_value + b.const_value)

@collapse.rule(UIntType.ADD)
def coll_ADD_self(expr: UInt, a: UInt, b: UInt):
    if a is b:
        # TODO: uint width of this is cooked, carry bit is lost
        return a << 1

@collapse.rule(UIntType.SUB)
def coll_SUB_const_fold(expr: UInt, a: UInt, b: UInt):
    if a.uint_type == b.uint_type == UIntType.CONST:
        diff = a.const_value << b.const_value
        width = max(a.width, b.width)
        if (diff < 0):
            # convert to two's complement
            diff += 1 << width
        return const_uint(diff, width)

@collapse.rule(UIntType.SUB)
def coll_SUB_self(expr: UInt, a: UInt, b: UInt):
    if a is b:
        return const_uint(0, a.width)

@collapse.rule(UIntType.MUL)
def coll_MUL_const_fold(expr: UInt, a: UInt, b: UInt):
    a, b = _sorted_uints(a, b)
    if a.uint_type == b.uint_type == UIntType.CONST:
        return const_uint(a.const_value * b.const_value)

@collapse.rule(UIntType.LSHIFT)
def coll_LSHIFT_const_fold(expr: UInt, a: UInt, b: UInt):
    if a.uint_type == b.uint_type == UIntType.CONST:
        return const_uint(a.const_value << b.const_value, a.width)

@collapse.rule(UIntType.RSHIFT)
def coll_RSHIFT_const_fold(expr: UInt, a: UInt, b: UInt):
    if a.uint_type == b.uint_type == UIntType.CONST:
        return const_uint(a.const_value >> b.const_value)


def optimize_expr(expr: ASTNode) -> ASTNode:
    """
    apply simplifications to minimize the complexity and gate requirements of the expression tree
    """
    expr = run_rule_pass(expr, decompose)
    expr = run_rule_pass(expr, collapse)
    return expr

