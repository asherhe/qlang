from dataclasses import dataclass
from enum import Enum, IntEnum, auto
from operator import attrgetter
from typing import Any, cast

# -------- AST node types --------

class BitType(IntEnum):
    # leaf nodes
    CONST = 1
    VAR = 2

    # operators
    NOT = auto()
    XOR = auto()
    AND = auto()
    OR = auto()
    EQ = auto()
    NE = auto()
    LT = auto()
    GT = auto()
    LE = auto()
    GE = auto()

class UIntType(Enum):
    # leaf nodes
    CONST = 1
    VAR = 2

    # operators
    ADD = auto()
    SUB = auto()
    MUL = auto()
    LSHIFT = auto()
    RSHIFT = auto()


class ASTNode: 
    name: str | None

    # if an operation, the list of operands it operates on
    operands: tuple[ASTNode, ...]

    # stores ASTNodes for fast lookup
    cache: dict[tuple[Any, ...], ASTNode] = {}

    # whether this ASTNode has no children
    @property
    def is_leaf(self) -> bool:
        return len(self.operands) == 0

    def __bool__(self): 
        raise TypeError(
            "AST nodes cannot be used as Python booleans. "
            "Use bitwise operators to build expressions "
            "or evaluate the expression explicitly"
        )

class Bit(ASTNode):
    """
    a Bit is a node in an expression tree. it can be evaluated to some value, depending on its bit_type:
    - CONST : a constant bit, evaluates to 0 or 1
    - VAR   : represents a bit of unknown state
    - other : a Bit created through an operation on other Bits

    do NOT call the Bit constructor directly, use const_bit() or make_var_bit()
    """
    bit_type: BitType

    # if a const bit, the value of the bit
    value: bool | None
    # if a variable bit, maps back to the Bit's index in the containing UInt if it exists
    index: int | None

    def __init__(
        self, 
        *, 
        bit_type: BitType,
        name: str | None = None,
        index: int | None = None,
        value: bool | None = None, 
        operands: tuple[ASTNode, ...] = (),
    ): 
        self.bit_type = bit_type
        self.name = name
        self.index = index
        self.value = value
        self.operands = operands

    @staticmethod
    def as_bit(val): 
        if isinstance(val, Bit): 
            return val
        if isinstance(val, (bool, int)): 
            return const_bit(val)
        raise TypeError(f"Type {type(val)} cannot be converted to Bit")

    @staticmethod
    def cache_bit(bit_type: BitType, operands: tuple[ASTNode, ...]) -> Bit:
        key = (Bit, bit_type, tuple(id(node) for node in operands))
        if key not in ASTNode.cache:
            ASTNode.cache[key] = Bit(
                bit_type=bit_type,
                operands=operands,
            )
        return cast(Bit, ASTNode.cache[key])

    @staticmethod
    def _operator(bit_type: BitType, **kwargs) -> Bit:
        left = Bit.as_bit(kwargs['self'])
        if 'other' not in kwargs: 
            return Bit.cache_bit(bit_type, (left,))
        
        right = Bit.as_bit(kwargs['other'])
        
        return Bit.cache_bit(bit_type, (left, right))
    
    def __and__(self, other): 
        return Bit._operator(BitType.AND, self=self, other=other)

    def __rand__(self, other): 
        return self.__and__(other)

    def __invert__(self): 
        return Bit._operator(BitType.NOT, self=self)

    def __xor__(self, other): 
        return Bit._operator(BitType.XOR, self=self, other=other)

    def __rxor__(self, other): 
        return self.__xor__(other)

    def __or__(self, other): 
        return Bit._operator(BitType.OR, self=self, other=other)

    def __ror__(self, other): 
        return self.__or__(other)

    def __eq__(self, other) -> Bit:  # type: ignore[override]
        return Bit._operator(BitType.EQ, self=self, other=other)

    def __ne__(self, other) -> Bit:  # type: ignore[override]
        return Bit._operator(BitType.NE, self=self, other=other)

    def __gt__(self, other): 
        return Bit._operator(BitType.GT, self=self, other=other)

    def __lt__(self, other): 
        return Bit._operator(BitType.LT, self=self, other=other)

    def __ge__(self, other): 
        return Bit._operator(BitType.GE, self=self, other=other)

    def __le__(self, other): 
        return Bit._operator(BitType.LE, self=self, other=other)

class UInt(ASTNode):
    """
    a wrapper for a sequence of Bits that together represent an integer
    
    supports bitwise, arithmetic, and comparison operations

    do NOT call the UInt constructor directly, use uint()
    """

    uint_type: UIntType
    width: int # consider making this a property based on bits or operands

    # if a const, Bits of this UInt with LSB as bits[0]
    bits: tuple[Bit, ...]

    def __init__(
        self, 
        *, 
        uint_type: UIntType, 
        width: int,
        name: str | None = None,
        bits: tuple[Bit, ...] = (),
        operands: tuple[ASTNode, ...] = (),
    ): 
        self.uint_type=uint_type
        self.width = width
        self.name = name
        self.bits = bits
        self.operands = operands

    @property
    def const_value(self) -> int:
        """the internal value held by the bits of a CONST UInt, as an int"""
        if self.uint_type != UIntType.CONST:
            raise ValueError("must be a constant UInt to convert to int")
        return sum((1 if bit.value else 0) << i for i, bit in enumerate(self.bits))

    @staticmethod
    def as_uint(val): 
        if isinstance(val, UInt): 
            return val
        if isinstance(val, (bool, int)): 
            return const_uint(val, width=None)
        raise TypeError(f"Type {type(val)} cannot be converted to UInt")
    
    @staticmethod
    def cache_int(uint_type: UIntType, width: int, operands) -> UInt:
        key = (UIntType, uint_type, width, tuple(id(node) for node in operands))
        if key not in ASTNode.cache: 
            ASTNode.cache[key] = UInt(
                uint_type=uint_type,
                width=width, 
                operands=operands,
            )
        return cast(UInt, ASTNode.cache[key])
    @staticmethod
    def _operator(uint_type: UIntType, **kwargs): 
        left = UInt.as_uint(kwargs['self'])
        if 'other' not in kwargs: 
            return UInt.cache_int(uint_type, left.width, (left,))
        
        right = UInt.as_uint(kwargs['other'])
        
        if uint_type == UIntType.ADD: 
            width = max(left.width, right.width) + 1
        elif uint_type == UIntType.MUL: 
            width = left.width + right.width
        elif uint_type in (UIntType.LSHIFT, UIntType.RSHIFT): 
            width = left.width                                 # this assumes fixed-width behavior for now
        else: 
            raise ValueError(f"unsupported UInt operation: {uint_type}")
        
        return UInt.cache_int(uint_type, width, (left, right))

    def _compare(self, bit_type: BitType, other) -> Bit:
        right = UInt.as_uint(other)
        return Bit.cache_bit(bit_type, (self, right))

    # python really isn't happy about overriding __eq__ to return a non-bool
    def __eq__(self, other) -> Bit:  # type: ignore[override]
        return self._compare(BitType.EQ, other)

    def __ne__(self, other) -> Bit:  # type: ignore[override]
        return self._compare(BitType.NE, other)
    
    def __gt__(self, other): 
        return self._compare(BitType.GT, other)

    def __lt__(self, other): 
        return self._compare(BitType.LT, other)

    def __ge__(self, other): 
        return self._compare(BitType.GE, other)

    def __le__(self, other): 
        return self._compare(BitType.LE, other)    

    def __add__(self, other):
        return UInt._operator(UIntType.ADD, self=self, other=other)

    def __radd__(self, other): 
        return self.__add__(other)
    
    def __sub__(self, other):
        return UInt._operator(UIntType.SUB, self=self, other=other)

    def __rsub__(self, other): 
        return UInt._operator(UIntType.SUB, self=other, other=self)

    def __mul__(self, other): 
        return UInt._operator(UIntType.MUL, self=self, other=other)

    def __rmul__(self, other): 
        return self.__mul__(other)

    def __lshift__(self, other): 
        return UInt._operator(UIntType.LSHIFT, self=self, other=other)

    def __rlshift__(self, other): 
        return UInt._operator(UIntType.LSHIFT, self=other, other=self)

    def __rshift__(self, other): 
        return UInt._operator(UIntType.RSHIFT, self=self, other=other)

    def __rrshift__(self, other): 
        return UInt._operator(UIntType.RSHIFT, self=other, other=self)

# -------- factory functions --------

def const_bit(value: int | bool) -> Bit:
    """create a constant Bit node with a given value"""

    if isinstance(value, int):
        value = value != 0
    return Bit(
        bit_type=BitType.CONST,
        value=value,
    )

def var_bit(name: str, index: int | None = None) -> Bit:
    """
    create a named variable Bit node. optional index field for Bits that are part of a UInt
    """
    return Bit(
        bit_type=BitType.VAR,
        name=name,
        index=index,
    )

def const_uint(value: int, width: int | None) -> UInt:
    """create a constant UInt of CONST Bits. if width is not given, infer the smallest width that fits val"""
    if width is None:
        width = max(1, value.bit_length())
    
    bits = tuple(const_bit((value >> i) & 1) for i in range(width))
    return UInt(
        uint_type=UIntType.CONST,
        width=width,
        bits=bits,
    )

def var_uint(name: str, width: int) -> UInt:
    """create a variable UInt"""
    bits = tuple(var_bit(name, i) for i in range(width))
    return UInt(
        uint_type=UIntType.VAR,
        width=width,
        name=name,
        bits=bits,
    )

def uint(val: int | str, width: int | None = None) -> UInt:
    """
    general factory for UInts, delegate to const_uint or var_uint depending on the type of val
    """
    if isinstance(val, int):
        return const_uint(val, width)
    else:
        if width is None:
            raise ValueError(f"uint({val}): variable UInts require a width")
        return var_uint(val, width)

# -------- expression optimization --------

def decompose_ops(expr: ASTNode) -> ASTNode:
    """
    decompose operations that cannot be directly built with quantum gates into primitive operators
    """
    if expr.is_leaf:
        return expr

    operands = tuple(map(decompose_ops, expr.operands))
    bit_args = cast(tuple[Bit, ...], operands)
    uint_args = cast(tuple[UInt, ...], operands)

    if isinstance(expr, Bit):
        expr = Bit.cache_bit(expr.bit_type, operands)
    elif isinstance(expr, UInt):
        expr = UInt.cache_int(expr.uint_type, expr.width, operands)
    else:
        raise TypeError("only decomposition of Bit and UInt nodes are supported")

    if isinstance(expr, Bit):
        if expr.bit_type == BitType.OR:
            # decompose into AND, NOT by de morgan's laws

            # assuming only binary OR operation
            # maybe we will do n-ary OR eventually but not now
            expr = ~(~bit_args[0] & ~bit_args[1])
        elif expr.bit_type == BitType.EQ:
            # != is simpler to implement in gates
            expr = ~(uint_args[0] != uint_args[1])
        # collapse all comparison operators to LT to keep compilation clean
        elif expr.bit_type == BitType.GT:
            expr = uint_args[1] < uint_args[0]
        elif expr.bit_type == BitType.LE:
            expr = ~(uint_args[1] < uint_args[0])
        elif expr.bit_type == BitType.GE:
            expr = ~(uint_args[0] < uint_args[1])
            
    elif isinstance(expr, UInt):
        pass
    
    return expr

def collapse_identities(expr: ASTNode) -> ASTNode:
    """
    eliminate superfluous expression nodes that are no-ops or can be simplified
    """
    if expr.is_leaf:
        return expr

    operands = tuple(map(collapse_identities, expr.operands))
    bit_args = cast(tuple[Bit, ...], operands)
    uint_args = cast(tuple[UInt, ...], operands)

    if isinstance(expr, Bit):
        expr = Bit.cache_bit(expr.bit_type, operands)
    elif isinstance(expr, UInt):
        expr = UInt.cache_int(expr.uint_type, expr.width, operands)
    else:
        raise TypeError("only collapse of Bit and UInt nodes are supported")

    if isinstance(expr, Bit):
        if expr.bit_type == BitType.NOT:
            # fixed values
            if bit_args[0].bit_type == BitType.CONST:
                expr = const_bit(not bit_args[0].value)
            elif bit_args[0].bit_type == BitType.NOT:
                expr = bit_args[0].operands[0]

        elif expr.bit_type == BitType.EQ:
            # skip - assume all EQ have been decomposed
            pass
        elif expr.bit_type == BitType.NE:
            # a != a  ->  false
            if uint_args[0] is uint_args[1]:
                expr = const_bit(False)

        elif expr.bit_type == BitType.LT:
            a, b = uint_args
            atype = a.uint_type
            btype = b.uint_type
            if atype == UIntType.CONST:
                if btype == UIntType.CONST:
                    expr = const_bit(a.const_value < b.const_value)
            elif btype == UIntType.CONST:
                bval = b.const_value
                if bval == 0:
                    expr = const_bit(False)
                elif bval >= (1 << a.width):
                    expr = const_bit(True)

        elif expr.bit_type == BitType.GT:
            # skip - assume all GT have been decomposed
            pass
        elif expr.bit_type == BitType.LE:
            # skip - assume all LE have been decomposed
            pass
        elif expr.bit_type == BitType.GE:
            # skip - assume all GE have been decomposed
            pass
        else:
            # now all commutative (Bit, Bit) -> Bit operators

            # sort arguments by type
            # places CONST and VAR at the front
            sorted_bits = sorted(bit_args, key=attrgetter("bit_type"))
            a, b = sorted_bits
            atype = a.bit_type
            btype = b.bit_type
            aval = cast(bool, a.value)
            bval = cast(bool, b.value)

            # both constants
            ab_const = atype == btype == BitType.CONST

            # a is ~b
            a_xor_b = (atype == BitType.NOT and a.operands[0] is b
                      or btype == BitType.NOT and b.operands[0] is a)
            if expr.bit_type == BitType.XOR:
                if ab_const:
                    # direct evaluation of const
                    expr = const_bit(aval ^ bval)
                elif atype == BitType.CONST:
                    # 0 ^ b  ->  b
                    # 1 ^ b  ->  ~b
                    expr = ~b if aval else b
                elif a is b:
                    # a ^ a  ->  0
                    expr = const_bit(False)
                elif a_xor_b:
                    # a ^ ~a  ->  1
                    expr = const_bit(True)

            elif expr.bit_type == BitType.AND:
                if ab_const:
                    # direct evaluation of const
                    expr = const_bit(aval & bval)
                elif atype == BitType.CONST:
                    # 0 & b  ->  0
                    # 1 & b  ->  b
                    expr = b if aval else const_bit(False)
                elif a is b:
                    # a & a  ->  1
                    expr = const_bit(True)
                elif a_xor_b:
                    # a & ~a  ->  0
                    expr = const_bit(False)

            elif expr.bit_type == BitType.OR:
                # skip - assume all OR have been decomposed
                pass
            
    elif isinstance(expr, UInt):
        if expr.uint_type == UIntType.ADD:
            pass
        elif expr.uint_type == UIntType.SUB:
            pass
        elif expr.uint_type == UIntType.MUL:
            pass
        elif expr.uint_type == UIntType.LSHIFT:
            pass
        elif expr.uint_type == UIntType.RSHIFT:
            pass
    
    return expr

def optimize_expr(expr: ASTNode) -> ASTNode:
    """
    apply simplifications to minimize the complexity and gate requirements of the expression tree
    """
    expr = decompose_ops(expr)
    expr = collapse_identities(expr)
    return expr

