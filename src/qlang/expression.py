from enum import Enum, auto
from typing import Union, Any
import hashlib

# -------- AST node types --------

class BitType(Enum):
    # leaf nodes
    CONST = auto()
    VAR = auto()

    # operators
    NOT = auto()
    XOR = auto()
    AND = auto()
    OR = auto()
    EQ = auto()
    NE = auto()
    GT = auto()
    LT = auto()
    GE = auto()
    LE = auto()

class UIntType(Enum):
    # leaf nodes
    CONST = auto()
    VAR = auto()

    # operators
    ADD = auto()
    MUL = auto()
    LSHIFT = auto()
    RSHIFT = auto()

ASTNode = Union["Bit", "UInt"]

class Bit:
    """
    a Bit is a node in an expression tree. it can be evaluated to some value, depending on its bit_type:
    - CONST : a constant bit, evaluates to 0 or 1
    - VAR   : represents a bit of unknown state
    - other : a Bit created through an operation on other Bits

    do NOT call the Bit constructor directly, use const_bit() or make_var_bit()
    """
    bit_type: BitType

    name: str | None

    # if a const bit, the value of the bit
    value: bool | None
    # if a variable bit, maps back to the Bit's index in the containing UInt if it exists
    index: int | None
    
    # if an operation, the list of operands it operates on
    operands: tuple[ASTNode, ...]

    cache: dict[tuple[Any, ...], ASTNode] = {}

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
    def _operator(type: BitType, **kwargs): 
        key = (type, kwargs)
        if key not in Bit.cache: 
            if 'other' in kwargs: 
                if isinstance(kwargs['other'], bool):
                    kwargs['other'] = const_bit(kwargs['other'])
                Bit.cache[key] = Bit(
                    bit_type=type, 
                    operands=(kwargs['self'], kwargs['other'])
                )
            else: 
                Bit.cache[key] = Bit(
                    bit_type=type, 
                    operands=(kwargs['self'])
                )
        return Bit.cache[key]
    
    def __and__(self, other): 
        return Bit._operator(BitType.AND, self=self, other=other)

    def __rand__(self, other): 
        return self.__and__(other)

    def __not__(self): 
        return Bit._operator(BitType.NOT, self=self)

    def __xor__(self, other): 
        return Bit._operator(BitType.XOR, self=self, other=other)

    def __rxor__(self, other): 
        return self.__xor__(other)

    def __or__(self, other): 
        return Bit._operator(BitType.OR, self=self, other=other)

    def __ror__(self, other): 
        return self.__or__(other)

    def __eq__(self, other) -> Any: 
        return Bit._operator(BitType.EQ, self=self, other=other)

    def __ne__(self, other) -> Any: 
        return Bit._operator(BitType.NE, self=self, other=other)

    def __gt__(self, other): 
        return Bit._operator(BitType.GT, self=self, other=other)

    def __lt__(self, other): 
        return Bit._operator(BitType.LT, self=self, other=other)

    def __ge__(self, other): 
        return Bit._operator(BitType.GE, self=self, other=other)

    def __le__(self, other): 
        return Bit._operator(BitType.LE, self=self, other=other)

class UInt:
    """
    a wrapper for a sequence of Bits that together represent an integer
    
    supports bitwise, arithmetic, and comparison operations

    do NOT call the UInt constructor directly, use uint()
    """

    name: str | None
    uint_type: UIntType
    width: int # consider making this a property based on bits or operands

    # if a const, Bits of this UInt with LSB as bits[0]
    bits: tuple[Bit, ...]

    # if an operator, the operands that this UInt operates on
    operands: tuple[ASTNode, ...]

    cache: dict[tuple[Any, ...], ASTNode] = {}

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

    @staticmethod
    def _operator(type: UIntType, **kwargs): 
        key = (type, kwargs)
        if key not in UInt.cache: 
            if 'other' in kwargs: 
                if isinstance(kwargs['other'], int):
                    kwargs['other'] = const_uint(kwargs['other'], width=None)
                UInt.cache[key] = UInt(
                    uint_type=type, 
                    width=kwargs['width'],
                    operands=(kwargs['self'], kwargs['other']),
                )
            else: 
                UInt.cache[key] = UInt(
                    uint_type=type, 
                    width=kwargs['width'],
                    operands=(kwargs['self']),
                )
        return UInt.cache[key]
    
    def __add__(self, other):
        return UInt._operator(UIntType.ADD, self=self, other=other)

    def __radd__(self, other): 
        return self.__add__(other)

    def __mul__(self, other): 
        return UInt._operator(UIntType.MUL, self=self, other=other)

    def __rmul__(self, other): 
        return self.__mul__(other)

    def __lshift__(self, other): 
        return UInt._operator(UIntType.LSHIFT, self=self, other=other)

    def __rlshift__(self, other): 
        return self.__lshift__(other)

    def __rshift__(self, other): 
        return UInt._operator(UIntType.RSHIFT, self=self, other=other)

    def __rrshift__(self, other): 
        return self.__rshift__(other)

# -------- factory functions --------

def const_bit(value: int | bool) -> Bit:
    """create a constant Bit node with a given value"""

    if isinstance(value, int):
        value = value == 0
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
        v = value
        width = 1
        while v > 1:
            width += 1
            v >>= 1

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

def optimize_expr(expr: ASTNode) -> ASTNode:
    """
    apply simplifications to minimize the complexity and gate requirements of the expression tree
    """
    ...

