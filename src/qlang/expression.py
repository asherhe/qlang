from enum import Enum, auto
from typing import Union

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

    # TODO: implement all the required bitwise operators
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

    def __and__(self, other): 
        if isinstance(other, bool): 
            other = const_bit(other)
        return Bit(
            name=None, 
            bit_type=BitType.AND, 
            value=None,
            operands=(self, other),
        )

    def __rand__(self, other): 
        return self.__and__(other)

    def __not__(self): 
        return Bit(
            name=None, 
            bit_type=BitType.NOT, 
            value=None, 
            operands=(self,),
        )

    def __xor__(self, other): 
        if isinstance(other, bool): 
            other = const_bit(other)
        return Bit(
            name=None, 
            bit_type=BitType.XOR, 
            value=None, 
            operands=(self, other),
        )

    def __rxor__(self, other): 
        return self.__xor__(other)

    def __or__(self, other): 
        if isinstance(other, bool): 
            other = const_bit(other)
        return Bit(
            name=None, 
            bit_type=BitType.OR, 
            value=None, 
            operands=(self, other),
        )

    def __ror__(self, other): 
        return self.__or__(other)

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
    
    # TODO: implement all the arithmetic, bitwise comparison operators    
    def __add__(self, other):
        if isinstance(other, int): 
            other = const_uint(other, width=None)
        return UInt(
            name=None, 
            uint_type=UIntType.ADD, 
            width=max(self.width, other.width) + 1, 
            operands=(self, other),
        )

    def __radd__(self, other): 
        return self.__add__(other)

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

