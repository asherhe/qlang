from __future__ import annotations
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
    # TODO: add more as you see fit. 

class UIntType(Enum):
    # leaf nodes
    CONST = auto()
    VAR = auto()

    # operators
    ADD = auto()
    AND = auto()
    # TODO: these are just some examples. add more as you see fit.

ASTNode = Union["Bit", "UInt"]

class Bit:
    """
    a Bit is a node in an expression tree. it can be evaluated to some value, depending on its bit_type:
    - CONST : a constant bit, evaluates to 0 or 1
    - VAR   : represents a bit of unknown state
    - other : a Bit created through an operation on other Bits

    do NOT call the Bit constructor directly, use const_bit() or make_var_bit()
    """
    name: str | None
    bit_type: BitType

    # if a const bit, the value of the bit
    value: bool | None

    # if an operation, the list of operands it operates on
    operands: tuple[ASTNode, ...]

    # TODO: implement all the required bitwise operators
    def __init__(
        self, 
        *, 
        name: str | None, 
        bit_type: BitType, 
        value: bool | None, 
        operands: tuple[ASTNode, ...] = (),
    ): 
        self.name = name
        self.bit_type = bit_type
        self.value = value
        self.operands = operands

    def __and__(self, other): 
        if isinstance(other, bool): 
            other = const_bit(other)
        if other.bit_type is BitType.VAR: 
            return NotImplemented
        return Bit(
            name=self.name, 
            bit_type=BitType.AND, 
            value=None,
            operands=(self, other),
        )

    def __rand__(self, other): 
        return self.__and__(other)

    def __not__(self): 
        return Bit(
            name=self.name, 
            bit_type=BitType.NOT, 
            value=None, 
            operands=(self,),
        )

    def __xor__(self, other): 
        if isinstance(other, bool): 
            other = const_bit(other)
        if other.bit_type is BitType.VAR: 
            return NotImplemented
        return Bit(
            name=self.name, 
            bit_type=BitType.XOR, 
            value=None, 
            operands=(self, other),
        )

    def __rxor__(self, other): 
        return self.__xor__(other)

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
        name: str | None, 
        uint_type: UIntType, 
        width: int, 
        bits: tuple[Bit, ...] = (),
        operands: tuple[ASTNode, ...] = (),
    ): 
        self.name = name
        self.uint_type=uint_type
        self.width = width
        self.bits = bits
        self.operands = operands
    
    # TODO: implement all the arithmetic, bitwise comparison operators    
    def __add__(self, other):
        if isinstance(other, int): 
            other = const_uint(other, width=None)
        if other.uint_type is UIntType.VAR: 
            return NotImplemented # TODO
        return UInt(
            name=self.name, 
            uint_type=UIntType.ADD, 
            width=max(self.width, other.width) + 1, 
            operands=(self, other),
        )

    def __radd__(self, other): 
        return self.__add__(other)

# -------- factory functions --------

def const_bit(val: int | bool) -> Bit:
    """create a constant Bit node with a given value"""
    ...

def var_bit(name: str, index: int | None = None) -> Bit:
    """
    create a named variable Bit node. optional index field for Bits that are part of a UInt
    """
    ...

def const_uint(val: int, width: int | None) -> UInt:
    """create a constant UInt of CONST Bits"""
    ...

def var_uint(name: str, width: int) -> UInt:
    """create a variable UInt"""
    ...

def uint(val: int | str, width: int | None = None) -> UInt:
    """
    general factory for UInts, delegate to const_uint or var_uint depending on the type of val
    """
    ...

# -------- expression optimization --------

def optimize_expr(expr: ASTNode) -> ASTNode:
    """
    apply simplifications to minimize the number of nodes in an expression tree.
    """
    ...

