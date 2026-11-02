from enum import Enum, auto

class BitType(Enum):
    # leaf nodes
    CONST = auto()
    VAR = auto()

    # operators
    NOT = auto()
    XOR = auto()
    AND = auto()

class Bit:
    """
    a Bit is a node in an expression tree. it can be evaluated to some value, depending on its bit_type:
    - CONST : a constant bit, evaluates to 0 or 1
    - VAR   : represents a bit of unknown state
    - other : a Bit created through an operation on other Bits

    do NOT call the Bit constructor directly, use const_bit() or make_var_bit()
    """
    name: str
    bit_type: BitType

    # type-specific fields TODO
    # i recommend using a single `payload` field to store the required data across all bit_types,
    # and add @property accessors to check bit_type before returning property as the right type.

    # TODO: implement all the required bitwise operators

class UInt:
    """
    a wrapper for a sequence of Bits that together represent an integer
    
    supports bitwise, arithmetic, and comparison operations

    do NOT call the UInt constructor directly, use uint()
    """

    # Bits of this UInt with LSB as bits[0]
    bits: tuple[Bit, ...]

    @property
    def width(self) -> int:
        return len(self.bits)

    # TODO: implement all the arithmetic, bitwise comparison operators

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

