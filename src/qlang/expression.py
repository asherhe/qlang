from enum import Enum, auto
from typing import Any

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


class ASTNode: 
    name: str | None

    # if an operation, the list of operands it operates on
    operands: tuple[ASTNode, ...]

    # stores ASTNodes for fast lookup
    cache: dict[tuple[Any, ...], ASTNode] = {}

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
        return NotImplemented

    @staticmethod
    def cache_bit(type: BitType, operands: tuple[ASTNode, ...]): 
        key = (Bit, type, tuple(id(node) for node in operands))
        if key not in ASTNode.cache: 
            ASTNode.cache[key] = Bit(
                bit_type=type, 
                operands=operands,
            )
        return ASTNode.cache[key]

    @staticmethod
    def _operator(type: BitType, **kwargs): 
        left = Bit.as_bit(kwargs['self'])
        if left is NotImplemented: 
            return NotImplemented
        if 'other' not in kwargs: 
            return Bit.cache_bit(type, (left,))
        
        right = Bit.as_bit(kwargs['other'])
        if right is NotImplemented: 
            return NotImplemented
        
        return Bit.cache_bit(type, (left, right))
    
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

    @staticmethod
    def as_uint(val): 
        if isinstance(val, UInt): 
            return val
        if isinstance(val, (bool, int)): 
            return const_uint(val, width=None)
        return NotImplemented
    
    @staticmethod
    def cache_int(type: UIntType, width: int, operands): 
        key = (UIntType, type, tuple(id(node) for node in operands))
        if key not in ASTNode.cache: 
            ASTNode.cache[key] = UInt(
                uint_type=type,
                width=width, 
                operands=operands,
            )
        return ASTNode.cache[key]

    @staticmethod
    def _operator(type: UIntType, **kwargs): 
        left = UInt.as_uint(kwargs['self'])
        if left is NotImplemented: 
            return NotImplemented
        if 'other' not in kwargs: 
            return UInt.cache_int(type, left.width, (left,))
        
        right = UInt.as_uint(kwargs['other'])
        if right is NotImplemented: 
            return NotImplemented
        
        if type == UIntType.ADD: 
            width = max(left.width, right.width) + 1
        elif type == UIntType.MUL: 
            width = left.width + right.width
        elif type in (UIntType.LSHIFT, UIntType.RSHIFT): 
            width = left.width                                 # this assumes fixed-width behavior for now
        else: 
            raise ValueError(f"unsupported UInt operation: {type}")
        
        return UInt.cache_int(type, width, (left, right))

    def _compare(self, type: BitType, other): 
        right = UInt.as_uint(other)
        if right is NotImplemented: 
            return NotImplemented
        return Bit.cache_bit(type, (self, right))

    def __eq__(self, other): 
        return self._compare(BitType.EQ, other)

    def __ne__(self, other): 
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

def optimize_expr(expr: ASTNode) -> ASTNode:
    """
    apply simplifications to minimize the complexity and gate requirements of the expression tree
    """
    ...

