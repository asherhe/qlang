from dataclasses import dataclass
from enum import Enum, IntEnum, auto
from typing import Any, TypeVar, cast

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

class UIntType(IntEnum):
    # leaf nodes
    CONST = 1
    VAR = 2

    # operators
    ADD = auto()
    SUB = auto()
    MUL = auto()
    LSHIFT = auto()
    RSHIFT = auto()


T = TypeVar("T", bound="ASTNode")

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

    @classmethod
    def _get_cache_key(cls, **kwargs) -> tuple[Any, ...]:
        """
        unique identifie for this ASTNode, implemented by subclassed types
        """
        raise NotImplementedError

    @classmethod
    def _create(cls: type[T], **kwargs) -> T:
        sub_key = cls._get_cache_key(**kwargs)
        key = (cls, sub_key)
        
        if key not in cls.cache:
            cls.cache[key] = cls(**kwargs)
        return cast(T, cls.cache[key])

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
    # used for validation if this belongs to a UInt
    parent_width: int | None

    def __init__(
        self, 
        *, 
        bit_type: BitType,
        name: str | None = None, 
        value: bool | None = None, 
        index: int | None = None,
        parent_width: int | None = None,
        operands: tuple[ASTNode, ...] = (),
    ): 
        self.bit_type = bit_type
        self.name = name
        self.value = value
        self.index = index
        self.parent_width = parent_width
        self.operands = operands

    @classmethod
    def _get_cache_key(
        cls,
        *,
        bit_type: BitType,
        name: str | None = None, 
        value: bool | None = None, 
        index: int | None = None,
        parent_width: int | None = None,
        operands: tuple[ASTNode, ...] = (),
        **kwargs
   ) -> tuple[Any, ...]:
        if bit_type == BitType.CONST:
            return (bit_type, value)
        elif bit_type == BitType.VAR:
            return (bit_type, name, index, parent_width)
        else:
            operand_ids = tuple(id(o) for o in operands)
            return (bit_type, operands)

    @staticmethod
    def as_bit(val): 
        if isinstance(val, Bit): 
            return val
        if isinstance(val, (bool, int)): 
            return const_bit(val)
        raise TypeError(f"Type {type(val)} cannot be converted to Bit")

    @staticmethod
    def _operator(bit_type: BitType, **kwargs) -> Bit:
        left = Bit.as_bit(kwargs['self'])
        if 'other' not in kwargs: 
            return Bit._create(bit_type=bit_type, operands=(left,))
        
        right = Bit.as_bit(kwargs['other'])
        
        return Bit._create(bit_type=bit_type, operands=(left, right))
    
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

    @classmethod
    def _get_cache_key(
        cls,
        *,
        uint_type: UIntType, 
        width: int,
        name: str | None = None,
        bits: tuple[Bit, ...] = (),
        operands: tuple[ASTNode, ...] = (),
        **kwargs
   ) -> tuple[Any, ...]:
        if uint_type == UIntType.CONST:
            bit_ids = tuple(id(b) for b in bits)
            return (uint_type, width, bit_ids)
        elif uint_type == UIntType.VAR:
            return (uint_type, width, name)
        else:
            operand_ids = tuple(id(o) for o in operands)
            return (uint_type, width, operand_ids)

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
    def _operator(uint_type: UIntType, **kwargs): 
        left = UInt.as_uint(kwargs['self'])
        if 'other' not in kwargs: 
            return UInt._create(uint_type=uint_type, width=left.width, operands=(left,))
        
        right = UInt.as_uint(kwargs['other'])
        
        if uint_type == UIntType.ADD: 
            width = max(left.width, right.width) + 1
        elif uint_type == UIntType.SUB:
            width = max(left.width, right.width)               # fixed-width, wraps modulo 2**width
        elif uint_type == UIntType.MUL:
            width = left.width + right.width
        elif uint_type in (UIntType.LSHIFT, UIntType.RSHIFT): 
            width = left.width                                 # this assumes fixed-width behavior for now
        else: 
            raise ValueError(f"unsupported UInt operation: {uint_type}")
        
        return UInt._create(uint_type=uint_type, width=width, operands=(left, right))

    def _compare(self, bit_type: BitType, other) -> Bit:
        right = UInt.as_uint(other)
        return Bit._create(bit_type=bit_type, operands=(self, right))

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
    return Bit._create(
        bit_type=BitType.CONST,
        value=value,
    )

def var_bit(name: str, index: int | None = None, parent_width: int | None = None) -> Bit:
    """
    create a named variable Bit node. optional index and parent width fields for Bits that are part of a UInt
    """
    if parent_width: 
        if parent_width <= 0: 
            raise ValueError("specified parent width must be positive")
        if not index: 
            raise ValueError("index must be present if parent width is present")
        if index < 0 or index >= parent_width: 
            raise ValueError("index is out of bounds compared to parent width") 
    return Bit._create(
        bit_type=BitType.VAR,
        name=name,
        index=index,
        parent_width=parent_width,
    )

def const_uint(value: int, width: int | None = None) -> UInt:
    """create a constant UInt of CONST Bits. if width is not given, infer the smallest width that fits val"""
    if value < 0: 
        raise ValueError("value must be non-negative")
    if width is None:
        width = max(1, value.bit_length())
    if width <= 0: 
        raise ValueError("specified width must be positive")
    if value >= (1 << width): 
        raise ValueError("value does not fit in specified width")
    
    bits = tuple(const_bit((value >> i) & 1) for i in range(width))
    return UInt._create(
        uint_type=UIntType.CONST,
        width=width,
        bits=bits,
    )

def var_uint(name: str, width: int) -> UInt:
    """create a variable UInt"""
    if width <= 0: 
        raise ValueError("specified width must be positive")
    bits = tuple(var_bit(name, i, parent_width=width) for i in range(width))
    return UInt._create(
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

