from expression import ASTNode, Bit, UInt

from collections.abc import Iterable

def search(
    params: dict[str, Iterable[int]],
    pred: Bit,
) -> list[dict[str, int]]:
    """
    search for solutions in the space of params that satisfy pred

    the search domain is the cartesian product of each parameter's search space specified in params.

    return a list of all solution sets that satisfy the predicate
    """
    ...

