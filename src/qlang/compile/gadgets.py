from qiskit.circuit import QuantumCircuit

def quantumAND(compute: QuantumCircuit,qubitA: int, qubitB: int, target_qubit: int) -> None:
    if qubitA == qubitB:
        compute.cx(qubitA, target_qubit)
    else:
        compute.ccx(qubitA, qubitB, target_qubit)


def quantumOR(compute: QuantumCircuit, qubitA: int, qubitB: int, target_qubit: int) -> None:
    if qubitA == qubitB:
        compute.cx(qubitA, target_qubit)
    else:
        compute.cx(qubitA, target_qubit)
        compute.cx(qubitB, target_qubit)
        compute.ccx(qubitA, qubitB, target_qubit)

def quantumXOR(compute: QuantumCircuit, qubitA: int, qubitB: int, target_qubit: int) -> None:
    compute.cx(qubitA, target_qubit)
    compute.cx(qubitB, target_qubit)

def quantumNOT(compute: QuantumCircuit, qubitA: int, target_qubit: int) -> None:
    compute.x(target_qubit)
    compute.cx(qubitA, target_qubit)

def quantumNE(compute: QuantumCircuit, left, right, target, fresh_scratch):
    if len(left) == 1:
        quantumXOR(compute, left[0], right[0], target)
        return

    differences = []
    for a, b in zip(left, right, strict=True):
        difference = fresh_scratch()
        quantumXOR(compute, a, b, difference)
        differences.append(difference)

    any_different = differences[0]
    for i, difference in enumerate(differences[1:], start=1):
        result = target if i == len(differences) - 1 else fresh_scratch()
        quantumOR(compute, any_different, difference, result)
        any_different = result

def quantumLT(compute: QuantumCircuit, left, right, target, fresh_scratch):
    # BUG: cooked when size of left and right mismatch
    lower_less = None

    for i, (a, b) in enumerate(zip(left, right, strict=True)):
        result = target if i == len(left) - 1 else fresh_scratch()
        not_a = fresh_scratch()
        quantumNOT(compute, a, not_a)

        if lower_less is None:
            quantumAND(compute, not_a, b, result)
        else:
            less_here = fresh_scratch()
            different = fresh_scratch()
            equal_here = fresh_scratch()
            carry_less = fresh_scratch()

            quantumAND(compute, not_a, b, less_here)
            quantumXOR(compute, a, b, different)
            quantumNOT(compute, different, equal_here)
            quantumAND(compute, equal_here, lower_less, carry_less)
            quantumOR(compute, less_here, carry_less, result)

        lower_less = result
    
def quantumADD(compute: QuantumCircuit, a: list[int], b: list[int], out: list[int], fresh_scratch, carry_in=None, carry_out=None):
    """
    construct a quantum uint addition circuit

    PARAMETERS
    ----------
    a, b       : list of qubits, LSB-first. can be any length
    out        : list of qubits, LSB-last. length is the max of a and b
    carry_in   : qubit for carry input
    carry_out  : qubit for carry output
    """

    # uses cuccaro adder, 1 ancila cubit required
    # TODO: draper adder if in-place allowed
    
    # normalize to len(a) >= len(b)
    if len(a) < len(b):
        tmp = a
        a = b
        b = tmp
    assert(len(a) == len(out))

    # TODO: use carry_in for ancilla if it is not used anywhere else
    carry_ancilla = fresh_scratch()
    # copy carry_in state
    if carry_in is not None:
        compute.cx(carry_in, carry_ancilla)

    # copy b onto out
    for i in range(len(b)):
        compute.cx(b[i], out[i])

    # forward sweep
    for i in range(len(b)):
        compute.cx(carry_ancilla, out[i])
        compute.cx(carry_ancilla, a[i])
        compute.ccx(a[i], out[i], carry_ancilla)
        
    # forward sweep where b is 0
    # simplifies to a toffoli gate
    for i in range(len(b), len(a)):
        compute.cx(carry_ancilla, a[i])

    # copy carry_out state
    if carry_out is not None:
        compute.cx(carry_ancilla, carry_out)

    # forward sweep
    for i in range(len(a)-1, len(b)-1, -1):
        compute.cx(carry_ancilla, a[i])
        
    # forward sweep where b is 0
    # simplifies to a toffoli gate
    for i in range(len(b)-1, -1, -1):
        compute.ccx(a[i], out[i], carry_ancilla)
        compute.cx(carry_ancilla, a[i])
        compute.cx(carry_ancilla, out[i])

