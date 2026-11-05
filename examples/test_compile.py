from qlang import *

x = uint("x", 3)
expr = x == 3

o = compile_oracle(expr)
print(o.circuit.draw())

