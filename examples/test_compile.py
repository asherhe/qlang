from qlang import *

x = uint("x", 2)
y = uint("y", 2)
expr = x + y == 6

o = compile_oracle(expr)
print(o.circuit.draw())

