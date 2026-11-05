from qlang import *

x = uint("x", 3)
y = uint("y", 3)
expr = x + y == 5

o = compile_oracle(expr)
print(o.circuit.draw())

