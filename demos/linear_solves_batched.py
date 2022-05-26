import phi; phi.set_logging_level('debug')
from phi.flow import *


@math.jit_compile_linear
def f(x):
    return (x.x[1:] - x.x[:-1]) * wrap([1, 2], batch('vb'))


y = math.ones(spatial(x=5)) * wrap((1, 2), batch('vb')) * math.linspace(0, 10, batch(b=11))
math.solve_linear(f, y.x[1:], Solve('biCG', 1e-5, 1e-5, x0=0 * y))
