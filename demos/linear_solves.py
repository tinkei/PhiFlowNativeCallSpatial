from functools import partial
import phi; phi.set_logging_level('debug')
from phi.torch.flow import *


math.set_global_precision(64)
lap = math.jit_compile_linear(partial(math.laplace, padding=extrapolation.ZERO))
# y = math.random_uniform(spatial(x=1024), low=-50, high=50)
y = math.ones(spatial(x=2 * 1024))

with math.SolveTape(record_trajectories=True) as solves:
    sol = math.solve_linear(lap, y, Solve('biCGstab', 0, 1e-6, x0=y*0, max_iterations=1e9))

print(f"{solves[0].iterations} iterations. Residuals:")
for x in solves[0].residual.trajectory[:4].trajectory: print(x)
print("...")
for x in solves[0].residual.trajectory[-4:].trajectory: print(x)
