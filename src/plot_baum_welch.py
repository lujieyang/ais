import numpy as np
import matplotlib.pyplot as plt
import exact_reduction
import value_iteration
import learn_graph as lg


na = 4
nb = 15
nu = 4
C = np.load("reduction_graph/C.npy")
C_det = np.load("reduction_graph/C_det.npy")
R = np.load("reduction_graph/R.npy")
P_ybu = np.load("reduction_graph/P_ybu.npy")
y_a = np.load("graph/y_a.npy")
D_, P_xu, b = exact_reduction.load_underlying_dynamics()
policy_b, V_b = exact_reduction.value_iteration(np.einsum('ijk->kij', C), R.T, nb, nu)

r_s = []
V_mse_s = []
nz_seed = [(11, 72), (12, 60), (12, 66), (13, 27), (13, 100), (15, 60), (17, 66), (17, 72), (19, 27)]
for nz, seed in nz_seed:
    A, B, initial_distribution, Ot = lg.load_graph(nz, seed, value_iteration.args)
    policy, V = value_iteration.value_iteration(A, B, nz, na, Ot, discount_factor=0.95)
    average_return, V_mse = value_iteration.eval_performance(policy, A, V, V_b, nb, D_, P_xu, C_det, y_a)
    r_s.append(average_return)
    V_mse_s.append(V_mse)

plt.plot(r_s, V_mse_s, '.')
plt.show()


