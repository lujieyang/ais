import matplotlib.pyplot as plt
import numpy as np
import exact_reduction

nb = 15
nu = 4
C = np.load("reduction_graph/C.npy")
C_det = np.load("reduction_graph/C_det.npy")
R = np.load("reduction_graph/R.npy")
P_ybu = np.load("reduction_graph/P_ybu.npy")
y_a = np.load("graph/y_a.npy")

nzs = np.arange(7, 14)
r_s = []
V_mse_s = []
for nz in nzs:
    Q, D, r_bar = exact_reduction.load_reduction_graph(nz, det=True, output_pred=True)
    B = Q@D.T@np.linalg.inv(D@D.T)
    r = r_bar.T@D.T@np.linalg.inv(D@D.T)
    policy, V = exact_reduction.value_iteration(B, r, nz, nu)
    policy_b, V_b = exact_reduction.value_iteration(np.einsum('ijk->kij', C), R.T, nb, nu)
    D_, P_xu, b = exact_reduction.load_underlying_dynamics()
    average_return, V_mse = exact_reduction.eval_performance(policy, D, C_det, V, V_b, y_a, nu, nb, b, D_, P_xu)
    r_s.append(average_return)
    V_mse_s.append(V_mse)

plt.plot(nzs, r_s)
plt.xlabel("AIS dimension")
plt.ylabel("Average return")
plt.title("Performance vs Compression Dimension")
plt.show()

plt.plot(nzs, V_mse_s)
plt.xlabel("AIS dimension")
plt.ylabel("V MSE")
plt.title("Value Function MSE vs Compression Dimension")
plt.show()
