import numpy as np
import learn_graph as lg

nx = 11
nu = 4
ny = 7
P_xu = np.zeros((nu, nx, nx))
current_ind = list(range(nx))
next_indices = [[0, 1, 2, 3, 4, 0, 2, 4, 5, 7, 6],
                [5, 1, 6, 3, 7, 8, 10, 9, 8, 9, 10],
                [1, 2, 3, 4, 4, 5, 6, 7, 8, 9, 10],
                [0, 0, 1, 2, 3, 5, 6, 7, 8, 9, 10]]

for i in range(nu):
    next_ind = next_indices[i]
    assert(len(next_ind) == len(current_ind))
    P_xu[i, next_ind, current_ind] = 1

D_ = np.zeros((ny, nx, nx))
ind = [[0],
       [1, 3],
       [2],
       [4],
       [5, 6, 7],
       [8, 9],
       [10]]
for i in range(ny):
    D_[ i, ind[i], ind[i]] = 1

nb = 15
uniform_distribution = np.ones(nx)/nx
initial_distribution = np.random.rand(nx, nb)
initial_distribution = initial_distribution/np.sum(initial_distribution,axis=0)
# b0 = np.hstack((uniform_distribution.reshape((-1,1)), initial_distribution))
b0 = uniform_distribution

y_a, y, a, O = lg.load_trajectory(lg.args)

bs = []
for r in range(len(y)):
    yr = np.array(y[r])
    ar = np.array(a[r])
    Or = O[r]
    bn = b0
    for t in range(len(yr)):
        bn = D_[yr[t]]@P_xu[ar[t]]@bn
        # bn = bn/np.sum(bn)
        bn = bn / np.sum(bn, axis=0)
        bs.append(bn)


b = np.unique(np.array(bs), axis=0)
nb = b.shape[0]
y_unique = np.unique(y.reshape((-1)))
ny = len(y_unique)

C = np.zeros((nb, nb, nu))
R = np.zeros((nb, nu))
N_ybu = np.zeros((ny, nb, nu))
# Find reward for each belief
for r in range(len(y)):
    if r % 10 == 0:
        print("r ", r)
    yr = np.array(y[r])
    ar = np.array(a[r])
    Or = O[r]
    bn = b0
    for t in range(len(yr)):
        # Probability (b_{n+1}|b_n, u_n)
        # Pbb = np.zeros(nb)
        # for i in range(nb):
        #     for yb in range(ny):
        #         Py = D_[ yb]@P_xu[ar[t]]@bn
        #         if np.array_equal(b[i, :], Py / np.sum(Py)):
        #             Pbb[i] += np.diag(D_[ yb])@P_xu[ar[t]]@bn
        if bn in b:
            ind = np.where((bn == b).all(axis=1))[0][0]
            R[ind, ar[t]] = Or[t][0]
            N_ybu[yr[t], ind, ar[t]] += 1
            # C[:, ind, ar[t]] = Pbb

        bn = D_[yr[t]]@P_xu[ar[t]]@bn
        bn = bn / np.sum(bn, axis=0)

P_ybu = N_ybu/np.sum(N_ybu, axis=0)[None, :]

# Calculate probability (b_{n+1}|b_n, u_n)
for j in range(nb):
    bn = b[j, :]
    for u in range(nu):
        Pbb = np.zeros(nb)
        for i in range(nb):
            for yb in range(ny):
                Py = D_[ yb]@P_xu[u]@bn
                if np.array_equal(b[i, :], Py / np.sum(Py)):
                    Pbb[i] += np.diag(D_[ yb])@P_xu[u]@bn
        C[:, j, u] = Pbb

# Calculate deterministic transition b_{n+1}= C_det(b_n, u_n, y_n)
n_ya = len(y_a)
C_det = np.zeros((nb, nb, n_ya))
for j in range(nb):
    bn = b[j, :]
    for k in range(n_ya):
        b_next = D_[ y_a[k][0]]@P_xu[y_a[k][1]]@bn
        b_next = b_next / np.sum(b_next, axis=0)
        for i in range(nb):
            if np.array_equal(b[i, :], b_next):
                C_det[i, j, k] = 1
                break


# np.save("reduction_graph/b", b)
# np.save("reduction_graph/C", C)
# np.save("reduction_graph/R", R)
# np.save("reduction_graph/C_det", C_det)
# np.save("reduction_graph/P_ybu", P_ybu)
np.save("reduction_graph/D_", D_)
np.save("reduction_graph/P_xu", P_xu)



