import numpy as np
import numpy.matlib as matlib
import argparse
import cvxpy as cp
import learn_graph as lg
import itertools
import multiprocessing as mp
from stirling_assignment import Stirling_Assignments
import gurobipy as gp

from gurobipy import GRB


parser = argparse.ArgumentParser(description='This code runs AIS using the Next Observation prediction version')
parser.add_argument("--save_graph", help="Save the transition probabilities", action="store_true")
parser.add_argument("--load_graph", help="Load the transition probabilities", action="store_true")
parser.add_argument("--AIS_state_size", type=int, help="Load the transition probabilities", default=11)
args = parser.parse_args()


def runGUROBIImpl(nz, nb, nu, C, R, C_det=None, P_ybu=None):
    with gp.Env(empty=True) as env:
        env.setParam("OutputFlag", 0)
        env.start()
        with gp.Model(env=env) as model:
            Q = []
            P_yzu = []
            for i in range(nu):
                Q.append(model.addVars(nz, nb, ub=1))
                if P_ybu is not None:
                    ny = P_ybu.shape[0]
                    P_ybu.append(model.addVars(ny, nz, ub=1))
            D = model.addVars(nz, nb, vtype=GRB.BINARY)
            # r = model.addVars(nb, nu)
            if C_det is not None:
                B_det = []
                for i in range(C_det.shape[2]):
                    B_det.append(model.addVars(nz, nz, vtype=GRB.BINARY))

            obj = 0
            for j in range(nb):
                for i in range(nu):
                    # obj += sum([(sum([B[i][k, l] * D[l, j] for l in range(nz)]) - sum([D[k, l] * C[l, j] for l in range(nb)]))^2 for k in range(nz)])
                    # obj += sum([(Q[i][l, j]-C[k, j]*D[l, k])**2 for l in range(nz) for k in range(nb)])
                    obj += sum([(C[k, j]*D[0, k]) for k in range(nb)])
                    # obj += (R[j, i] - sum([r[k, i] * D[k, j] for k in range(nz)]))

            model.setObjective(obj, GRB.MINIMIZE)
            model.addConstrs((D.sum("*", i) == 1 for i in range(nb)))
            model.addConstrs((D.sum(i, "*") >= 1 for i in range(nz)))
            for i in range(nu):
                model.addConstrs((Q[i].sum("*", k) == 1 for k in range(nb)))


            model.optimize()

            if not (model.status == GRB.OPTIMAL):
                print("unsuccessful...")
            else:
                print("Objective: ", obj.getValue())

def runCVXPYImpl(nz, nb, nu, C, R, C_det=None, P_ybu=None):
    Q1 = cp.Variable((nz, nb), nonneg=True)
    Q2 = cp.Variable((nz, nb), nonneg=True)
    Q3 = cp.Variable((nz, nb), nonneg=True)
    Q4 = cp.Variable((nz, nb), nonneg=True)
    if C_det is not None:
        Q_det = []
        for k in range(C_det.shape[2]):
            Q_det.append(cp.Variable((nz, nb), boolean=True))
    Q = [Q1, Q2, Q3, Q4]
    # D = cp.Variable((nz, nb), boolean=True)
    # Manually initialize the projection matrix
    D_value = np.load("reduction_graph/stirling/D_wB_11.npy")
    D = cp.Parameter((nz, nb), boolean=True, value=D_value)
    r_bar = cp.Variable((nb, nu))
    if P_ybu is not None:
        P_ybu_bar = []
        ny = P_ybu.shape[0]
        for i in range(nu):
            P_ybu_bar.append(cp.Variable((ny, nb), nonneg=True))
    loss = 0
    constraints = []
    constraints += [cp.sum(D) == nb,
                    cp.matmul(np.ones((1, nz)), D) == 1,
                    cp.matmul(D, np.ones((nb, 1))) >= 1, ]
    # Eliminating similarity transform of D
    for i in range(1, nz):
        for j in range(i):
            constraints += [D[i, j] == 0, ]
    # D in row echelon form
    f = np.zeros(nb)
    for i in range(nz-1):
        f[i] = 1
        W = generate_W(i, nz)
<<<<<<< HEAD
        print(((W@D_value@f) >= -1e-8).all())
        constraints += [W@D@f >= -1e-8, ]
=======
        # print(((W@D_value@f) >= 0).all())
        constraints += [W@D@f >= 0, ]
>>>>>>> fe301f0da50602bfb709cc9aaa76474e728c78ce

    # Eq 13, 14 enforced on columns of D
    t = cp.Variable(int(nb * (nb - 1) / 2), boolean=True)
    for j in range(nb):
        for l in range(j + 1, nb):
            jl = jl_to_flat(j, l)
            constraints += [D[:, j] - D[:, l] <= 1 - t[jl],
                            D[:, j] - D[:, l] >= t[jl] - 1,
                            D[:, j] + D[:, l] <= t[jl] + 1, ]
    constraints += [sum(t) >= nb - nz,
                    sum(t) <= (nb - nz + 1) * (nb - nz) / 2]

    for i in range(nu):
        constraints += [cp.matmul(np.ones((1, nz)), Q[i]) == 1, ]
        if P_ybu is not None:
            constraints += [cp.matmul(np.ones((1, ny)), P_ybu_bar[i]) == 1, ]
        for j in range(nb):
            b_one_hot = np.zeros(nb)
            b_one_hot[j] = 1
            # Match transition distributions
            loss += cp.norm(cp.matmul(Q[i], b_one_hot)-cp.matmul(D, C[:, :, i]@b_one_hot))
            # Match reward
            loss += cp.norm(R[j, i] - cp.matmul(r_bar[:, i], b_one_hot))

            for l in range(j+1, nb):
                constraints += [Q[i][:, j] - Q[i][:, l] <= 1-t[jl_to_flat(j, l)], ]

    # Match observation prediction
    if P_ybu is not None:
        for i in range(nu):
            constraints += [cp.matmul(np.ones((1, ny)), P_ybu_bar[i]) == 1, ]
            for j in range(nb):
                b_one_hot = np.zeros(nb)
                b_one_hot[j] = 1
                loss += cp.norm(cp.matmul(P_ybu_bar[i], b_one_hot) - P_ybu[:, :, i] @ b_one_hot)
                for l in range(j + 1, nb):
                    constraints += [P_ybu_bar[i][:, j] - P_ybu_bar[i][:, l] <= 1 - t[jl_to_flat(j, l)], ]


    if C_det is not None:
        # B_det is also probability transition matrix
        constraints += [cp.matmul(np.ones((1, nz)), Q_det[k]) == 1, ]
        for k in range(C_det.shape[2]):
            for j in range(nb):
                b_one_hot = np.zeros(nb)
                b_one_hot[j] = 1
                loss += cp.norm(cp.matmul(Q_det[k], b_one_hot) - cp.matmul(D, C_det[:, :, k] @ b_one_hot))
                for l in range(j + 1, nb):
                    constraints += [Q_det[k][:, j] - Q_det[k][:, l] <= 1 - t[jl_to_flat(j, l)], ]

    objective = cp.Minimize(loss)
    problem = cp.Problem(objective, constraints)

    # solve problem
    problem.solve(solver=cp.GUROBI, verbose=True)

    if not (problem.status == cp.OPTIMAL):
        print("unsuccessful...")
    else:
        print("loss ", loss.value)

    if C_det is not None:
        Q_det_out = []
        for i in range(len(Q_det)):
            Q_det_out.append(Q_det[i].value)
        return np.array(Q_det_out), np.array([Q1.value, Q2.value, Q3.value, Q4.value]), D.value, r_bar.value
    else:
        return np.array([Q1.value, Q2.value, Q3.value, Q4.value]), D.value, r_bar.value


def generate_W(i, nz):
    nr = nz-i-1
    Rblock = matlib.repmat(-np.eye(nr), i+1, 1)
    Lblock = np.zeros(((i+1)*(nz-i-1), i+1))
    for l in range(i+1):
        Lblock[l*nr:(l+1)*nr, l] = 1
    return np.hstack((Lblock, Rblock))


def jl_to_flat(j, l):
    return int((nb*2-1-j)*j/2+l-(j+1))


def bilinear_alternation(nz, nb, nu, C, R, epsilon=1e-5, C_det=None, P_ybu=None):
    B = np.random.random((nz, nz, nu))
    B = B/np.sum(B, axis=0)
    B = np.einsum('ijk->kij', B)
    old_B = B
    r = np.zeros((nz, nu))
    converge = False

    while not converge:
        D = solve_D(nz, nb, nu, C, R, B, r, C_det, P_ybu)
        B, r = solve_B_r(nz, nb, nu, C, R, D, C_det, P_ybu)
        if np.linalg.norm(B-old_B) < epsilon:
            converge = True
    return B, D, r


def parallel_convex_opt(nz, nb, nu, C, R, sample_D=None, C_det=None, P_ybu=None):
    """
    sample_D: boolean. True:
    """
    pool = mp.Pool(mp.cpu_count())
    Ds = []
    s = Stirling_Assignments(nb, nz)
    if sample_D is not None:
        for D in s.all_partitions_generator_D_rep():
            Ds.append(D)
        ind = np.random.choice(len(Ds), sample_D, replace=False)
        for i in ind:
            if i % 1000 == 0:
                print(i, "/", sample_D)
            D = Ds[i]
            pool.apply_async(solve_B_r, args=(nz, nb, nu, C, R, D, C_det, P_ybu), callback=collect_result)
    else:
        for D in s.all_partitions_generator_D_rep():
            pool.apply_async(solve_B_r, args=(nz, nb, nu, C, R, D, C_det, P_ybu), callback=collect_result)
    # parallel_results = [pool.apply(solve_B_r, args=(nz, nb, nu, C, R, D, C_det, P_ybu)) for D in Ds]
    pool.close()
    pool.join()
    ind = np.argmin(np.array(parallel_loss))
    if sample_D is not None:
        np.save("reduction_graph/sample/parallel_loss_{}".format(nz), parallel_loss)
        np.save("reduction_graph/sample/parallel_matrices_{}".format(nz), parallel_matrices)
    else:
        np.save("reduction_graph/parallel_loss_{}".format(nz), parallel_loss)
        np.save("reduction_graph/parallel_matrices_{}".format(nz), parallel_matrices)
    # loss = [result[0] for result in parallel_results]
    # ind = np.argmin(np.array(loss))
    # return parallel_results[ind][1:]
    return parallel_matrices[ind]


def collect_result(result):
    global parallel_loss, parallel_matrices
    parallel_loss.append(result[0])
    parallel_matrices.append(result[1:])


def solve_D(nz, nb, nu, C, R, B, r, C_det=None, P_ybu=None, B_det=None, P_yzu=None):
    for i in range(B.shape[0]):
        assert((abs(np.ones((1, nz))@B[i]-1) < 1e-5).all())
    D = cp.Variable((nz, nb), boolean=True)

    loss = 0
    constraints = []
    constraints += [cp.sum(D) == nb,
                    cp.matmul(np.ones((1, nz)), D) == 1,
                    cp.matmul(D, np.ones((nb, 1))) >= 1, ]
    for j in range(nb):
        b_one_hot = np.zeros(nb)
        b_one_hot[j] = 1
        for i in range(nu):
            # Match transition distributions
            loss += cp.norm(B[i]@cp.matmul(D, b_one_hot)-cp.matmul(D, C[:, :, i]@b_one_hot))
            # Match reward
            loss += cp.norm(R[j, i] - r[:, i]@cp.matmul(D, b_one_hot))

            if P_ybu is not None:
                loss += cp.norm(cp.matmul(P_yzu[i], D@b_one_hot) - P_ybu[:, :, i] @ b_one_hot)

        if C_det is not None:
            for k in range(C_det.shape[2]):
                loss += cp.norm(cp.matmul(B_det[k], D@b_one_hot) - D@C_det[:, :, k] @ b_one_hot)

    objective = cp.Minimize(loss)
    problem = cp.Problem(objective, constraints)

    # solve problem
    problem.solve(solver=cp.GUROBI, verbose=False)

    if not (problem.status == cp.OPTIMAL):
        print("unsuccessful...")
    else:
        print("loss ", loss.value)

    return D.value


def solve_B_r(nz, nb, nu, C, R, D, C_det=None, P_ybu=None):
    assert(np.sum(D) == nb)
    assert((np.ones((1, nz))@D == 1).all())
    assert((D@np.ones((nb, 1)) >= 1).all())
    B = []
    for i in range(nu):
        B.append(cp.Variable((nz, nz), nonneg=True))
    if C_det is not None:
        B_det = []
        for k in range(C_det.shape[2]):
            B_det.append(cp.Variable((nz, nz), boolean=True))

    r = cp.Variable((nz, nu))
    if P_ybu is not None:
        P_yzu = []
        ny = P_ybu.shape[0]
        for i in range(nu):
            P_yzu.append(cp.Variable((ny, nz), nonneg=True))

    loss = 0
    constraints = []
    for j in range(nb):
        b_one_hot = np.zeros(nb)
        b_one_hot[j] = 1
        for i in range(nu):
            # Match transition distributions
            loss += cp.norm(cp.matmul(B[i], D@b_one_hot)-D@C[:, :, i]@b_one_hot)
            # Match reward
            loss += cp.norm(R[j, i] - cp.matmul(r[:, i], D@b_one_hot))
            constraints += [cp.matmul(np.ones((1, nz)), B[i]) == 1,]

            if P_ybu is not None:
                loss += cp.norm(cp.matmul(P_yzu[i], D@b_one_hot) - P_ybu[:, :, i] @ b_one_hot)
                constraints += [cp.matmul(np.ones((1, ny)), P_yzu[i]) == 1, ]

        if C_det is not None:
            for k in range(C_det.shape[2]):
                loss += cp.norm(cp.matmul(B_det[k], D@b_one_hot) - D@C_det[:, :, k] @ b_one_hot)
                # B_det is also part of permutation matrix
                constraints += [cp.matmul(np.ones((1, nz)), B_det[k]) == 1, ]

    objective = cp.Minimize(loss)
    problem = cp.Problem(objective, constraints)

    # solve problem
    problem.solve(solver=cp.GUROBI, verbose=False)

    if not (problem.status == cp.OPTIMAL):
        print("unsuccessful...")
    else:
        # print("loss ", loss.value)
        pass

    B_out = []
    for i in range(len(B)):
        B_out.append(B[i].value)
    if C_det is not None:
        B_det_out = []
        for i in range(len(B_det)):
            B_det_out.append(B_det[i].value)
        return loss.value, np.array(B_det_out), B, D, r_bar.value
    else:
        return loss.value, np.array(B_out), D, r.value


def value_iteration(B, r, nz, na, epsilon=0.0001, discount_factor=0.95):
    """
    Value Iteration Algorithm.

    Args:
        B: numpy array of size(na, nz, nz). transition probabilities of the environment P(z(t+1)|z(t), a(t)).
        r: numpy array of size (na, nz). reward function r(z(t),a(t))
        nz: number of AIS in the environment.
        na: number of actions in the environment.
        epsilon: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.

    Returns:
        A tuple (policy, V) of the optimal policy and the optimal value function.
    """

    def one_step_lookahead(V, a, z):
        z_one_hot = np.zeros(nz)
        z_one_hot[z] = 1
        z_next = B[a, :, :]@z_one_hot
        # ind = (z_next > epsilon)
        v = r[a, :]@z_next + discount_factor * z_next@V

        return v

    # start with initial value function and initial policy
    V = np.zeros(nz)
    policy = np.zeros([nz, na])

    n = 0
    # while not the optimal policy
    while True:
        print('Iteration: ', n)
        # for stopping condition
        delta = 0

        # loop over state space
        for z in range(nz):

            actions_values = np.zeros(na)

            # loop over possible actions
            for a in range(na):
                # apply bellman eqn to get actions values
                actions_values[a] = one_step_lookahead(V, a, z)

            # pick the best action
            best_action_value = max(actions_values)

            # get the biggest difference between best action value and our old value function
            delta = max(delta, abs(best_action_value - V[z]))

            # apply bellman optimality eqn
            V[z] = best_action_value

            # to update the policy
            best_action = np.argmax(actions_values)

            # update the policy
            policy[z] = np.eye(na)[best_action]


        # if optimal value function
        if (delta < epsilon):
            break
        n += 1

    return policy, V


def eval_performance(policy, D, C_det, V, V_b, y_a, na, nb, b, D_, P_xu, B_det=None, n_episodes=100, epsilon=1e-8, beta=0.95):
    returns = []
    Vs = []
    V_bs = []
    for n_eps in range(n_episodes):
        reward_episode = []
        y = lg.env.reset()

        uniform_distribution = np.ones(nb) / nb
        bn = np.ones(11) / 11
        while True:
            # sample b from initial distribution
            ind_b = np.where(np.random.multinomial(1, uniform_distribution) == 1)[0][0]
            # check b agrees with the first observation
            if (C_det[ind_b, :, y == y_a[:, 0]] > epsilon).any():
                b_one_hot = np.zeros(nb)
                b_one_hot[ind_b] = 1
                break
        z_one_hot = D@b_one_hot

        for j in range(1000):
            try:
                ind_z = np.where(z_one_hot == 1)[0][0]
            except:
                pass
                # print("No corresponding z")
            Vs.append(V[ind_z])
            V_bs.append(V_b[b_one_hot == 1][0])

            action = np.arange(na)[policy[ind_z].astype(bool)][0]

            y, reward, done, _ = lg.env.step(action)
            reward_episode.append(reward)

            # Update one-hot b with belief values in case the first guess is incorrect
            bn = D_[y]@P_xu[action]@bn
            bn = bn / np.sum(bn, axis=0)
            ind_b = np.where((bn == b).all(axis=1))[0][0]
            b_one_hot = np.zeros(nb)
            b_one_hot[ind_b] = 1
            # ind_ya = np.where((y == y_a[:, 0])*(action == y_a[:, 1]))[0][0]
            # b_one_hot = C_det[:, :, ind_ya]@b_one_hot
            if B_det is not None:
                z_one_hot = B_det@z_one_hot
            else:
                z_one_hot = D@b_one_hot

            if done:
                break

        rets = []
        R = 0
        for i, r in enumerate(reward_episode[::-1]):
            R = r + beta * R
            rets.insert(0, R)
        returns.append(rets[0])

    average_return = np.mean(returns)
    V_mse = np.linalg.norm(np.array(Vs)-np.array(V_bs))
    print("Average reward: ", average_return)
    print("V mse: ", V_mse)
    print("Average V mse", V_mse/len(Vs))
    return average_return, V_mse


def save_reduction_graph(Q, D, r_bar, nz, Q_det=None, output_pred=False):
    folder_name = "reduction_graph/"
    if Q_det is not None:
        folder_name += "det/"
        if output_pred:
            np.save(folder_name + "Q_{}_y".format(nz), Q)
            np.save(folder_name + "D_{}_y".format(nz), D)
            np.save(folder_name + "r_fit_{}_y".format(nz), r_bar)
            np.save(folder_name + "Q_det_{}_y".format(nz), Q_det)
        else:
            np.save(folder_name + "Q_{}".format(nz), Q)
            np.save(folder_name + "D_{}".format(nz), D)
            np.save(folder_name + "r_fit_{}".format(nz), r_bar)
            np.save(folder_name + "Q_det_{}".format(nz), Q_det)
    else:
        if output_pred:
            np.save(folder_name + "Q_{}_y".format(nz), Q)
            np.save(folder_name + "D_{}_y".format(nz), D)
            np.save(folder_name + "r_fit_{}_y".format(nz), r_bar)
        else:
            np.save(folder_name + "Q_{}".format(nz), Q)
            np.save(folder_name + "D_{}".format(nz), D)
            np.save(folder_name + "r_fit_{}".format(nz), r_bar)


def save_B_r(B, D, r, nz, B_det=None, sample=False):
    folder_name = "reduction_graph/"
    if sample:
        folder_name += "sample/"
    if B_det is not None:
        folder_name += "det/"
        np.save(folder_name + "B_{}".format(nz), B)
        np.save(folder_name + "D_wB_{}".format(nz), D)
        np.save(folder_name + "r_{}".format(nz), r)
        np.save(folder_name + "B_det_{}".format(nz), B_det)
    else:
        np.save(folder_name + "B_{}".format(nz), B)
        np.save(folder_name + "D_wB_{}".format(nz), D)
        np.save(folder_name + "r_{}".format(nz), r)


def load_reduction_graph(nz, det=False):
    folder_name = "reduction_graph/"
    if det:
        folder_name += "det/"
        Q = np.load(folder_name + "Q_{}.npy".format(nz))
        D = np.load(folder_name + "D_{}.npy".format(nz))
        r_bar = np.load(folder_name + "r_fit_{}.npy".format(nz))
        Q_det = np.load(folder_name + "Q_det{}.npy".format(nz))
        return Q_det, Q, D, r_bar
    else:
        Q = np.load(folder_name + "Q_{}.npy".format(nz))
        D = np.load(folder_name + "D_{}.npy".format(nz))
        r_bar = np.load(folder_name + "r_fit_{}.npy".format(nz))
        return Q, D, r_bar


def load_B_r(nz, det=False, sample=False):
    folder_name = "reduction_graph/"
    if sample:
        folder_name += "sample/"
    if det:
        folder_name += "det/"
        B = np.load(folder_name + "B_{}.npy".format(nz))
        D = np.load(folder_name + "D_wB_{}.npy".format(nz))
        r = np.load(folder_name + "r_{}.npy".format(nz))
        B_det = np.load(folder_name + "B_det{}.npy".format(nz))
        return B_det, B, D, r
    else:
        B = np.load(folder_name + "B_{}.npy".format(nz))
        D = np.load(folder_name + "D_wB_{}.npy".format(nz))
        r = np.load(folder_name + "r_{}.npy".format(nz))
        return B, D, r


def load_underlying_dynamics():
    D_ = np.load("reduction_graph/D_.npy")
    P_xu = np.load("reduction_graph/P_xu.npy")
    b = np.load("reduction_graph/b.npy")
    return D_, P_xu, b


if __name__ == "__main__":
    np.random.seed(0)
    nz = lg.args.AIS_state_size
    nb = 15
    nu = 4
    C = np.load("reduction_graph/C.npy")
    C_det = np.load("reduction_graph/C_det.npy")
    R = np.load("reduction_graph/R.npy")
    P_ybu = np.load("reduction_graph/P_ybu.npy")
    y_a = np.load("graph/y_a.npy")
    parallel_loss = []
    parallel_matrices = []

    if args.load_graph:
        Q, D, r_bar = load_reduction_graph(nz)
        # B, D, r = load_B_r(nz, sample=True)
    else:
        Q, D, r_bar = runCVXPYImpl(nz, nb, nu, C, R, P_ybu=P_ybu)
        # Q, D, r_bar = runGUROBIImpl(nz, nb, nu, C, R)
        # Q_det, Q, D, r_bar = runCVXPYImpl(nz, nb, nu, C, R, C_det=C_det)
        # B, D, r = bilinear_alternation(nz, nb, nu, C, R)
        # [B, D, r] = parallel_convex_opt(nz, nb, nu, C, R, 50000)
        # r = r.T
        if args.save_graph:
            save_reduction_graph(Q, D, r_bar, nz, output_pred=True)
            # save_B_r(B, D, r, nz)


    B = Q@D.T@np.linalg.inv(D@D.T)
    r = r_bar.T@D.T@np.linalg.inv(D@D.T)
    policy, V = value_iteration(B, r, nz, nu)
    policy_b, V_b = value_iteration(np.einsum('ijk->kij', C), R.T, nb, nu)
    D_, P_xu, b = load_underlying_dynamics()
    eval_performance(policy, D, C_det, V, V_b, y_a, nu, nb, b, D_, P_xu)




