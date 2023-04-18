# This script contains the python implementation of FPLinQ scheduling algorithm (original work by K. Shen and W. Yu)

import numpy as np
from settings_MIMO import *
from tqdm import trange

# Parallel computation over multiple layouts
def FP_power_control(g):
    n_layouts = np.shape(g)[0]
    assert np.shape(g)==(n_layouts, N_LINKS, N_LINKS)
    # For this project, not doing weighted sum-rate optimization
    weights = np.ones([n_layouts, N_LINKS])
    g_diag = np.diagonal(g, axis1=1, axis2=2)
    g_nondiag = g * ((np.identity(N_LINKS) < 1).astype(float))
    # For matrix multiplication and dimension matching requirement, reshape into column vectors
    weights = np.expand_dims(weights, axis=-1)
    g_diag = np.expand_dims(g_diag, axis=-1)
    x = np.ones([n_layouts, N_LINKS, 1])
    tx_powers = np.ones([n_layouts, N_LINKS, 1]) * TX_POWER  # assume same power for each transmitter
    # In the computation below, every step's output is with shape: number of samples X N X 1
    for i in range(150):
        # Compute z
        p_x_prod = x * tx_powers
        z_denominator = np.matmul(g_nondiag, p_x_prod) + NOISE_POWER
        z_numerator = g_diag * p_x_prod
        z = z_numerator / z_denominator
        # compute y
        y_denominator = np.matmul(g, p_x_prod) + NOISE_POWER
        y_numerator = np.sqrt(z_numerator * weights * (z + 1))
        y = y_numerator / y_denominator
        # compute x
        x_denominator = np.matmul(np.transpose(g, (0,2,1)), np.power(y, 2)) * tx_powers
        x_numerator = y * np.sqrt(weights * (z + 1) * g_diag * tx_powers)
        x_new = np.power(x_numerator / x_denominator, 2)
        x_new[x_new > 1] = 1  # thresholding at upperbound 1
        x = x_new
    assert np.shape(x)==(n_layouts, N_LINKS, 1)
    x = np.squeeze(x, axis=-1)
    return x

def dinkelbach_onestep(pl_dl, pl_cl, y):
    # objective
    c = [0]*N_LINKS+[-1]
    # constraint on min dummy variable
    A_ub = pl_cl*y-np.diag(pl_dl)
    A_ub = np.concatenate([A_ub, np.ones([N_LINKS,1])],axis=1)
    b_ub = -np.ones([N_LINKS,1])*(NOISE_POWER / TX_POWER)
    # constraint on power control variables 0~1
    bounds = []
    for i in range(N_LINKS):
        bounds.append((0,1))
    bounds.append((0, None))
    res = linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds)
    x = res['x'][:-1]
    t = res['x'][-1]
    return x, t

# Due to large computation time, try to load first if results have been computed
def maxmin_power_control(pl_dl, pl_cl, algo):
    VISUALIZE_SINRS = False
    print("============================{} Max Min-Rate Power Control==================".format(algo))
    tolerance = 1e-17
    n_layouts = np.shape(pl_dl)[0]
    pc_save_path = "Data/dinkelbach_{}.npy".format(SETTING_STRING) if algo=="Dinkelbach" else "Data/gp_{}.npy".format(SETTING_STRING)
    try:
        pc = np.load(pc_save_path)
    except:
        pc = []
        for i in trange(n_layouts): # use GP to solve one layout at a time
            pl_dl_one_layout, pl_cl_one_layout = pl_dl[i], pl_cl[i]
            if algo == "GP":
                pc_one_layout = gp_one_layout(pl_dl_one_layout, pl_cl_one_layout)
            else:
                pc_one_layout = np.zeros(N_LINKS, dtype=np.float64)
                while True:
                    sinrs_numerators = pc_one_layout * pl_dl_one_layout  # (N, )
                    sinrs_denominators = np.squeeze(np.matmul(pl_cl_one_layout, np.expand_dims(pc_one_layout, axis=-1))) + NOISE_POWER / TX_POWER  # (N, )
                    sinrs = sinrs_numerators/sinrs_denominators
                    min_sinr = np.min(sinrs)
                    pc_one_layout, t = dinkelbach_onestep(pl_dl_one_layout, pl_cl_one_layout, min_sinr)
                    if (t <= tolerance):
                        break
                    if np.max(sinrs)-min_sinr < 0.03*min_sinr:
                        break
            pc.append(pc_one_layout)
        pc = np.array(pc); assert np.shape(pc)==(n_layouts, N_LINKS)
        print("Saving computed {} allocations at: {}".format(algo, pc_save_path))
        np.save(pc_save_path, pc)
    else:
        print("Loaded existing {} allocations from {}".format(algo, pc_save_path))
    finally:
        assert np.shape(pc)==(n_layouts, N_LINKS)
        sinrs_numerators = pc * pl_dl  # layouts X N
        sinrs_denominators = np.squeeze(np.matmul(pl_cl, np.expand_dims(pc, axis=-1))) + NOISE_POWER / TX_POWER  # layouts X N
        sinrs = sinrs_numerators / sinrs_denominators  # layouts X N
        largest_percentage_offset = 100*np.max((np.max(sinrs, axis=1)-np.min(sinrs, axis=1))/np.min(sinrs, axis=1))
        print("All layouts, largest percentage offset on SINR: {:5.2f}%; min SINR: {:5.2f}".format(largest_percentage_offset, np.min(sinrs)))
        return pc


def gp_one_layout(pl_dl, pl_cl):
    t = Variable("t", "-", "min sinr")
    pc = VectorVariable(N_LINKS, "pc", "-", "power control")
    objective = 1/t
    constraints = pc<=1
    for i in range(N_LINKS):
        constraint_LHS = NOISE_POWER*t/(pc[i]*pl_dl[i]*TX_POWER)
        for j in range(N_LINKS):
            if j!=i:
                constraint_LHS += pl_cl[i,j]*t*pc[j]/(pc[i]*pl_dl[i])
        constraints.append(constraint_LHS<=1)
    m = Model(objective, constraints)
    sol = m.solve(verbosity=0)
    return sol["variables"]["pc"]
