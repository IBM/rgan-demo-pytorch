import numpy as np
from sklearn.mixture import GaussianMixture

def rejection_step(q_X_N_rej, q_Y_gm, mm):
    q_Y_N_rej = mm(q_X_N_rej)
    q_Y_rej_gm_pars = {"n_components": 40, "reg_covar": 1e-3, "tol": 1e-3, "max_iter": 100}
    q_Y_rej_gm = GaussianMixture(**q_Y_rej_gm_pars).fit(q_Y_N_rej)
    q_Y_log_prob = q_Y_gm.score_samples(q_Y_N_rej)
    q_Y_rej_log_prob = q_Y_rej_gm.score_samples(q_Y_N_rej)
    log_diff = np.minimum(q_Y_log_prob - q_Y_rej_log_prob, 0.0)
    ratio = np.exp(log_diff)
    uniform = np.random.uniform(size=ratio.size)
    accept = uniform < ratio
    return q_X_N_rej[accept]
