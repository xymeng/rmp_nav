import numpy as np


def log_psd_matrix(m):
    w, v = np.linalg.eig(m)
    return np.linalg.multi_dot([v, np.diag(np.log(w + 1.0)), v.T])


def exp_psd_matrix(m):
    w, v = np.linalg.eig(m)
    return np.linalg.multi_dot([v, np.diag(np.exp(w) - 1.0), v.T])
