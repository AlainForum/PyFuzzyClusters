import numpy as np


def try_again(x, q, algorithm, attempts, verbose=False, **kwargs):
    j_best = np.infty
    n = 0
    while True:
        if n > attempts:
            break
        trial = algorithm(x, q, **kwargs).optimize()
        if trial.J < j_best:
            best_trial = trial
            j_best = trial.J
            if verbose:
                print(
                    "Trial {}/{}: j_best = {:.2f}".format(n, attempts, j_best))
        elif np.isnan(trial.J):
            continue
        n += 1
    return best_trial
