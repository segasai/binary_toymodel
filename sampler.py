import model_lognormper
import emcee
import numpy as np
import multiprocessing as mp
import corner
import matplotlib.pyplot as plt


def dosamp():
    args = ('test_data_4_123/', 'test_data_4_123/', 1)

    nw = 40
    ndim = 5
    nsteps = 300
    nthreads = 36
    seed = 44442323
    rst = np.random.default_rng(seed)
    xs = []
    while len(xs) < nw:
        # meanper, stdper, binfrac, meanvel, sigvel
        p = np.array([
            rst.uniform(-1, 1),
            rst.uniform(0, 1),
            rst.uniform(0, 1),
            rst.normal(0, 10),
            rst.uniform(1, 10)
        ])
        if model_lognormper.hierarch_perbf_like(p, *args) > -1e20:
            xs.append(p)

    # xs = np.random.normal(size=(nw, 5)) * .2 + .5
    with mp.Pool(nthreads) as poo:
        es = emcee.EnsembleSampler(nw,
                                   ndim,
                                   model_lognormper.hierarch_perbf_like,
                                   args=args,
                                   pool=poo)
        es.random_state = np.random.mtrand.RandomState(seed).get_state()
        R = es.run_mcmc(xs, nsteps)
        xs = R[0]
        es.reset()
        R = es.run_mcmc(xs, nsteps)
    ptrue = [.5, .2, .6, 12, 10]
    corner.corner(R.chain[:, 0:, :].reshape(-1, 5),
                  truths=ptrue,
                  labels=['mlogp', 'slogp', 'bfrac', 'vel', 'disp'])
    plt.savefig('chain.png')
    return es


if __name__ == '__main__':
    dosamp()
