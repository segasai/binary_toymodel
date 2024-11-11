import model_lognormper
import emcee
import numpy as np
import multiprocessing as mp
import corner
import matplotlib.pyplot as plt


def dosamp():
    args = ('test_data_test240930_1_123/', 'test_data_test240930_1_123/', 1)
    #args = ('tmp3/', 'tmp3/', 1)
    min_per, max_per = 0.1, 1e9  # prior limits
    kwargs = {'min_per': min_per, 'max_per': max_per}

    nw = 72
    ndim = 5
    nsteps = 300
    nthreads = 36
    seed = 44442323
    rst = np.random.default_rng(seed)
    xs = []
    niter = 1000
    while len(xs) < nw:
        # meanper, stdper, binfrac, meanvel, sigvel
        p = np.array([
            rst.uniform(0, 3),
            rst.uniform(0, 1),
            rst.uniform(0, 1),
            rst.normal(0, 10),
            rst.uniform(1, 100)
        ])
        if model_lognormper.hierarch_perbf_like(p, *args, **kwargs) > -1e20:
            xs.append(p)
        niter -= 1
        if len(xs) == 0 and niter == 0:
            raise Exception('oops')
    # xs = np.random.normal(size=(nw, 5)) * .2 + .5
    with mp.Pool(nthreads) as poo:
        es = emcee.EnsembleSampler(nw,
                                   ndim,
                                   model_lognormper.hierarch_perbf_like,
                                   args=args,
                                   kwargs=kwargs,
                                   pool=poo)
        es.random_state = np.random.mtrand.RandomState(seed).get_state()
        es.run_mcmc(xs, nsteps)
        lp = es.get_log_prob()[-int(0.2 * nsteps):]  # last 20% of burnin
        chain = es.get_chain()[-int(0.2 * nsteps):]
        chain = chain[lp > lp.max() - 10]  # pick likely enough samples
        xs = chain[rst.permutation(len(chain))][:nw]
        # xs = R[0]
        es.reset()
        es.run_mcmc(xs, nsteps)
    ptrue = [2.5, 2, .5, 0, 100]
    corner.corner(es.chain[:, 0:, :].reshape(-1, ndim),
                  truths=ptrue,
                  labels=['mlogp', 'slogp', 'bfrac', 'vel', 'disp'])
    plt.savefig('chain.png')
    return es


if __name__ == '__main__':
    es = dosamp()
