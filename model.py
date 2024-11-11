import numpy as np
import dynesty
import scipy.special
from idlsave import idlsave
import multiprocessing as mp
import glob

VREF = 0

DISP_PRIOR = 100  # width of gaussian prior on velocity


def make_samp(Nstars,
              Npt,
              Nsamp,
              dt=5,
              bin_frac=0.5,
              min_per=0.001,
              max_per=10,
              vel_disp=5,
              vel_err=0.5,
              seed=44):
    rng = np.random.default_rng(seed)
    per = 10**rng.uniform(np.log10(min_per), np.log10(max_per), size=Nstars)
    phase = rng.uniform(0, 2 * np.pi, size=Nstars)
    v0 = rng.normal(size=Nstars) * vel_disp
    cosi = rng.uniform(0, 1, size=Nstars)
    sini = np.sqrt(1 - cosi**2)
    amp0 = VREF / per**(1. / 3) * sini
    res = []
    is_bin = (rng.uniform(0, 1, size=Nstars) < bin_frac).astype(int)
    for i in range(Nstars):
        ts = rng.uniform(0, dt, size=Npt)
        v = (v0[i] +
             amp0[i] * is_bin[i] * np.sin(2 * np.pi / per[i] * ts - phase[i]) +
             rng.normal(size=Npt) * vel_err)
        ev = v * 0 + vel_err
        res.append([ts, v, ev])
    truep = np.array([v0, per, phase, sini, is_bin]).T
    return res, truep


class Prior:

    def __init__(self, vel_sig, min_per, max_per):
        self.MINV = -1000
        self.min_per = min_per
        self.max_per = max_per
        self.vel_sig = vel_sig

    def __call__(self, x):
        V = scipy.special.ndtri(x[0]) * self.vel_sig
        per = self.min_per * (self.max_per / self.min_per)**x[1]
        x1 = x * 0
        x1[0] = V
        x1[1] = per
        x1[2] = x[2] * 2 * np.pi
        x1[3] = np.sqrt(1 - x[3]**2)  # sini
        return x1


class PriorNB:

    def __init__(self, vel_sig):
        self.MINV = -1000
        self.vel_sig = vel_sig

    def __call__(self, x):
        V = scipy.special.ndtri(x[0]) * self.vel_sig
        x1 = x * 0
        x1[0] = V
        return x1


def like(p, data):
    t, v, ev, bin_switch = data
    if bin_switch:
        v0, per, phase, sini = p
    else:
        v0 = p[0]
        per, phase = 1, 0
        sini = 0
    amp0 = VREF / per**(1. / 3) * sini
    model = amp0 * np.sin(2 * np.pi / per * t - phase) + v0
    logl = -0.5 * np.sum(((model - v) / ev)**2)
    return logl


def posterior(t, v, ev, binary, minp=None, maxp=None, seed=1):
    if binary:
        pri = Prior(DISP_PRIOR, minp, maxp)
        ndim = 4
        periodic = [2]
    else:
        pri = PriorNB(DISP_PRIOR, )
        ndim = 1
        periodic = None
    data = ((t, v, ev, binary), )

    rng = np.random.default_rng(seed)
    nlive = 1000
    dns = dynesty.DynamicNestedSampler(like,
                                       pri,
                                       ndim,
                                       rstate=rng,
                                       nlive=nlive,
                                       bound='multi',
                                       sample='rslice',
                                       periodic=periodic,
                                       logl_args=data)
    # dns.run_nested(n_effective=10000, print_progress=False)
    print_progress = False
    dns.run_nested(n_effective=10000,
                   print_progress=print_progress)  #, maxbatch=1)
    # for i in range(10):
    #    dns.add_batch(mode='full')
    res = dns.results.samples_equal()
    logz = dns.results.logz[-1]
    return res, logz


class si:
    cache = None


def hierarch(nsamp=10000, seed=12):
    nsamp0 = 10000
    if si.cache is None:
        si.cache = np.array([
            idlsave.restore(_, 'curr')[0][:, 0][:nsamp0]
            for _ in glob.glob('tmp/*psav')
        ])
    ARR = si.cache
    rng = np.random.default_rng(seed)
    permut = rng.integers(nsamp0, size=(ARR.shape[0], nsamp))
    ARR = ARR[np.arange(ARR.shape[0])[:, None] + permut * 0, permut]

    def func(S):
        lpdf1 = scipy.stats.norm(0, S).logpdf(ARR)
        lpdf2 = scipy.stats.norm(0, DISP_PRIOR).logpdf(ARR)
        return (scipy.special.logsumexp(lpdf1 - lpdf2, axis=1) -
                np.log(nsamp)).sum()

    xgrid = 10**np.linspace(np.log10(0.1), np.log10(20), 100)
    ygrid = np.array([func(_) for _ in xgrid])
    return xgrid, ygrid


if __name__ == '__main__':
    Nstars = 1000
    Npt = 1
    vel_disp = DISP_PRIOR
    Nsamp = 1000
    min_per = 0.1
    max_per = 10000
    vel_err = 0.5
    bin_frac = 0.5
    pref0 = 'tmp3/'
    S, truep = make_samp(Nstars,
                         Npt,
                         Nsamp,
                         dt=5,
                         bin_frac=bin_frac,
                         min_per=min_per,
                         max_per=max_per,
                         vel_disp=vel_disp,
                         vel_err=vel_err,
                         seed=44)
    for binary_model in [False, True]:
        pref = pref0 + {False: 'nbin', True: 'ybin'}[binary_model]
        with mp.Pool(36) as poo:
            R = []
            for i in range(Nstars):
                cur_dat = S[i]
                cur_truep = truep[i]
                args = (cur_dat[0], cur_dat[1], cur_dat[2], binary_model,
                        min_per, max_per, i)
                R.append((i, poo.apply_async(posterior,
                                             args), cur_dat, cur_truep))
            for cur_i, cur_r, cur_dat, cur_true in R:
                cur_samp, cur_logz = cur_r.get()
                idlsave.save(f'{pref}_{cur_i:05d}.psav',
                             'dat, samp, logz, truep', cur_dat, cur_samp,
                             cur_logz, cur_true)
