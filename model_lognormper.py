import numpy as np
import dynesty
import scipy.special
import multiprocessing as mp
import glob
import sys
from idlsave import idlsave  # importing sergey's idlsave

# Rachel's Path things
import os

# HOME = os.environ["HOME"] + '/'
# VERA = HOME  # +'Research/Vera/'
VERA = os.getcwd() + '/'
# os.sys.path.append(VERA + 'Research/')
# from binary_bayes.utils import TruncatedLogNormal

VREF = 30

DISP_PRIOR = 100  # width of gaussian prior on velocity


def make_samp(Nstars,
              Npt,
              Nsamp,
              dt=5,
              v0=0,
              bin_frac=0.5,
              min_per=0.001,
              max_per=10,
              mean_logper=2.4,
              std_logper=2.28,
              vel_disp=5,
              vel_err=0.5,
              seed=44):
    rng = np.random.default_rng(seed)

    per = 10**(rng.normal(mean_logper, std_logper, size=Nstars * 100))
    per = per[(per > min_per) & (per < max_per)][:Nstars]

    # per = 10**rng.uniform(np.log10(min_per), np.log10(max_per), size=Nstars)
    phase = rng.uniform(0, 2 * np.pi, size=Nstars)
    v0s = v0 + rng.normal(size=Nstars) * vel_disp
    cosi = rng.uniform(0, 1, size=Nstars)
    sini = np.sqrt(1 - cosi**2)
    amp0 = VREF / per**(1. / 3) * sini
    res = []
    is_bin = (rng.uniform(0, 1, size=Nstars) < bin_frac).astype(int)
    for i in range(Nstars):
        ts = rng.uniform(0, dt, size=Npt)
        v = (v0s[i] +
             amp0[i] * is_bin[i] * np.sin(2 * np.pi / per[i] * ts - phase[i]) +
             rng.normal(size=Npt) * vel_err)
        ev = v * 0 + vel_err
        res.append([ts, v, ev])
    truep = np.array([v0s, per, phase, sini, is_bin]).T
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
        x1[2] = x[2] * 2 * np.pi  # phase
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
                   print_progress=print_progress)  # , maxbatch=1)
    # for i in range(10):
    #    dns.add_batch(mode='full')
    res = dns.results.samples_equal()
    logz = dns.results.logz[-1]
    return res, logz


class si:
    cache_per = None
    cache_logz = None
    cache_logz_nb = None


def hierarch_perbf_like(p,
                        prefix,
                        prefix_nb,
                        mult=-1,
                        nsamp=1000,
                        seed=12,
                        min_per=0.1,
                        max_per=10):
    #                        stdper=0.2):
    """
    Hierarchical likelihood
    """
    # meanper, binfrac = p
    meanper, stdper, binfrac, meanvel, sigvel = p

    if binfrac >= 1 or binfrac <= 0 or np.abs(
            meanper) > 5 or stdper < 0 or stdper > 5 or sigvel < 0:
        return -1e100 * mult

    # load both bin and nb samps and logz's
    nsamp0 = 10000
    if si.cache_logz is None:
        persamp = []
        velsamp = []
        evid = []
        for _ in sorted(glob.glob(prefix + 'bin*psav')):
            temp = idlsave.restore(_, 'samp, logz')
            persamp.append(temp[0][:, 1][:nsamp0])
            velsamp.append(temp[0][:, 0][:nsamp0])
            evid.append(temp[1])
        si.cache_per = np.array(persamp)  # store binary period samples
        si.cache_vel = np.array(velsamp)  # store binary period samples
        si.cache_logz = np.array(evid)  # store binary evidence Z

        ARR = si.cache_per
        rng = np.random.default_rng(seed)
        permut = rng.integers(nsamp0, size=(ARR.shape[0], nsamp))
        ARR = ARR[np.arange(ARR.shape[0])[:, None] + permut * 0, permut]
        si.PER = si.cache_per[np.arange(ARR.shape[0])[:, None] + permut * 0,
                              permut]
        si.VEL_bin = si.cache_vel[np.arange(ARR.shape[0])[:, None] +
                                  permut * 0, permut]
    PER = si.PER
    VEL_bin = si.VEL_bin
    if si.cache_logz_nb is None:
        evid = []
        velsamp = []
        for _ in sorted(glob.glob(prefix_nb + 'nb*psav')):
            temp = idlsave.restore(_, 'samp, logz')
            evid.append(temp[1])
            velsamp.append(temp[0][:, 0][:nsamp0])
        si.cache_logz_nb = np.array(evid)  # store non-binary evidence Z
        si.cache_vel = np.array(velsamp)
        si.VEL_nbin = si.cache_vel[np.arange(ARR.shape[0])[:, None] +
                                   permut * 0, permut]
    VEL_nbin = si.VEL_nbin
    # likelihood part

    # fiducial per prior
    pi0_per = scipy.stats.loguniform(min_per, max_per).logpdf(PER)

    # normalizationtruncated lognorm
    NN = scipy.stats.norm(meanper, stdper)

    NNrv = scipy.stats.norm(meanvel, sigvel)
    NNrv0 = scipy.stats.norm(0, DISP_PRIOR)
    lrat_bin = NNrv.logpdf(VEL_bin) - NNrv0.logpdf(VEL_bin)
    lrat_nbin = NNrv.logpdf(VEL_nbin) - NNrv0.logpdf(VEL_nbin)

    pernorm = NN.cdf(np.log10(max_per)) - NN.cdf(np.log10(min_per))
    model_per = NN.logpdf(np.log10(PER)) - np.log(pernorm) - np.log(
        PER * np.log(10))

    # prior ratio
    perpr1 = scipy.special.logsumexp(model_per - pi0_per + lrat_bin,
                                     axis=1) - np.log(nsamp)
    perpr2 = scipy.special.logsumexp(lrat_nbin, axis=1) - np.log(nsamp)

    like1 = perpr1 + si.cache_logz + np.log(binfrac)
    like2 = perpr2 + si.cache_logz_nb + np.log(1 - binfrac)
    ret = np.logaddexp(like1, like2).sum(axis=0) * mult
    if not np.isfinite(ret):
        print('oops', p, ret, pernorm)
        ret = mult * (-1e100)
    return ret


if __name__ == '__main__':
    binary_model = bool(int(sys.argv[1]))

    Nstars = 2000
    Npt = 4
    vel_err = 0.5
    v0 = 12
    Nsamp = 1000

    vel_disp = 10  # DISP_PRIOR
    mean_logper = 0.5
    std_logper = 0.2
    min_per = 0.1
    max_per = 10
    bin_frac = 0.6

    seed = 123
    path = VERA + 'test_data_%d_%d/' % (Npt, seed)
    try:
        os.mkdir(path)
    except OSError:
        pass
    print(path, mean_logper, std_logper, vel_disp, bin_frac)

    # binary_model = False

    if binary_model:
        pref = path + 'bin'
    else:
        pref = path + 'nb'

    print('Make curves \n')
    S, truep = make_samp(Nstars,
                         Npt,
                         Nsamp,
                         dt=5,
                         v0=v0,
                         bin_frac=bin_frac,
                         min_per=min_per,
                         max_per=max_per,
                         mean_logper=mean_logper,
                         std_logper=std_logper,
                         vel_disp=vel_disp,
                         vel_err=vel_err,
                         seed=seed)
    print('sampling \n')
    with mp.Pool(mp.cpu_count()) as poo:
        R = []
        for i in range(Nstars):
            cur_dat = S[i]
            cur_truep = truep[i]
            args = (cur_dat[0], cur_dat[1], cur_dat[2], binary_model, min_per,
                    max_per, i)
            R.append((i, poo.apply_async(posterior, args), cur_dat, cur_truep))
        for cur_i, cur_r, cur_dat, cur_true in R:
            cur_samp, cur_logz = cur_r.get()
            idlsave.save(f'{pref}_{cur_i:05d}.psav', 'dat, samp, logz, truep',
                         cur_dat, cur_samp, cur_logz, cur_true)
