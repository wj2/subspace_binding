
import numpy as np
import scipy.stats as sts

import composite_tangling.code_creation as cc
import composite_tangling.code_analysis as ca

def get_corr_rate_n_stim(n_stims, n_draws, trades, **kwargs):
    corr = np.zeros((len(n_stims), len(trades), n_draws))
    for i, n_stim in enumerate(n_stims):
        out = get_corr_rate(n_stim, n_draws, trades, **kwargs)
        corr[i] = out
    return corr

def get_corr_rate(n_stim, n_draws, trades, **kwargs):
    corr = np.zeros((len(trades), n_draws))
    for i, t in enumerate(trades):
        out = construct_multi_responses(n_stim, n_draws, t, **kwargs)
        corr[i] = out[-1]
    return corr

def pwr_range_corr(corr, pwrs, n_feats, n_vals, sigma=1, n_stim=2):
    out_gen = np.zeros(len(pwrs))
    out_bind = np.zeros(len(pwrs))
    corr = np.array([corr])
    for i, pwr in enumerate(pwrs):    
        out_gen[i] = vector_corr_ccgp(pwr, corr, sigma=sigma)[1]
        out_bind[i] = vector_corr_swap(pwr, n_feats, n_vals, corr,
                                       sigma=sigma)[1]
    return out_bind, out_gen

def vector_corr_ccgp(sum_d2, trades, sigma=1, sem=0):
    dl = np.sqrt(sum_d2*trades)
    dn = np.sqrt(sum_d2*(1 - trades))
    dn = np.sqrt(dn**2 + sem**2)
    r = dl**2/(dl**2 + dn**2)
    err = sts.norm(0, 1).cdf(-dl**2/(np.sqrt(dl**2 + dn**2)*2*sigma))
    return r, err

def vector_corr_swap(sum_d2, n_feats, n_vals, trades, sigma=1,
                     n_stim=2, sem=0):
    dl = np.sqrt(trades*sum_d2)
    dn = np.sqrt((1 - trades)*sum_d2)
    r = dl**2/(dl**2 + dn**2 + sem**2)
    supply_ds = (dl, dn)
    te, etypes = ca.compute_composite_hamming_theory(sum_d2, n_feats, n_vals,
                                                     supply_ds=supply_ds,
                                                     noise_var=sigma,
                                                     ret_types=True,
                                                     mean=True, n_stim=n_stim)
    return r, etypes[2]

def construct_multi_responses(n_stim, n_draws, trade, total_power=100, n_feats=2,
                              n_values=5, n_neurs=1000):
    code = cc.make_code(trade, total_power, n_feats, n_values, n_neurs)
    stim, reps = code.sample_stim_reps(n_draws, n_stim)
    stim_combs, probs = code.get_posterior_probability(reps, n_stim)
    max_ind = np.argmax(probs, axis=1)
    dec_stim = stim_combs[max_ind]
    corr = hamming_set(stim, dec_stim)
    return stim, probs, stim_combs, dec_stim, corr

def get_n_most_likely(probs, stim_combs, n=10):
    s_args = np.argsort(probs, axis=1)
    sc_sorted = np.zeros((probs.shape[0], n,) + stim_combs.shape[1:])
    p_sorted = np.zeros((probs.shape[0], n))
    for i in range(probs.shape[0]):
        p_sorted[i] = probs[i, s_args[i]][-n:]
        sc_sorted[i] = stim_combs[s_args[i, -n:]]
    return p_sorted, sc_sorted

def get_ccgp_error_tradeoff_theory(pwrs, n_feats, n_values, n_stim, tradeoff,
                                   hamming_reps=1000, ccgp_reps=50):
    out = ca.compute_composite_hamming_theory(pwrs**2, n_feats, n_values, 
                                              n_stim=n_stim, tradeoff=tradeoff,
                                              ret_types=True)
    err_theory, ts = out
    out = ca.ccgp_error_rate(pwrs**2*tradeoff, (1 - tradeoff)*pwrs**2, n_feats,
                             n_values)
    gen_theory, _, _ = out
    return err_theory, gen_theory, ts

def get_ccgp_error_tradeoff_empirical(pwrs, n_feats, n_values, n_stim, tradeoff,
                                      hamming_reps=1000, ccgp_reps=50):
    err_emp = ca.compute_composite_hamming_empirical(pwrs**2, n_feats, n_values,
                                                     tradeoff=tradeoff,
                                                     n_stim=n_stim,
                                                     n_samps=hamming_reps)
    gen_emp = ca.compute_composite_ccgp_empirical(pwrs**2, n_feats, n_values,
                                                  tradeoff=tradeoff,
                                                  n_samps=ccgp_reps)
    return err_emp, gen_emp

def get_ccgp_error_dep(pwrs, tradeoffs, n_feats, n_values, n_stim,
                       hamming_reps=500, ccgp_reps=50, empirical=True,
                       theory=True):
    err_emp = np.zeros((len(tradeoffs), len(pwrs), hamming_reps))
    err_theor = np.zeros((len(tradeoffs), len(pwrs)))
    swap_theor = np.zeros((len(tradeoffs), len(pwrs)))
    gen_emp = np.zeros((len(tradeoffs), len(pwrs), ccgp_reps))
    gen_theor = np.zeros((len(tradeoffs), len(pwrs)))

    for i, tradeoff in enumerate(tradeoffs):
        if theory:
            out = get_ccgp_error_tradeoff_theory(pwrs, n_feats, n_values, n_stim,
                                                 tradeoff,
                                                 hamming_reps=hamming_reps,
                                                 ccgp_reps=ccgp_reps)
            err_theor[i] = np.mean(out[0], axis=0)
            gen_theor[i] = out[1]
            swap_theor[i] = np.mean(out[2][-1], axis=0)
        if empirical:
            out = get_ccgp_error_tradeoff_empirical(pwrs, n_feats, n_values,
                                                    n_stim, tradeoff,
                                                    hamming_reps=hamming_reps,
                                                    ccgp_reps=ccgp_reps)
            err_emp[i] = out[0].T
            gen_emp[i] = out[1].T
    return err_theor, err_emp, swap_theor, gen_theor, gen_emp
