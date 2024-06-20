
import itertools as it
import functools as ft
import os
import pickle

import numpy as np
import sklearn.preprocessing as skp
import sklearn.linear_model as sklm
import sklearn.svm as skc
import sklearn.model_selection as skms
import sklearn.metrics as skm
import sklearn.decomposition as skd
import sklearn.utils as sku
import scipy.stats as sts
import scipy.io as sio
import scipy.optimize as sopt
import scipy.linalg as sla
import statsmodels.api as sm
import statsmodels.formula.api as sm_formula
import pandas as pd
import scipy.special as ss
import sklearn.manifold as skman

import general.utility as u
import general.neural_analysis as na
import general.data_io as gio
import general.stan_utility as  su
import multiple_representations.theory as mrt
import multiple_representations.auxiliary as mraux
import composite_tangling.code_creation as cc
import composite_tangling.code_analysis as ca
import multiple_representations.direct_theory as mrdt

import rsatoolbox as rsa

# import oak.model_utils as mu

visual_regions = ('VISp', 'VISI', 'VISpm', 'VISam', 'VISrl', 'VISa')
choice_regions = ('MOs', 'PL', 'ILA', 'ORB', 'MOp', 'SSp', 'SCm', 'MRN')



def linear_inverse_decoding(theta, sig, stim=None, n_samps=1000, distrib=None):
    rng = np.random.default_rng()
    a = np.array([[1, 0]]).T
    b = np.array([[np.cos(theta), np.sin(theta)]]).T

    A = np.concatenate((a, b), axis=1)

    if distrib is None:
        distrib = rng.normal
    if stim is None:
        stim = distrib(size=(n_samps, 2))
    rep = stim @ A.T
    A_lst = np.linalg.lstsq(rep, stim)[0]
    rs = a.T @ b

    err_theor = (2*sig**2)/(np.sin(theta)**2)
    rep = stim @ A.T + rng.normal(0, sig, size=stim.shape)
    dec = rep @ A_lst
    errs = np.mean(np.sum((dec - stim)**2, axis=1))
    return stim, rep, dec, A, errs, err_theor



def normalize_embedding_power(rdm_dict, **kwargs):
    rdm_dict_new = {}
    for r, (rdm_arr, rdm_list, ambig_dists, pop_mat) in rdm_dict.items():
        factor = compute_embedding_power_factor(rdm_arr, **kwargs)
        rdm_arr_new = rdm_arr / factor
        ambig_dists_new = {}
        for k, v in ambig_dists.items():
            ambig_dists_new[k] = v / factor
        rdm_dict_new[r] = (rdm_arr_new, rdm_list, ambig_dists_new, pop_mat)
    return rdm_dict_new


def compute_embedding_power_factor(rdm_arr, target_pwr=1, single_factor=True):
    n_dims = rdm_arr.shape[1]
    if single_factor:
        m = skman.MDS(n_dims, dissimilarity="precomputed", normalized_stress="auto")
        embed = m.fit_transform(np.mean(rdm_arr, axis=0))
        factors = np.sqrt(np.sum(np.var(embed, axis=0)))
    else:
        factors = np.zeros(rdm_arr.shape)
        for i, rdm in enumerate(rdm_arr):
            m = skman.MDS(n_dims, dissimilarity="precomputed", normalized_stress="auto")
            embed = m.fit_transform(rdm)
            factors[i] = np.sqrt(np.sum(np.var(embed, axis=0)))
    return factors
        

def estimate_distances(pwrs, tradeoffs, n_feats, n_vals, n_units=100, n_samples=320,
                       n_boots=100, sg_resamples=1):
    est_l = np.zeros((len(pwrs), len(tradeoffs), n_boots, sg_resamples))
    est_nl = np.zeros_like(est_l)
    est_sigma = np.zeros_like(est_l)
    true_l = np.zeros((len(pwrs), len(tradeoffs)))
    true_nl = np.zeros_like(true_l)
    pwrs = np.array(pwrs)**2

    rng = np.random.default_rng()
        
    for (i, j) in u.make_array_ind_iterator(true_l.shape):
        dl_t = ca.get_linear_theory_distance(
            pwrs[i]*tradeoffs[j], n_feats, n_vals
        )
        dn_t = ca.get_mixed_theory_distance(
            pwrs[i]*(1 - tradeoffs[j]), n_feats, n_vals
        )
        true_l[i, j] = dl_t
        true_nl[i, j] = dn_t
        for k in range(n_boots):
            code = cc.make_code(tradeoffs[j], pwrs[i], n_feats, n_vals, n_units)
            stim, reps = code.sample_stim_reps(n_samples)

            stim = np.squeeze(stim)
            u_stim = np.unique(stim, axis=0)
            stim_groups = {}
            for us in u_stim:
                mask = np.all(stim == us, axis=1)
                use_reps = reps[mask].T
                if sg_resamples > 1:
                    ur_groups = np.zeros((sg_resamples,) + use_reps.shape)
                    for l in range(sg_resamples):
                        inds = rng.choice(use_reps.shape[1], use_reps.shape[1])
                        ur_groups[l] = use_reps[..., inds]
                    stim_groups[tuple(us)] = np.expand_dims(ur_groups, -1)
                else:
                    stim_groups[tuple(us)] = np.expand_dims(use_reps, (0, -1))

            if sg_resamples > 1:
                out = mrdt.combined_ccgp_bind_est(
                    (stim_groups[(0, 0)], stim_groups[(0, 1)]),
                    (stim_groups[(1, 0)], stim_groups[(1, 1)]),
                )
                bind_est, gen_est, (d_l, d_n, sigma, sem) = out
                est_l[i, j, k] = d_l
                est_nl[i, j, k] = np.mean(d_n, axis=1)*np.sqrt(2)
                est_sigma[i, j, k] = sigma
            else:
                out = mrdt.direct_ccgp_bind_est_pops(
                    (stim_groups[(0, 0)], stim_groups[(0, 1)]),
                    (stim_groups[(1, 0)], stim_groups[(1, 1)]),
                    test_prop=0, empirical=False,
                )
                bind_est, gen_est, (d_l, d_n, sigma, sem) = out
                est_l[i, j, k] = np.mean(d_l)
                est_nl[i, j, k] = np.mean(d_n, axis=1)*np.sqrt(2)
                est_sigma[i, j, k] = np.mean(sigma)
    out = {
        'dl': (true_l, est_l),
        'dn': (true_nl, est_nl),
        'sigma': est_sigma,
    }
    return out


def estimate_submanifolds(data, tbeg, tend,
                          dead_perc=30, winsize=300, tstep=300,
                          pop_resamples=20, kernel='linear',
                          dec_tzf='Offer 2 on',
                          early_tzf='Offer 1 on',
                          decode_var='subj_ev',
                          min_trials=80, pre_pca=.99,
                          shuffle_trials=True, c1_targ=2,
                          c2_targ=3, f1_mask=None, f2_mask=None,
                          use_split_dec=None,
                          max_trials=None,
                          regions=None,
                          **kwargs):
    o1_target = '{} offer 1'.format(decode_var)
    o2_target = '{} offer 2'.format(decode_var)
    out = _compute_masks(data, o1_target, o2_target, dead_perc=dead_perc,
                         use_split_dec=use_split_dec, c1_targ=c1_targ,
                         c2_targ=c2_targ)
    mask_o1_l, mask_o1_h, mask_o2_l, mask_o2_h = out

    mask_o1_left = data['side of offer 1 (Left = 1 Right =0)'] == 1
    mask_o1_right = data['side of offer 1 (Left = 1 Right =0)'] == 0
    mask_o2_left = data['side of offer 1 (Left = 1 Right =0)'] == 0
    mask_o2_right = data['side of offer 1 (Left = 1 Right =0)'] == 1

    mask_o1_h_left = mask_o1_h.rs_and(mask_o1_left)
    mask_o1_l_left = mask_o1_l.rs_and(mask_o1_left)
    mask_o1_h_right = mask_o1_h.rs_and(mask_o1_right)
    mask_o1_l_right = mask_o1_l.rs_and(mask_o1_right)

    mask_o2_h_left = mask_o2_h.rs_and(mask_o2_left)
    mask_o2_l_left = mask_o2_l.rs_and(mask_o2_left)
    mask_o2_h_right = mask_o2_h.rs_and(mask_o2_right)
    mask_o2_l_right = mask_o2_l.rs_and(mask_o2_right)

    masks_o1 = (
        mask_o1_h_left,
        mask_o1_l_left,
        mask_o1_h_right,
        mask_o1_l_right,
    )
    masks_o2 = (
        mask_o2_h_left,
        mask_o2_l_left,
        mask_o2_h_right,
        mask_o2_l_right,
    )

    masks_all = masks_o2 + masks_o1 + masks_o1
    tzfs_all = ((dec_tzf,)*len(masks_o2) + (dec_tzf,)*len(masks_o1)
                + (early_tzf,)*len(masks_o1))
    out = data.make_pseudo_pops(winsize, tbeg, tend, tstep, *masks_all,
                                tzfs=tzfs_all,
                                min_trials=min_trials,
                                resamples=pop_resamples,
                                regions=regions)
    xs, pops_all = out

    outs_all = []
    for i in range(pop_resamples):
        pops_all_i = tuple(p[i] for p in pops_all)
        pops_pre = mrdt.rsa_preproc(
            pops_all_i, (),
            norm=True, pre_pca=pre_pca, accumulate_time=True,
            ret_out_pops=True,
        )
        outs_all.append(pops_pre)

    return outs_all, xs


def scale_matrix(mat, ret_trs=False):
    mu = np.mean(mat, axis=0, keepdims=True)
    mat = mat - mu
    tr = np.trace(mat @ mat.T)
    out = mat / np.sqrt(tr)
    if ret_trs:
        def trs(x):
            return (x - mu) / np.sqrt(tr)
        out = (mat, trs)
    return out


def make_all_cond_masks(
        data,
        decode_var="subj_ev",
        dead_perc=30,
        use_split_dec=None,
        c1_targ=2,
        c2_targ=3,
        correct_only=False,
        mask_var=None,
        mask_func=None,
):
    suff1 = " offer 1"
    suff2 = " offer 2"
    o1_target = '{} offer 1'.format(decode_var)
    o2_target = '{} offer 2'.format(decode_var)
    m1, m2 = _make_var_masks(data, mask_var, mask_func, suff1, suff2)
    out = _compute_masks(data, o1_target, o2_target, dead_perc=dead_perc,
                         use_split_dec=use_split_dec, c1_targ=c1_targ,
                         dec_mask=m1, gen_mask=m2, c2_targ=c2_targ)
    mask_o1_h, mask_o1_l, mask_o2_h, mask_o2_l = out

    mask_o1_left = data['side of offer 1 (Left = 1 Right =0)'] == 1
    mask_o1_right = data['side of offer 1 (Left = 1 Right =0)'] == 0
    mask_o2_left = data['side of offer 1 (Left = 1 Right =0)'] == 0
    mask_o2_right = data['side of offer 1 (Left = 1 Right =0)'] == 1

    mask_o1_l_left = mask_o1_l.rs_and(mask_o1_left)
    mask_o1_l_right = mask_o1_l.rs_and(mask_o1_right)
    mask_o2_l_left = mask_o2_l.rs_and(mask_o2_left)
    mask_o2_l_right = mask_o2_l.rs_and(mask_o2_right)
    mask_o1_h_left = mask_o1_h.rs_and(mask_o1_left)
    mask_o1_h_right = mask_o1_h.rs_and(mask_o1_right)
    mask_o2_h_left = mask_o2_h.rs_and(mask_o2_left)
    mask_o2_h_right = mask_o2_h.rs_and(mask_o2_right)

    mask_ll_leftright = mask_o1_l_left.rs_and(mask_o2_l_right)
    mask_lh_leftright = mask_o1_l_left.rs_and(mask_o2_h_right)
    mask_hl_leftright = mask_o1_h_left.rs_and(mask_o2_l_right)
    mask_hh_leftright = mask_o1_h_left.rs_and(mask_o2_h_right)

    mask_ll_rightleft = mask_o1_l_right.rs_and(mask_o2_l_left)
    mask_lh_rightleft = mask_o1_l_right.rs_and(mask_o2_h_left)
    mask_hl_rightleft = mask_o1_h_right.rs_and(mask_o2_l_left)
    mask_hh_rightleft = mask_o1_h_right.rs_and(mask_o2_h_left)

    if correct_only:
        ch_left_mask = data['Choice left (==1) or Right (==0)'] == 1
        ch_right_mask = data['Choice left (==1) or Right (==0)'] == 0

        mask_lh_leftright = mask_lh_leftright.rs_and(ch_right_mask)
        mask_lh_rightleft = mask_lh_rightleft.rs_and(ch_left_mask)
        mask_hl_leftright = mask_hl_leftright.rs_and(ch_left_mask)
        mask_hl_rightleft = mask_hl_rightleft.rs_and(ch_right_mask)

    masks_t2 = (
        mask_ll_leftright,
        mask_lh_leftright,
        mask_hl_leftright,
        mask_hh_leftright,
        mask_ll_rightleft,
        mask_lh_rightleft,
        mask_hl_rightleft,
        mask_hh_rightleft,
    )
    return masks_t2

def estimate_rdm_conditions(data, tbeg, tend,
                            dead_perc=30, winsize=300, tstep=300,
                            pop_resamples=20, kernel='linear',
                            dec_tzf='Offer 2 on',
                            decode_var='subj_ev',
                            min_trials=80, pre_pca=.99,
                            shuffle_trials=True, c1_targ=2,
                            c2_targ=3, f1_mask=None, f2_mask=None,
                            use_split_dec=None,
                            max_trials=None,
                            regions=None,
                            min_neurs=15,
                            use_pseudo=True,
                            correct_only=False,
                            mask_var=None,
                            mask_func=None,
                            subsample_neurons=None,
                            **kwargs):
    masks_t2 = make_all_cond_masks(
        data,
        decode_var=decode_var,
        c1_targ=c1_targ,
        c2_targ=c2_targ,
        dead_perc=dead_perc,
        use_split_dec=use_split_dec,
        correct_only=correct_only,
        mask_var=mask_var,
        mask_func=mask_func,
    )
    if use_pseudo:
        out = data.make_pseudo_pops(winsize, tbeg, tend, tstep, *masks_t2,
                                    tzfs=(dec_tzf,)*len(masks_t2),
                                    min_trials=min_trials,
                                    resamples=pop_resamples,
                                    regions=regions,
                                    subsample_neurons=subsample_neurons)
        xs, cond_pops_t2 = out
        print(regions)
        print(list(cond_pops_t2[i][0].shape for i in range(len(cond_pops_t2))))
    else:
        session_mask = data['n_neurs'] > min_neurs
        data_n_neurs = data.session_mask(session_mask)
        out = data_n_neurs._get_dec_pops(winsize, tbeg, tend, tstep, *masks_t2,
                                         tzfs=(dec_tzf,)*len(masks_t2))
        xs, cond_pops_t2 = out

        pop_resamples = len(cond_pops_t2[0])
        # cond_pops_t2 = []
        # for i in range(pop_resamples):
        #     cond_pops_t2.append(
        #         tuple(cond_pops_t2_org[j][i] for j in range(len(cond_pops_t2_org)))
        #     )

    # print(cond_pops_t2[0].shape)
    n_stim = len(cond_pops_t2)
    rdm_mat = np.zeros((pop_resamples, n_stim, n_stim))
    rdm_list = []
    pop_mat = []
    for i in range(pop_resamples):
        cp_i = tuple(cp[i] for cp in cond_pops_t2)
        pops_pre = mrdt.rsa_preproc(
            cp_i, (), norm=True, pre_pca=pre_pca, accumulate_time=True,
            ret_out_pops=True,
        )
        if max_trials is None:
            max_trials_i = np.min(list(p_i.shape[-1] for p_i in pops_pre))
        else:
            max_trials_i = max_trials
        pops_pre = tuple(p_i[..., :max_trials_i] for p_i in pops_pre)
        # print(i, pops_pre[0].shape)
        pops_comb = np.concatenate(pops_pre, axis=1).T

        if use_pseudo:
            pop_mat.append(np.mean(np.stack(pops_pre, axis=0), axis=2))

        pops = np.squeeze(pops_comb)

        p_mask = np.var(pops, axis=0) > 0
        pops = pops[:, p_mask]

        stim = list((str(i),)*p_ind.shape[1]
                    for i, p_ind in enumerate(pops_pre))
        stim = np.concatenate(stim)

        data = rsa.data.Dataset(pops, obs_descriptors={'stimulus': stim})

        # dof = data.measurements.shape[0] - 1
        # noise = rsa.data.noise.prec_from_measurements(data, 'stimulus',
        #                                               dof=None,
        #                                               method='shrinkage_diag')
        # print(noise)
        rdm = rsa.rdm.calc_rdm(data, descriptor='stimulus', noise=None,
                               method='crossnobis')

        rdm_mat[i] = rdm.get_matrices()[0]
        rdm_list.append(rdm)

    ambig_dists = get_cond_pairs(rdm_mat)
    return rdm_mat, rdm_list, ambig_dists, pop_mat


default_ambiguities = {
    # different side-order
    # value is low for all stimuli
    "low": ((0, 4),),
    # trial has the same side-order
    # but the values switch to different sides
    # (i.e., left-first + left-high vs left-first + left-low)
    "same order, side-values flip": ((1, 2), (5, 6)),
    # trial has different side-order
    # and the sides switch values
    # (i.e., left-first + left-high vs. right-first + left-low)
    "order flips, side-values flip": ((1, 5), (2, 6)),
    # trial has different side-order
    # but the sides keep their values
    # (i.e., left-first + left-high vs. right-first + left-high)
    "order flips, side-values same": ((1, 6), (2, 5)),
    # different side-order
    # value is high for all stimuli
    "high": ((3, 7),),
    "offer 1 value": ((0, 2), (1, 3), (4, 6), (5, 7)),
    "offer 2 value": ((0, 1), (2, 3), (4, 5), (6, 7)),
}

def get_cond_pairs(rdm_mat, ambiguities=None):
    if ambiguities is None:
        ambiguities = default_ambiguities
    ambig_dists = {}
    for k, inds in ambiguities.items():
        ambig_dists[k] = np.mean(
            list(rdm_mat[:, ind[0], ind[1]] for ind in inds), axis=0
        )
    return ambig_dists


default_full_schema = np.array(
    [  # val, side, time, (high x left), (low x left),
        [0,
         0, 1, 0, 1,  # low on left and low on right
         0, 1, 0, 1,  # low first and low second
         1, 0, 0, 0, 0, 0, 0, 0],
        [1,
         0, 1, 1, 0,  # low on left and high on right
         0, 1, 1, 0,  # low first and high second
         0, 1, 0, 0, 0, 0, 0, 0],
        [1,
         1, 0, 0, 1,  # high on left and low on right
         1, 0, 0, 1,  # high first and low second
         0, 0, 1, 0, 0, 0, 0, 0],
        [2,
         1, 0, 1, 0,  # high on left and high on right
         1, 0, 1, 0,  # high first and high second
         0, 0, 0, 1, 0, 0, 0, 0],
        [0,
         0, 1, 0, 1,  # low on left and low on right
         0, 1, 0, 1,  # low first and low second
         0, 0, 0, 0, 1, 0, 0, 0],
        [1,
         1, 0, 0, 1,  # high on left and low on right
         0, 1, 1, 0,  # low first and high second
         0, 0, 0, 0, 0, 1, 0, 0],
        [1,
         0, 1, 1, 0,  # low on left and high on right
         1, 0, 0, 1,  # high first and low second
         0, 0, 0, 0, 0, 0, 1, 0],
        [2,
         1, 0, 1, 0,  # high on left and high on right
         1, 0, 1, 0,  # high first and high second
         0, 0, 0, 0, 0, 0, 0, 1],
    ]
)


default_full_schema = np.array(
    [  # val, side, time, (high x left), (low x left),
        [0,
         0, 0,  # low on left and low on right
         0, 0,  # low first and low second
         1, 0, 0, 0, 0, 0, 0, 0],
        [1,
         0, 1,  # low on left and high on right
         0, 1,  # low first and high second
         0, 1, 0, 0, 0, 0, 0, 0],
        [1,
         1, 0,  # high on left and low on right
         1, 0,  # high first and low second
         0, 0, 1, 0, 0, 0, 0, 0],
        [2,
         1, 1,  # high on left and high on right
         1, 1,  # high first and high second
         0, 0, 0, 1, 0, 0, 0, 0],
        [0,
         0, 0,  # low on left and low on right
         0, 0,  # low first and low second
         0, 0, 0, 0, 1, 0, 0, 0],
        [1,
         1, 0,  # high on left and low on right
         0, 1,  # low first and high second
         0, 0, 0, 0, 0, 1, 0, 0],
        [1,
         0, 1,  # low on left and high on right
         1, 0,  # high first and low second
         0, 0, 0, 0, 0, 0, 1, 0],
        [2,
         1, 1,  # high on left and high on right
         1, 1,  # high first and high second
         0, 0, 0, 0, 0, 0, 0, 1],
    ]
)


def decompose_full_rdm(rdm_mat, schema=default_full_schema):
    # ds_init = 5*np.abs(sts.norm(0, 1).rvs(4))
    # ds_init = 5*np.abs(sts.norm(0, 1).rvs(schema.shape[1]))
    ds_init = 5*np.abs(sts.norm(0, 1).rvs(6))

    def _make_mat(dists):
        # d_v, d_vs, d_vo, d_vso = dists
        d_v, d_vs, d_vo, d_vso = dists[:1], dists[1:3], dists[3:5], dists[5:]
        new_dmat = np.zeros_like(rdm_mat)
        for (i, j) in it.combinations(range(rdm_mat.shape[0]), 2):
            sd = np.abs(schema[i] - schema[j])
            # print(i, j, sd)
            d_ij = (np.sum(sd[:1]*d_v**2)
                    + np.sum(sd[1:3]*d_vs**2)
                    + np.sum(sd[3:5]*d_vo**2)
                    + np.sum(sd[5:]*d_vso**2))
            new_dmat[i, j] = d_ij
            new_dmat[j, i] = d_ij
        return np.sum((rdm_mat - new_dmat)**2)

    res = sopt.minimize(_make_mat, ds_init,
                        bounds=((0, None),)*len(ds_init))
    return res


def estimate_choice_corr(data, tbeg, tend,
                         dead_perc=30, winsize=300, tstep=300,
                         pop_resamples=20, kernel='linear',
                         dec_tzf='Offer 2 on',
                         decode_var='subj_ev',
                         min_trials=80, pre_pca=.99,
                         shuffle_trials=False, c1_targ=2,
                         c2_targ=3, f1_mask=None, f2_mask=None,
                         use_split_dec=None,
                         only_pairs=None,
                         pseudo=False,
                         n_folds=20,
                         test_prop=.1,
                         gap=5,
                         **kwargs):
    l_target = '{}_left'.format(decode_var)
    r_target = '{}_right'.format(decode_var)
    side_diffs = data[l_target] - data[r_target]
    l_opt = side_diffs > gap
    r_opt = side_diffs < -gap
    ch_left = data['Choice left (==1) or Right (==0)'] == 1
    ch_right = data['Choice left (==1) or Right (==0)'] == 0

    o1_target = '{} offer 1'.format(decode_var)
    o2_target = '{} offer 2'.format(decode_var)
    time_diffs = data[o1_target] - data[o2_target]
    o1_opt = time_diffs > gap
    o2_opt = time_diffs < -gap
    ch_o1 = data['choice offer 1 (==1) or 2 (==0)'] == 1
    ch_o2 = data['choice offer 1 (==1) or 2 (==0)'] == 0

    out_dict = {}
    mask_pairs = {
        "side choice":
        (((l_opt.rs_and(ch_left), r_opt.rs_and(ch_right)),
          (l_opt.rs_and(ch_right), r_opt.rs_and(ch_left))),
         1),
        "time choice":
        (((o1_opt.rs_and(ch_o1), o2_opt.rs_and(ch_o2)),
          (o1_opt.rs_and(ch_o2), o2_opt.rs_and(ch_o1))),
         1),
    }
    if only_pairs is not None:
        use_keys = only_pairs
    else:
        use_keys = mask_pairs.keys()
    for k in use_keys:
        ((dec_mask, targ_mask), min_factor) = mask_pairs[k]
        mask_c1, mask_c2 = dec_mask
        gen_mask_c1, gen_mask_c2 = targ_mask
        print(k)
        out1 = data.decode_masks(mask_c1, mask_c2, winsize, tbeg, tend, tstep,
                                 pseudo=pseudo, time_zero_field=dec_tzf,
                                 min_trials_pseudo=min_trials/min_factor,
                                 resample_pseudo=pop_resamples, ret_pops=True,
                                 shuffle_trials=shuffle_trials, pre_pca=pre_pca,
                                 decode_tzf=dec_tzf, decode_m1=gen_mask_c1,
                                 decode_m2=gen_mask_c2, n_folds=n_folds,
                                 test_prop=test_prop, **kwargs)
        out_dict[k] = out1

    return out_dict


def estimate_bhv_corr(
        data,
        tbeg,
        tend,
        dead_perc=30,
        winsize=300,
        tstep=300,
        pop_resamples=20,
        kernel='linear',
        dec_tzf='Offer 2 on',
        decode_var='subj_ev',
        min_trials=80,
        pre_pca=.99,
        shuffle_trials=False,
        c1_targ=2,
        c2_targ=3,
        f1_mask=None,
        f2_mask=None,
        use_split_dec=None,
        only_pairs=None,
        pseudo=False,
        only_main=True,
        n_folds=20,
        test_prop=.05,
        use_time=False,
        **kwargs
):
    l_target = '{}_left'.format(decode_var)
    r_target = '{}_right'.format(decode_var)
    out = _compute_masks(data, l_target, r_target, dead_perc=dead_perc,
                         use_split_dec=use_split_dec, c1_targ=c1_targ, c2_targ=c2_targ)
    mask_H_L, mask_L_L, mask_H_R, mask_L_R = out

    o1_target = '{} offer 1'.format(decode_var)
    o1_left_mask = data['side of offer 1 (Left = 1 Right =0)'] == 1
    o1_right_mask = data['side of offer 1 (Left = 1 Right =0)'] == 0
    out = _compute_masks(data, o1_target, o1_target, dead_perc=dead_perc,
                         use_split_dec=use_split_dec, dec_mask=o1_left_mask,
                         gen_mask=o1_right_mask, c1_targ=c1_targ, c2_targ=c2_targ)
    mask_o1_H_L, mask_o1_L_L, mask_o1_H_R, mask_o1_L_R = out

    o2_target = '{} offer 2'.format(decode_var)
    o2_left_mask = data['side of offer 1 (Left = 1 Right =0)'] == 0
    o2_right_mask = data['side of offer 1 (Left = 1 Right =0)'] == 1
    out = _compute_masks(data, o2_target, o2_target, dead_perc=dead_perc,
                         use_split_dec=use_split_dec, dec_mask=o2_left_mask,
                         gen_mask=o2_right_mask, c1_targ=c1_targ, c2_targ=c2_targ)
    mask_o2_H_L, mask_o2_L_L, mask_o2_H_R, mask_o2_L_R = out

    out = _compute_masks(data, o1_target, o2_target, dead_perc=dead_perc,
                         use_split_dec=use_split_dec, c1_targ=c1_targ, c2_targ=c2_targ)
    mask_H_o1, mask_L_o1, mask_H_o2, mask_L_o2 = out

    ch_left_mask = data['Choice left (==1) or Right (==0)'] == 1
    ch_right_mask = data['Choice left (==1) or Right (==0)'] == 0
    ch_o1_mask = data['choice offer 1 (==1) or 2 (==0)'] == 1
    ch_o2_mask = data['choice offer 1 (==1) or 2 (==0)'] == 0

    out_dict = {}
    mask_pairs = {
        'left-higher vs right-higher 1 corr': ((
            (mask_o1_L_L.rs_and(mask_o2_H_R).rs_and(ch_o2_mask),
             mask_o1_H_L.rs_and(mask_o2_L_R).rs_and(ch_o1_mask)),
            (mask_o1_H_R.rs_and(mask_o2_L_L).rs_and(ch_o1_mask),
             mask_o1_L_R.rs_and(mask_o2_H_L).rs_and(ch_o2_mask))),
                                        2),
        'left-higher vs right-higher 1 err': ((
            (mask_o1_L_L.rs_and(mask_o2_H_R).rs_and(ch_o2_mask),
             mask_o1_H_L.rs_and(mask_o2_L_R).rs_and(ch_o1_mask)),
            (mask_o1_H_R.rs_and(mask_o2_L_L).rs_and(ch_o2_mask),
             mask_o1_L_R.rs_and(mask_o2_H_L).rs_and(ch_o1_mask))),
                                        2),
        'left-higher vs right-higher 2 corr': ((
            (mask_o1_H_R.rs_and(mask_o2_L_L).rs_and(ch_o1_mask),
             mask_o1_L_R.rs_and(mask_o2_H_L).rs_and(ch_o2_mask)),
            (mask_o1_L_L.rs_and(mask_o2_H_R).rs_and(ch_o2_mask),
             mask_o1_H_L.rs_and(mask_o2_L_R).rs_and(ch_o1_mask))),
                                        2),
        'left-higher vs right-higher 2 err': ((
            (mask_o1_H_R.rs_and(mask_o2_L_L).rs_and(ch_o1_mask),
             mask_o1_L_R.rs_and(mask_o2_H_L).rs_and(ch_o2_mask)),
            (mask_o1_L_L.rs_and(mask_o2_H_R).rs_and(ch_o1_mask),
             mask_o1_H_L.rs_and(mask_o2_L_R).rs_and(ch_o2_mask))),
                                        2),
        'left-higher vs right-higher -- no order': (
            ((mask_L_L.rs_and(mask_H_R).rs_and(ch_right_mask),
              mask_H_L.rs_and(mask_L_R).rs_and(ch_left_mask)),
             (mask_L_L.rs_and(mask_H_R).rs_and(ch_left_mask),
              mask_H_L.rs_and(mask_L_R).rs_and(ch_right_mask))),
            1),
        'first-higher vs second-higher 1 corr': ((
            (mask_o1_L_L.rs_and(mask_o2_H_R).rs_and(ch_o2_mask),
             mask_o1_H_L.rs_and(mask_o2_L_R).rs_and(ch_o1_mask)),
            (mask_o1_L_R.rs_and(mask_o2_H_L).rs_and(ch_o2_mask),
             mask_o1_H_R.rs_and(mask_o2_L_L).rs_and(ch_o1_mask))),
                                        2),
        'first-higher vs second-higher 1 err': ((
            (mask_o1_L_L.rs_and(mask_o2_H_R).rs_and(ch_o2_mask),
             mask_o1_H_L.rs_and(mask_o2_L_R).rs_and(ch_o1_mask)),
            (mask_o1_L_R.rs_and(mask_o2_H_L).rs_and(ch_o1_mask),
             mask_o1_H_R.rs_and(mask_o2_L_L).rs_and(ch_o2_mask))),
                                        2),
        'first-higher vs second-higher 2 corr': ((
            (mask_o1_L_R.rs_and(mask_o2_H_L).rs_and(ch_o2_mask),
             mask_o1_H_R.rs_and(mask_o2_L_L).rs_and(ch_o1_mask)),
            (mask_o1_L_L.rs_and(mask_o2_H_R).rs_and(ch_o2_mask),
             mask_o1_H_L.rs_and(mask_o2_L_R).rs_and(ch_o1_mask)),
        ),
                                                 2),
        'first-higher vs second-higher 2 err': ((
            (mask_o1_L_R.rs_and(mask_o2_H_L).rs_and(ch_o2_mask),
             mask_o1_H_R.rs_and(mask_o2_L_L).rs_and(ch_o1_mask)),
            (mask_o1_L_L.rs_and(mask_o2_H_R).rs_and(ch_o1_mask),
             mask_o1_H_L.rs_and(mask_o2_L_R).rs_and(ch_o2_mask)),
        ),
                                                2),
        'first-higher vs second-higher -- no side': (
            ((mask_L_o1.rs_and(mask_H_o2).rs_and(ch_o2_mask),
              mask_H_o1.rs_and(mask_L_o2).rs_and(ch_o1_mask)),
             (mask_L_o1.rs_and(mask_H_o2).rs_and(ch_o1_mask),
              mask_H_o1.rs_and(mask_L_o2).rs_and(ch_o2_mask))),
            1),
    }

    if only_main:
        only_pairs = (
            'first-higher vs second-higher -- no side',
            'left-higher vs right-higher -- no order',
        )
    if only_pairs is not None:
        use_keys = only_pairs
    else:
        use_keys = mask_pairs.keys()
    for k in use_keys:
        ((dec_mask, targ_mask), min_factor) = mask_pairs[k]
        mask_c1, mask_c2 = dec_mask
        gen_mask_c1, gen_mask_c2 = targ_mask
        print(k)
        out1 = data.decode_masks(
            mask_c1,
            mask_c2,
            winsize,
            tbeg,
            tend,
            tstep,
            pseudo=pseudo,
            time_zero_field=dec_tzf,
            min_trials_pseudo=min_trials/min_factor,
            resample_pseudo=pop_resamples,
            ret_pops=True,
            shuffle_trials=shuffle_trials,
            pre_pca=pre_pca,
            decode_tzf=dec_tzf,
            decode_m1=gen_mask_c1,
            decode_m2=gen_mask_c2,
            n_folds=n_folds,
            test_prop=test_prop,
            n_jobs=1,
            use_time=use_time,
            **kwargs
        )
        out_dict[k] = out1

    return out_dict


def estimate_decoding_late(data, tbeg, tend,
                           dead_perc=30, winsize=300, tstep=300,
                           pop_resamples=20, kernel='linear',
                           dec_tzf='Offer 2 on',
                           decode_var='subj_ev',
                           min_trials=80, pre_pca=.99,
                           shuffle_trials=True, c1_targ=2,
                           c2_targ=3, f1_mask=None, f2_mask=None,
                           use_split_dec=None,
                           only_pairs=None,
                           **kwargs):
    l_target = '{}_left'.format(decode_var)
    r_target = '{}_right'.format(decode_var)
    out = _compute_masks(data, l_target, r_target, dead_perc=dead_perc,
                         use_split_dec=use_split_dec, c1_targ=c1_targ, c2_targ=c2_targ)
    mask_L_L, mask_H_L, mask_L_R, mask_H_R = out

    o1_target = '{} offer 1'.format(decode_var)
    o1_left_mask = data['side of offer 1 (Left = 1 Right =0)'] == 1
    o1_right_mask = data['side of offer 1 (Left = 1 Right =0)'] == 0
    out = _compute_masks(data, o1_target, o1_target, dead_perc=dead_perc,
                         use_split_dec=use_split_dec, dec_mask=o1_left_mask,
                         gen_mask=o1_right_mask, c1_targ=c1_targ, c2_targ=c2_targ)
    mask_o1_L_L, mask_o1_H_L, mask_o1_L_R, mask_o1_H_R = out

    o2_target = '{} offer 2'.format(decode_var)
    o2_left_mask = data['side of offer 1 (Left = 1 Right =0)'] == 0
    o2_right_mask = data['side of offer 1 (Left = 1 Right =0)'] == 1
    out = _compute_masks(data, o2_target, o2_target, dead_perc=dead_perc,
                         use_split_dec=use_split_dec, dec_mask=o2_left_mask,
                         gen_mask=o2_right_mask, c1_targ=c1_targ, c2_targ=c2_targ)
    mask_o2_L_L, mask_o2_H_L, mask_o2_L_R, mask_o2_H_R = out

    out = _compute_masks(data, o1_target, o2_target, dead_perc=dead_perc,
                         use_split_dec=use_split_dec, c1_targ=c1_targ, c2_targ=c2_targ)
    mask_L_o1, mask_H_o1, mask_L_o2, mask_H_o2 = out

    ch_left_mask = data['Choice left (==1) or Right (==0)'] == 1
    ch_right_mask = data['Choice left (==1) or Right (==0)'] == 0
    ch_o1_mask = data['choice offer 1 (==1) or 2 (==0)'] == 1
    ch_o2_mask = data['choice offer 1 (==1) or 2 (==0)'] == 0

    out_dict = {}
    mask_pairs = {
        'choice time': (((ch_o1_mask.rs_and(o1_left_mask),
                          ch_o2_mask.rs_and(o1_left_mask)),
                        (ch_o1_mask.rs_and(o1_right_mask),
                         ch_o2_mask.rs_and(o1_right_mask))),
                        1),
        'choice side': (((ch_left_mask.rs_and(o1_left_mask),
                         ch_right_mask.rs_and(o1_left_mask)),
                        (ch_left_mask.rs_and(o1_right_mask),
                         ch_right_mask.rs_and(o1_right_mask))),
                        1),
        # value decoder for left, generalizing to right
        'offer 1': (((mask_o1_L_L,
                      mask_o1_H_L),
                     (mask_o1_L_R,
                      mask_o1_H_R)),
                    1),
        'offer 2': (((mask_o2_L_L,
                      mask_o2_H_L),
                     (mask_o2_L_R,
                      mask_o2_H_R)),
                    1),
        # decoder for higher side (right higher vs left higher)
        # generalizing across presentation order
        'left-higher vs right-higher -- no order': (((mask_L_L.rs_and(mask_H_R),
                                                      mask_H_L.rs_and(mask_L_R)),
                                                     (mask_L_L.rs_and(mask_H_R),
                                                      mask_H_L.rs_and(mask_L_R))),
                                                    1),
        'left-higher vs right-higher': (((mask_o1_L_L.rs_and(mask_o2_H_R),
                                          mask_o1_H_L.rs_and(mask_o2_L_R)),
                                         (mask_o1_H_R.rs_and(mask_o2_L_L),
                                          mask_o1_L_R.rs_and(mask_o2_H_L))),
                                        2),
        'first-higher vs second-higher -- no side': (((mask_L_o1.rs_and(mask_H_o2),
                                                       mask_H_o1.rs_and(mask_L_o2)),
                                                      (mask_L_o1.rs_and(mask_H_o2),
                                                       mask_H_o1.rs_and(mask_L_o2))),
                                                     1),
        'first-higher vs second-higher': (((mask_o1_L_L.rs_and(mask_o2_H_R),
                                            mask_o1_H_L.rs_and(mask_o2_L_R)),
                                           (mask_o1_L_R.rs_and(mask_o2_H_L),
                                            mask_o1_H_R.rs_and(mask_o2_L_L))),
                                          2),
        'offer order': (((o1_left_mask, o1_right_mask),
                         (o2_right_mask, o2_left_mask)),
                        .5),
    }

    if only_pairs is not None:
        use_keys = only_pairs
    else:
        use_keys = mask_pairs.keys()
    for k in use_keys:
        ((dec_mask, targ_mask), min_factor) = mask_pairs[k]
        mask_c1, mask_c2 = dec_mask
        gen_mask_c1, gen_mask_c2 = targ_mask
        out1 = data.decode_masks(mask_c1, mask_c2, winsize, tbeg, tend, tstep,
                                 pseudo=True, time_zero_field=dec_tzf,
                                 min_trials_pseudo=min_trials/min_factor,
                                 resample_pseudo=pop_resamples, ret_pops=True,
                                 shuffle_trials=shuffle_trials, pre_pca=pre_pca,
                                 decode_tzf=dec_tzf, decode_m1=gen_mask_c1,
                                 decode_m2=gen_mask_c2, **kwargs)
        out2 = data.decode_masks(gen_mask_c1, gen_mask_c2, winsize, tbeg, tend, tstep,
                                 pseudo=True, time_zero_field=dec_tzf,
                                 min_trials_pseudo=min_trials/min_factor,
                                 resample_pseudo=pop_resamples, ret_pops=True,
                                 shuffle_trials=shuffle_trials, pre_pca=pre_pca,
                                 decode_tzf=dec_tzf, decode_m1=mask_c1,
                                 decode_m2=mask_c2, **kwargs)
        out_dict[k] = (out1, out2)

    return out_dict


def estimate_decoding_regions(*args, **kwargs):
    return _estimate_across_regions(*args, func=estimate_decoding_late, **kwargs)


def estimate_rdm_regions(*args, **kwargs):
    return _estimate_across_regions(*args, func=estimate_rdm_conditions, **kwargs)


def _estimate_across_regions(
    data,
    tbeg,
    tend,
    region_list=('OFC', 'PCC', 'pgACC', 'vmPFC', 'VS', None),
    func=estimate_decoding_late,
    **kwargs
):
    out_dict = {}
    for region in region_list:
        if region == "all":
            use_region = None
        elif region is not None:
            use_region = (region,)
        else:
            use_region = None

        out_dict[region] = func(
            data,
            tbeg,
            tend,
            regions=use_region,
            **kwargs,
        )
    return out_dict


def make_prediction_pops(dec_dict, **kwargs):
    out_dict = {}
    for region, r_dict in dec_dict.items():
        r_cond_dict = {}
        for cond, (o1, o2) in r_dict.items():
            (_, _, p1_o1, p2_o1, p3_o1, p4_o1, _) = o1
            out_o1 = mrdt.direct_ccgp_bind_est_pops(
                (p1_o1, p2_o1), (p3_o1, p4_o1), **kwargs
            )

            (_, _, p1_o2, p2_o2, p3_o2, p4_o2, _) = o2
            out_o2 = mrdt.direct_ccgp_bind_est_pops(
                (p1_o2, p2_o2), (p3_o2, p4_o2), **kwargs
            )
            r_cond_dict[cond] = (out_o1, out_o2)
        out_dict[region] = r_cond_dict
    return out_dict


def _gen_pops(c1_list, c2_list, pop, all_cons=True):
    if all_cons:
        u_cons = np.unique(np.concatenate((c1_list, c2_list)))
        for con in u_cons:
            c1_mask = c1_list == con
            c2_mask = c2_list == con
            c1_con_list = c1_list[c2_mask]
            c2_con_list = c2_list[c1_mask]
            pop1 = pop[c2_mask]
            pop2 = pop[c1_mask]
            yield (c1_con_list, c2_con_list, pop1, pop2)
    else:
        yield (c1_list, c2_list, pop, pop)

def contrast_regression(data, tbeg, tend, binsize=None, binstep=None,
                        include_field='include',
                        contrast_left='contrastLeft',
                        contrast_right='contrastRight', regions=None,
                        tz_field='stimOn', model=sklm.Ridge, norm=True,
                        pca=None, t_ind=0, cache=True, fix_con_mask=True,
                        n_cv=10):
    if binsize is None:
        binsize = tend - tbeg
    dec_pipe = na.make_model_pipeline(model, norm=norm, pca=pca)

    inc_mask = data[include_field] == 1
    d, pops, xs = data.mask_population(inc_mask, binsize, tbeg, tend,
                                       binstep=binstep, skl_axes=False,
                                       time_zero_field=tz_field,
                                       regions=regions, cache=cache)
    c_left = d[contrast_left]
    c_right = d[contrast_right]
    scores_left = []
    scores_right = []
    wi_angs = []
    ac_angs = []
    for i, pop_i in enumerate(pops):
        c_l_i = c_left[i]
        c_r_i = c_right[i]
        pop_i = pop_i[..., t_ind]
        data_generator = _gen_pops(c_l_i, c_r_i, pop_i,
                                   all_cons=fix_con_mask)
        for (c_l_i, c_r_i, pop_i_l, pop_i_r) in data_generator:
            if (pop_i.shape[1] > 0 and pop_i_l.shape[0] > n_cv*2
                and pop_i_r.shape[0] > n_cv*2):

                score_l = skms.cross_validate(dec_pipe, pop_i_l, c_l_i,
                                              return_estimator=True, cv=n_cv)
                score_r = skms.cross_validate(dec_pipe, pop_i_r, c_r_i,
                                              return_estimator=True, cv=n_cv)
                coefs_l = np.array(list(e.steps[-1][1].coef_
                                        for e in score_l['estimator']))
                coefs_r = np.array(list(e.steps[-1][1].coef_
                                        for e in score_r['estimator']))
                wi_angs_l = u.pairwise_cosine_similarity(coefs_l)
                wi_angs_r = u.pairwise_cosine_similarity(coefs_r)
                ac_angs_lr = u.pairwise_cosine_similarity(coefs_l, coefs_r)

                scores_left.append(score_l['test_score'])
                scores_right.append(score_r['test_score'])
                wi_angs.append((wi_angs_l, wi_angs_r))
                ac_angs.append(ac_angs_lr)
    return np.array(scores_left), np.array(scores_right), wi_angs, ac_angs

def regression_cv(pops, rts, model=sklm.Ridge, norm=True, pca=.95):
    pipe = na.make_model_pipeline(model, norm=norm, pca=pca)
    outs = np.zeros(len(pops))
    for i, pop in enumerate(pops):
        rt = rts[i]
        out = skms.cross_validate(pipe, pop, rt, cv=10,
                                  return_train_score=True)
        outs[i] = np.mean(out['test_score'])
    return outs

class CompositeElasticNetRidge:

    def __init__(self, **kwargs):
        self.en = sklm.ElasticNetCV()
        self.r = sklm.Ridge()
        merge_dict = {}
        rm_keys = []
        for k, v in kwargs.items():
            if k != 'mask' and len(k.split('__')) == 1:
                merge_dict['elnet__' + k] = v
                merge_dict['ridge__' + k] = v
                rm_keys.append(k)
        kwargs.update(merge_dict)
        list(kwargs.pop(k_i) for k_i in rm_keys)
        self.set_params(**kwargs)

    def get_params(self, deep=True):
        ridge_pars = self.r.get_params()
        ridge_dict = {'ridge__' + k:v for k, v in ridge_pars.items()}
        elnet_pars = self.en.get_params()
        elnet_dict = {'elnet__' + k:v for k, v in elnet_pars.items()}
        full_dict = {}
        full_dict.update(ridge_dict)
        full_dict.update(elnet_dict)
        full_dict['mask'] = self.mask
        return full_dict

    def set_params(self, mask=None, **kwargs):
        self.mask = mask
        ridge_dict = {}
        elnet_dict = {}
        for k, v in kwargs.items():
            est, key = k.split('__')
            if est == 'ridge':
                ridge_dict[key] = v
            elif est == 'elnet':
                elnet_dict[key] = v
        self.r.set_params(**ridge_dict)
        self.en.set_params(**elnet_dict)
        return self

    def _mask_x(self, X):
        if np.all(~self.mask):
            new_X = np.zeros((X.shape[0], 1))
        else:
            new_X = X[:, self.mask]
        return new_X
        
    def fit(self, X, y):
        self.en.fit(X, y)
        self.mask = np.abs(self.en.coef_) > 0
        self.r.fit(self._mask_x(X), y)
        coef = np.zeros(X.shape[1])
        if np.any(self.mask):
            coef[self.mask] = self.r.coef_
        self.coef_ = coef
        return self

    def predict(self, X):
        return self.r.predict(self._mask_x(X))
        
    def score(self, X, y):
        return self.r.score(self._mask_x(X), y)
    
default_cond_list = ((-1, 1), (1, 1), (-1, -1), (1, -1))
def predict_ccgp(pop_list, cond_list=default_cond_list,
                 model=sklm.MultiTaskElasticNetCV, multi_task=True,
                 l1_ratio=.5, pre_pca=.95, test_prop=.1, eps=1e-1):
    lm, nlm, nv, r2 = fit_linear_models(pop_list, cond_list, folds_n=1, 
                                        model=model, multi_task=multi_task,
                                        l1_ratio=l1_ratio,
                                        pre_pca=pre_pca, test_prop=test_prop,
                                        eps=eps)
    return compute_ccgp_bin_prediction(lm, nlm, nv, r2, multi_task=multi_task)

def compute_ccgp_bin_prediction(lm, nlm, nv, r2=None, multi_task=False):
    ccgp_all = []
    bin_err_all = []
    # print(r2.shape)
    # print(lm.shape)
    # lm_noise = lm/nv
    # nlm_noise = nlm/nv
    lm = np.mean(lm, axis=1)
    nlm = np.mean(nlm, axis=1)
    nv_avg = np.mean(np.sqrt(nv), axis=1)
    if r2 is not None and not multi_task:
        r2 = np.mean(r2, axis=1)
        r2_mask = r2 > 0
        lm_rep = np.zeros_like(lm)
        r2_mask_lm = np.stack((r2_mask,)*lm.shape[2], axis=2)
        lm_rep[r2_mask_lm] = lm[r2_mask_lm]
        nlm_rep = np.zeros_like(nlm) 
        r2_mask_nlm = np.stack((r2_mask,)*nlm.shape[2], axis=2)
        nlm_rep[r2_mask_nlm] = nlm[r2_mask_nlm]
        lm = lm_rep
        nlm = nlm_rep
    for pop_ind in range(lm.shape[0]):
        # bin_err, ccgp_err = predict_ccgp_binding(lm[pop_ind], nlm[pop_ind],
        #                                          nv_avg[pop_ind])
        lm_red = np.squeeze(lm[pop_ind])[:, 0:1]
        nlm_red = np.squeeze(nlm[pop_ind])
        bin_err, ccgp_err = predict_asymp_dists(lm_red, nlm_red,
                                                nv_avg[pop_ind])

        ccgp_all.append(1 - ccgp_err)
        bin_err_all.append(1 - bin_err)
    ccgp = np.stack(ccgp_all, axis=0)
    bin_err = np.stack(bin_err_all, axis=0)
    return ccgp, bin_err, (lm, nlm, nv, r2)

def _get_percentile_mask(field_data, dead_perc, additional_data=None,
                         ref_data=None):
    if ref_data is None:
        ref_data = field_data
    high_masks, low_masks = [], []
    if additional_data is not None:
        high_masks_ad, low_masks_ad = [], []
    for i, rd in enumerate(ref_data):
        low_thr = np.percentile(rd, 50 - dead_perc/2)
        high_thr = np.percentile(rd, 50 + dead_perc/2)
        if high_thr == low_thr:
            high_mask = np.zeros_like(field_data[i], dtype=bool)
            low_mask = np.zeros_like(field_data[i], dtype=bool)
        else:
            high_mask = field_data[i] >= high_thr
            low_mask = field_data[i] <= low_thr
        high_masks.append(high_mask)
        low_masks.append(low_mask)
        if additional_data is not None:
            high_masks_ad.append(additional_data[i] >= high_thr)
            low_masks_ad.append(additional_data[i] <= low_thr)
    
    out = (gio.ResultSequence(high_masks), gio.ResultSequence(low_masks))
    if additional_data is not None:
        out = out + (gio.ResultSequence(high_masks_ad),
                     gio.ResultSequence(low_masks_ad))
    return out

def full_lm_organization(data, tbeg, tend, dead_perc=30, winsize=500, tstep=20,
                         turn_feature=('ev_left', 'ev_right'),
                         tzf_1='Offer 1 on', tzf_2='Offer 2 on',
                         pre_pca=.95, **kwargs):
    """
    (1, -1) presented stim is high vs low EV
    (1, -1) presented stim is first vs second
    (1, -1) presented stim is left vs right
    (1, -1) presented stim is eventually chosen vs unchosen
    """
    ev_1 = 'Expected value offer 1'
    ev_2 = 'Expected value offer 2'
    side_1 = 'side of offer 1 (Left = 1 Right =0)'
    choice = 'choice offer 1 (==1) or 2 (==0)'
    out = _get_percentile_mask(data[ev_1], dead_perc, data[ev_2])
    mask1_high, mask1_low, mask2_high, mask2_low = out
    cond_pop_pair = []
    pops1, xs = data.get_psth(binsize=winsize, begin=tbeg, end=tend,
                              binstep=tstep,
                              time_zero_field=tzf_1, **kwargs)
    pops2, xs = data.get_psth(binsize=winsize, begin=tbeg, end=tend,
                              binstep=tstep,
                              time_zero_field=tzf_2, **kwargs)
    for i, session in enumerate(data.data['data']):
        m1h_i = mask1_high[i]
        m1l_i = mask1_low[i]
        m2h_i = mask2_high[i]
        m2l_i = mask2_low[i]
        c1_i = np.zeros((len(session), 4))
        c1_i[m1h_i, 0] = 1
        c1_i[m1l_i, 0] = -1
        c1_i[:, 1] = 1
        left_mask = session[side_1] == 1
        c1_i[left_mask, 2] = 1
        c1_i[np.logical_not(left_mask), 2] = -1
        chosen_mask = session[choice] == 1
        c1_i[chosen_mask, 3] = 1
        c1_i[np.logical_not(chosen_mask), 3] = -1

        c2_i = np.zeros((len(session), 4))
        c2_i[m2h_i, 0] = 1
        c2_i[m2l_i, 0] = -1
        c2_i[:, 1] = -1

        c2_i[left_mask, 2] = -1
        c2_i[np.logical_not(left_mask), 2] = 1

        c2_i[chosen_mask, 3] = -1
        c2_i[np.logical_not(chosen_mask), 3] = 1

        include_m1 = c1_i[:, 0] != 0
        include_m2 = c2_i[:, 0] != 0
        c1_i = c1_i[include_m1]
        c2_i = c2_i[include_m2]

        fc_i = np.concatenate((c1_i, c2_i), axis=0)
        pop_i = np.concatenate((pops1[i][include_m1], pops2[i][include_m2]),
                               axis=0)
        ohe = skp.OneHotEncoder(sparse=False)
        cond_list = list(tuple(c_ij) for c_ij in fc_i)
        u_conds = np.unique(cond_list, axis=0)
        c_dict = {tuple(cond):i for i, cond in enumerate(u_conds)}
        ohe.fit(np.arange(len(u_conds)).reshape(-1, 1))
        int_list = np.array(list(c_dict[tuple(cond)]
                                 for cond in cond_list)).reshape(-1, 1)
        nonlin_mat = ohe.transform(int_list)
        cond_pop_pair.append(((fc_i, nonlin_mat), pop_i))
    trl_pop_pair = []
    for conds, pop_i in cond_pop_pair:
        if pop_i.shape[1] > 0:
            trl_pop_pair.append((conds, pop_i))
    return trl_pop_pair, xs

def _reformat_mat(mat):
    mat = np.swapaxes(mat, 0, 2)
    mat = np.expand_dims(mat, -1)
    mat = np.swapaxes(mat, 1, -1)
    return mat

def construct_lin_nonlin_mat(cpp, lm_dicts, filter_r2=True, r2_key='test_score',
                             mat_key='estimator'):
    lin_mat = []
    nonlin_mat = []
    nv_mat = []
    for i, ((lc, _), _) in enumerate(cpp):
        lin_cols = lc.shape[1]
        r2 = lm_dicts[i][r2_key]
        nv_mat.append(1 - r2)
        mat = lm_dicts[i][mat_key]
        if filter_r2:
            mask = r2 > 0
            new_mat = np.zeros_like(mat)
            new_mat[:] = np.nan
            new_mat[mask] = mat[mask]
            mat = new_mat
        lin_mat.append(mat[..., :lin_cols])
        nonlin_mat.append(mat[..., lin_cols:])
    lm_full = np.concatenate(lin_mat, axis=0)
    lm_full = _reformat_mat(lm_full)
    nlm_full = np.concatenate(nonlin_mat, axis=0)
    nlm_full = _reformat_mat(nlm_full)
    nv_full = np.concatenate(nv_mat, axis=0)
    print(nv_full.shape)
    nv_full = _reformat_mat(nv_full)
    print(nv_full.shape)
    return lm_full, nlm_full, nv_full

def predict_tradeoff_betas(betas):
    lm = np.expand_dims(betas[:, :5], (0, 1))
    nlm = np.expand_dims(betas[:, 6:], (0, 1))
    resid_var = np.expand_dims(1, (0, 1))
    out = predict_ccgp_binding(lm, nlm, resid_var, n=2)
    return out

def fit_simple_full_lms(cond_pop_pairs, model=sklm.LinearRegression,
                        norm=True, pre_pca=None, time_ind=None, **model_kwargs):
    m = model(fit_intercept=False,  **model_kwargs)
    pop_outs = []
    scaler = na.make_model_pipeline(norm=True, pca=pre_pca)
    for i, ((l_conds, n_conds), pop) in enumerate(cond_pop_pairs):
        if time_ind is None:
            pop = mraux.accumulate_time(pop)
        else:
            pop = pop[..., time_ind:time_ind+1]
        print(pop.shape)
        pop_trs = na.apply_transform_tc(pop, scaler)
        conds = np.concatenate((l_conds, n_conds), axis=1)
        m.fit(conds, pop_trs[..., 0])
        u_conds = np.unique(conds, axis=0)
        pop_outs.append((u_conds, pop_trs, m.coef_))
    return pop_outs

def _get_coeff_masks(neur):
    p_coeff = neur['pval'][0, 0][0, 0]['beta']
    if p_coeff.shape[0] == 4:
        lin_mask = np.array([True, True, False, False])
        nl_mask = np.array([False, False, True, False])
        p_val_mask = np.array([True, False, False, False])
        p_space_mask = np.array([False, True, False, False])
    elif p_coeff.shape[0] == 8:
        p_val_mask = np.array([True, False, False, True, False, True, False,
                               False])
        p_space_mask = np.array([False, True, False, False, False, False, False,
                                 False])
        lin_mask = np.array([True, True, False, True, False, True, False, False])
        nl_mask = np.array([False, False, True, False, True, False, True, False])
    return lin_mask, nl_mask, (p_val_mask, p_space_mask)

def _get_mask_sig(neur, mask, p_thr=.05):
    ps = neur['pval'][0, 0][0, 0]
    p_coeff = ps['beta']
    return np.sum(p_coeff[mask] < p_thr)

def get_mult_select(neur, p_thr=.05):
    lin_mask, nl_mask, (p_val_m, p_space_m) = _get_coeff_masks(neur)
    n_val_p = _get_mask_sig(neur, p_val_m, p_thr=p_thr)
    n_space_p = _get_mask_sig(neur, p_space_m, p_thr=p_thr)
    n_nl = _get_mask_sig(neur, nl_mask, p_thr=p_thr)
    n_lin = _get_mask_sig(neur, lin_mask, p_thr=p_thr)

    out_pre = (n_val_p > 0) + (n_space_p > 0) 
    out = (out_pre > 0) and (n_nl > 0)
    out = (n_lin > 0 ) and (n_nl > 0)
    if (n_lin + n_nl) == 0:
        out = np.nan
    return out

def get_lin_select(neur, p_thr=.05):
    lin_mask, nl_mask, _ = _get_coeff_masks(neur)
    n_lin = _get_mask_sig(neur, lin_mask, p_thr=p_thr)
    n_nl = _get_mask_sig(neur, nl_mask, p_thr=p_thr)
    out = (n_lin > 0) and (n_nl == 0)
    if (n_lin + n_nl) == 0:
        out = np.nan
    return out


def get_pure_select(neur, p_thr=.05):
    lin_mask, nl_mask, (p_val_m, p_space_m) = _get_coeff_masks(neur)
    n_val_p = _get_mask_sig(neur, p_val_m, p_thr=p_thr)
    n_space_p = _get_mask_sig(neur, p_space_m, p_thr=p_thr)
    n_nl = _get_mask_sig(neur, nl_mask, p_thr=p_thr)
    n_lin = _get_mask_sig(neur, lin_mask, p_thr=p_thr)

    out = np.logical_xor(n_val_p > 0, n_space_p > 0) and n_nl == 0
    if (n_lin + n_nl) == 0:
        out = np.nan
    return out

def get_lin_mix_select(neur, p_thr=.05):
    lin_mask, nl_mask, (p_val_m, p_space_m) = _get_coeff_masks(neur)
    n_val_p = _get_mask_sig(neur, p_val_m, p_thr=p_thr)
    n_space_p = _get_mask_sig(neur, p_space_m, p_thr=p_thr)
    n_nl = _get_mask_sig(neur, nl_mask, p_thr=p_thr)
    n_lin = _get_mask_sig(neur, lin_mask, p_thr=p_thr)

    out = np.logical_and(n_val_p > 0, n_space_p > 0) and n_nl == 0
    if (n_lin + n_nl) == 0:
        out = np.nan
    return out

def get_conj_select(neur, p_thr=.05):
    lin_mask, nl_mask, _ = _get_coeff_masks(neur)
    n_lin = _get_mask_sig(neur, lin_mask, p_thr=p_thr)
    n_nl = _get_mask_sig(neur, nl_mask, p_thr=p_thr)

    out = (n_lin == 0) and (n_nl > 0)
    if (n_lin + n_nl) == 0:
        out = np.nan
    return out    

def apply_coeff_func(path, func, regions=None, epoch='E1_E', model_version='md1',
                     model_type='OLS', **kwargs):
    x = sio.loadmat(path)
    x = x[model_type]
    if regions is None:
        regions = x.dtype.names
    out_dict = {}
    for region in regions:
        y = x[region][0, 0][0]
        out_dict[region] = []
        for i, y_i in enumerate(y):
            neurs = y_i[0, 0]['full'][0]
            out_dict[region].append(np.zeros(neurs.shape))
            for j, neur in enumerate(neurs):
                n_j = neur[0, 0]['model'][epoch][0, 0]['models'][0, 0]
                n_j = n_j[model_version][0, 0]
                out_dict[region][i][j] = func(n_j, **kwargs)
    return out_dict
            

rand_splitter = skms.ShuffleSplit
def fit_full_lms(cond_pop_pairs, model=sklm.MultiTaskElasticNetCV, norm=True,
                 pca=None, rand_splitter=rand_splitter, test_prop=None,
                 folds_n=20, shuffle=False, pre_pca=None, multi_task=True,
                 **model_kwargs):
    """ conds is the same length as pops, and gives the feature values """
    if test_prop is None:
        test_prop = 1/folds_n
    scaler = na.make_model_pipeline(norm=True, pca=pre_pca)
    if rand_splitter is None:
        splitter = skms.KFold(folds_n, shuffle=True)
        internal_splitter = skms.KFold(folds_n, shuffle=True)
    else:
        splitter = rand_splitter(folds_n, test_size=test_prop)
        internal_splitter = rand_splitter(folds_n, test_size=test_prop)
    m = model(fit_intercept=False,  **model_kwargs)
    pop_outs = []
    for i, ((l_conds, n_conds), pop) in enumerate(cond_pop_pairs):
        pop_trs = na.apply_transform_tc(pop, scaler)
        conds = np.concatenate((l_conds, n_conds), axis=1)
        keep_keys = {'test_score':(pop_trs.shape[-2], pop_trs.shape[-1],
                                   folds_n),
                     'estimator':(pop_trs.shape[-2], pop_trs.shape[-1], folds_n,
                                  conds.shape[1])}
        out = {k:np.zeros(v) for k, v in keep_keys.items()}
        out = na.skl_model_target_dim_tc(m, conds, pop_trs, out=out,
                                         cv=splitter, 
                                         return_estimator=True,
                                         keep_keys=keep_keys)
        pop_outs.append(out)
    return pop_outs

def format_pops(*pops, accumulate_time=True, norm=True, pca=.95,
                polynomial_value_degree=1, interaction=False,
                make_one_hot=False):
    feat_list = []
    activity_list = []
    for i, pop in enumerate(pops):
        for j, pj in enumerate(pop):
            if accumulate_time:
                pj = np.concatenate(list(pj[..., k] for k in range(pj.shape[-1])),
                                    axis=1)
            pj = np.swapaxes(pj, 1, 3)
            pj = pj[:, :, 0]
            feat_shape = pj.shape[:2]
            activity_list.append(pj)
            fa = np.ones(feat_shape + (2,))
            fa[..., 0] = fa[..., 0]*i
            fa[..., 1] = fa[..., 1]*j
            feat_list.append(fa)
    feats = np.concatenate(feat_list, axis=1)
    acts = np.concatenate(activity_list, axis=1)
    act_pipe = na.make_model_pipeline(norm=norm, pca=pca,
                                      post_norm=True)
    feat_pipe = na.make_model_pipeline(norm=norm)
    new_feats = []
    for i, ppop in enumerate(feats):
        pos_mask = np.array([True, False])
        feats[i, :, ~pos_mask] = feat_pipe.fit_transform(ppop[:, ~pos_mask]).T
        new_act = act_pipe.fit_transform(acts[i])
        acts[i, :, :new_act.shape[1]] = new_act
        acts[i, :, new_act.shape[1]:] = np.nan
        new_feats.append(feats[i])
        if make_one_hot:
            ohe = skp.OneHotEncoder(sparse=False)
            new_pos = ohe.fit_transform(new_feats[i][:, pos_mask])
            new_feats[i] = np.concatenate((new_pos,
                                           new_feats[i][:, ~pos_mask]),
                                          axis=1)
            pos_mask = np.array((True,)*new_pos.shape[1] + (False,))
        if polynomial_value_degree > 1:
            pf = skp.PolynomialFeatures(degree=polynomial_value_degree,
                                        include_bias=False)
            new_val = pf.fit_transform(new_feats[i][:, ~pos_mask])
            pos = new_feats[i][:, pos_mask]
            new_feats[i] = np.concatenate((pos, new_val),
                                          axis=1)
            pos_mask = np.concatenate((pos_mask[pos_mask],
                                       np.zeros(new_val.shape[1], dtype=bool)))
        if interaction:
            val_inds = np.where(~pos_mask)[0]
            for ind in np.where(pos_mask)[0]:
                inter_feats = (new_feats[i][:, val_inds]
                               *new_feats[i][:, ind:ind+1])
                new_feats[i] = np.concatenate((new_feats[i], inter_feats),
                                              axis=1)

    feats = np.stack(new_feats, axis=0)
    return feats, acts

rand_splitter = skms.ShuffleSplit
def fit_pseudo_lms(l_pops, r_pops, model=sklm.ElasticNet, norm=True,
                   pca=None, shuffle=False, pre_pca=None, multi_task=True,
                   interaction=True, make_one_hot=True,
                   polynomial_value_degree=1,
                   splitter=rand_splitter,
                   use_kfold=False, 
                   n_folds=20, leave_out=.05, **model_kwargs):
    """ conds is the same length as pops, and gives the feature values """
    feats, acts = format_pops(l_pops, r_pops, interaction=interaction,
                              make_one_hot=make_one_hot,
                              polynomial_value_degree=polynomial_value_degree)
    models = np.zeros((feats.shape[0], n_folds), dtype=object)
    coeffs = np.zeros((feats.shape[0], n_folds, acts.shape[2], feats.shape[2]))
    vis_targ = np.zeros((feats.shape[0], n_folds), dtype=object)
    vals = np.unique(feats[..., 2])
    val_resps_pos1 = np.zeros((feats.shape[0], n_folds, acts.shape[2],
                               len(vals)))
    val_resps_pos2 = np.zeros_like(val_resps_pos1)
    
    val_resps_pos1[:] = np.nan
    val_resps_pos2[:] = np.nan
    for i, feat in enumerate(feats):
        pos1_mask = feat[:, 0] > 0
        pos2_mask = feat[:, 1] > 0
        act = acts[i]
        not_nan_mask = ~np.isnan(act[0])
        act = act[:, not_nan_mask]        
        m = model(**model_kwargs)
        if use_kfold:
            splitter = skms.KFold(2, shuffle=True)
        else:
            splitter = rand_splitter(n_splits=n_folds, test_size=leave_out)
        results = na.cross_validate_wrapper(m, feat, act, cv=splitter)
        print(results['test_score'])
        for k, m in enumerate(results['estimator']):
            models[i, k] = m
            coeffs[i, k, not_nan_mask] = m.coef_
            vis_targ[i, k] = results['test_targ'][k]
            for j, val in enumerate(vals):
                p1_match = np.array([1, 0, val])
                p1_mask = np.all(feat[:, :3] == p1_match, axis=1)
                p2_match = np.array([0, 1, val])
                p2_mask = np.all(feat[:, :3] == p2_match, axis=1)
                val_resps_pos1[i, k, not_nan_mask, j] = m.predict(
                    feat[p1_mask][0:1]
                )
                val_resps_pos2[i, k, not_nan_mask, j] = m.predict(
                    feat[p2_mask][0:1]
                )
        
    return (models, coeffs, val_resps_pos1, val_resps_pos2, vis_targ,
            feats, acts)

def _nan_mask(arr, ret_mask=False):
    mask = ~np.all(np.isnan(arr), axis=0)
    out = arr[:, mask]
    if ret_mask:
        out = (out, mask)
    return out

def shared_subspace_from_pops(p1_group, p2_group, t_ind=0, n_folds=10,
                              concatenate_time=True, **kwargs):
    out_same = np.zeros((p1_group[0].shape[0], n_folds, 2))
    out_diff = np.zeros((p1_group[0].shape[0], n_folds, 2))
    p1_comb = np.concatenate(p1_group, axis=3)
    p2_comb = np.concatenate(p2_group, axis=3)
    
    for i, p1_i in enumerate(p1_comb):
        p2_i = np.squeeze(p2_comb[i])
        p1_i = np.squeeze(p1_i)
        if concatenate_time:
            p1_i = np.concatenate(list(p1_i[..., j]
                                       for j in range(p1_i.shape[-1])),
                                  axis=0)
            p2_i = np.concatenate(list(p2_i[..., j]
                                       for j in range(p2_i.shape[-1])),
                                  axis=0)
        else:
            p1_i = p1_i[..., t_ind]
            p2_i = p2_i[..., t_ind]

        tv, trs_v = shared_subspace(p1_i.T, p2_i.T, n_folds=n_folds, **kwargs)

        out_same[i] = tv
        out_diff[i] = trs_v
    return out_same, out_diff


def fit_bootstrap_models(session_dict, n_boots=500, model=sklm.Ridge,
                         fit_separately=False, **kwargs):
    out_dict = {}
    for key,  pred in session_dict.items():
        predictors, targets = pred[:2]
        shape_mat = (n_boots, targets.shape[1], predictors.shape[1], targets.shape[2])
        mat1 = np.zeros(shape_mat)
        mat2 = np.zeros_like(mat1)

        shape_inter = (n_boots, targets.shape[1], 1, targets.shape[2])
        inter1 = np.zeros(shape_inter)
        inter2 = np.zeros_like(inter1)
        for i in range(n_boots):
            pred1, targ1 = sku.resample(predictors, targets)
            pred2, targ2 = sku.resample(predictors, targets)
            for j in range(targ1.shape[-1]):
                t1_j = targ1[..., j]
                t2_j = targ2[..., j]
                if not fit_separately:
                    m1 = model()
                    m2 = model()
                    m1.fit(pred1, t1_j)
                    m2.fit(pred2, t2_j)
                    mat1[i, ..., j] = m1.coef_
                    mat2[i, ..., j] = m2.coef_
                    inter1[i, ..., j] = np.expand_dims(m1.intercept_, 1)
                    inter2[i, ..., j] = np.expand_dims(m2.intercept_, 1)
                else:
                    for k in range(targ1.shape[1]):
                        m1 = model()
                        m2 = model()
                        m1.fit(pred1, t1_j[:, k])
                        m2.fit(pred2, t2_j[:, k])
                        mat1[i, k, ..., j] = m1.coef_
                        mat2[i, k, ..., j] = m2.coef_
                        inter1[i, k, ..., j] = m1.intercept_
                        inter2[i, k, ..., j] = m2.intercept_
        out_dict[key] = ((mat1, inter1), (mat2, inter2))
    return out_dict      


def _make_factor(v, u_, s):
    s_mat = np.expand_dims(np.sqrt(s), 1) * np.identity(s.shape[0])
    v_factor = u_ @ s_mat @ v
    v_align = sla.orth(v_factor)
    return v_align


def _compute_null_alignment_index(s1, s2, thresh=1e-10, full_data=None, dim=-1,):
    dim = -1
    # s1 is a resamps x pts x neurons matrix
    s_comb = np.concatenate((s1, s2), axis=1)
    # cov should be resamps x neuron x neuron matrix
    ais = np.zeros((len(s_comb), s_comb.shape[dim]))
    for i, s_i in enumerate(s_comb):
        for j in range(s_comb.shape[dim]):
            s_ij = np.take(s_i, j, axis=dim)
            n_dims = np.sum(
                skd.PCA().fit(s_ij).explained_variance_ratio_ > thresh
            )
            mu = np.mean(s_ij, axis=0, keepdims=True)
            cov = (s_ij - mu).T @ (s_ij - mu)
            u_, s, vh = np.linalg.svd(cov)
            v1_raw = sts.norm(0, 1).rvs((n_dims, s_ij.shape[1])).T
            v2_raw = sts.norm(0, 1).rvs((n_dims, s_ij.shape[1])).T
            v1 = _make_factor(v1_raw, u_, s)
            v2 = _make_factor(v2_raw, u_, s)
        
            ais[i, j] = np.trace(v1.T @ v2 @ v2.T @ v1) / n_dims
    return ais


def _compute_alignment_index(s1, s2, thresh=1e-10, null=False, dim=-1, full_data=None):
    dim = -1
    ais = np.zeros((len(s1), s1.shape[dim]))
    for i, s1_i in enumerate(s1):
        s2_i = s2[i]
        for j in range(s1.shape[dim]):
            s1_ij = np.take(s1_i, j, axis=dim)
            s2_ij = np.take(s2_i, j, axis=dim)
            p1 = skd.PCA()
            p1.fit(s1_ij)
            p2 = skd.PCA()
            p2.fit(s2_ij)
            mask1 = p1.explained_variance_ratio_ > thresh
            u1 = p1.components_[mask1].T
            mask2 = p2.explained_variance_ratio_ > thresh
            u2 = p2.components_[mask2].T

            norm = min(u1.shape[1], u2.shape[1])
            ais[i, j] = np.trace(u1.T @ u2 @ u2.T @ u1)/norm
    return ais
    

def compute_alignment_index(s1, s2, null=False, **kwargs):
    if len(s1.shape) == 2:
        s1 = np.expand_dims(s1, 0)
        s2 = np.expand_dims(s2, 0)
    if null:
        out = _compute_null_alignment_index(s1, s2, **kwargs)
    else:
        out = _compute_alignment_index(s1, s2, **kwargs)
    return out


def compute_corr(s1, s2, pearson_brown=None, null=False, full_data=None, dim=-1):
    if len(s1.shape) == 2:
        s1 = np.expand_dims(s1, 0)
        s2 = np.expand_dims(s2, 0)
    if null and full_data is None:
        half = s1.shape[1]
        s_comb = np.concatenate((s1, s2), axis=1)
        rng = np.random.default_rng()
        s_comb = rng.permutation(s_comb, axis=1)
        s1 = s_comb[:, :half]
        s2 = s_comb[:, half:]
        s1_uv = u.make_unit_vector(
            s1[:, 0] - s1[:, -1], dim=dim, squeeze=False
        )
        s2_uv = u.make_unit_vector(
            s2[:, 0] - s2[:, -1], dim=dim, squeeze=False
        )
    elif null:
        s1_uv = np.zeros((len(full_data),) + full_data.shape[2:])
        s2_uv = np.zeros_like(s1_uv)
        for i, s_i in enumerate(full_data):
            mu = np.mean(s_i, axis=0, keepdims=True)
            cov = (s_i - mu).T @ (s_i - mu)
            u_, s, vh = np.linalg.svd(cov)
            v1_raw = sts.norm(0, 1).rvs((1, s_i.shape[1])).T
            v2_raw = sts.norm(0, 1).rvs((1, s_i.shape[1])).T
            s1_uv = _make_factor(v1_raw, u_, s)
            s2_uv = _make_factor(v2_raw, u_, s)
    else:
        s1_uv = u.make_unit_vector(s1[:, 0] - s1[:, -1], dim=dim, squeeze=False)
        s2_uv = u.make_unit_vector(s2[:, 0] - s2[:, -1], dim=dim, squeeze=False)
    r = np.sum(s1_uv*s2_uv, axis=dim)
    if pearson_brown is not None:
        r = pearson_brown*r/(1 + (pearson_brown - 1)*r)
    return r


def fit_split_half_models(session_dict, n_splits=100,
                          cv=skms.ShuffleSplit,
                          **kwargs):
    out_dict = {}
    for key, (predictors, targets) in session_dict.items():
        splitter = cv(n_splits=n_splits, test_size=.5)
        mat1 = np.zeros((n_splits, targets.shape[1], predictors.shape[1]))
        mat2 = np.zeros((n_splits, targets.shape[1], predictors.shape[1]))
        for i, (inds1, inds2) in enumerate(splitter.split(predictors)):
            pred1 = predictors[inds1]
            targ1 = targets[inds1]
            pred2 = predictors[inds2]
            targ2 = targets[inds2]
            m1 = sklm.Ridge()
            m1.fit(pred1, targ1)
            m2 = sklm.Ridge()
            m2.fit(pred2, targ2)
            mat1[i] = m1.coef_
            mat2[i] = m2.coef_
        out_dict[key] = (mat1, mat2)
    return out_dict      


def _make_dict_and_fit(predictors, targets, sigma_prior=1, mu_prior=1,
                       noise_model=False,
                       model_path='general/stan_models/lm.pkl',
                       **kwargs):
    N, K = predictors.shape
    stan_dict = {
        'N':N,
        'K':K,
        'x':predictors,
        'sigma_std_prior':sigma_prior,
        'mu_std_prior':mu_prior,
    }
    fits = []
    diags = []
    if noise_model:
        model_path = 'general/stan_models/noise.pkl'
    for i in range(targets.shape[1]):
        stan_dict['y'] = targets[:, i]
        _, fit_az, diag = su.fit_model(stan_dict, model_path, **kwargs)
        fits.append(fit_az)
        diags.append(diag)
    stan_dict['y'] = targets
    return stan_dict, fits, diags


def fit_stan_models(session_dict, **kwargs):
    out_dict = {}
    for key, (predictors, targets) in session_dict.items():
        stan_dict, fits, diags = _make_dict_and_fit(predictors, targets,
                                                    **kwargs)
        out_dict[key] = (stan_dict, fits, diags)
    return out_dict      


def make_model_alternatives(
        data,
        save_size=1,
        folder='multiple_representations/model_dicts/',
        add="",
        **kwargs
):
    params_dicts = {
        'null': {'include_interaction': False},
        'interaction': {'include_interaction': True},
        'null_spline': {'transform_value': True, 'include_interaction': False},
        'interaction_spline': {'include_interaction': True,
                               'transform_value': True},
    }
    model_dicts = {}
    if len(add) > 0:
        add = "_" + add
    for key, params in params_dicts.items():
        new_kwargs = {}
        new_kwargs.update(kwargs)
        new_kwargs.update(params)
        session_dict, _ = make_predictor_matrices(data, **new_kwargs)
        template = 'sd{}_{}'.format(add, key) + '_{}.pkl'
        num = split_and_save_subdicts(session_dict, folder, template=template,
                                      save_size=save_size)
        model_dicts[key] = session_dict
    return model_dicts, num


def make_psth_val_avg(
        data,
        targ_date,
        neur_ind,
        val1_key='subj_ev offer 1',
        val2_key='subj_ev offer 2',
        side_key='side of offer 1 (Left = 1 Right =0)',
        o1_on_key='Offer 1 on',
        o2_on_key='Offer 2 on',
        t_beg=-100,
        t_end=1000,
        twin=100,
        tstep=20,
):
    data_red = data.session_mask(data['date'] == targ_date)
    v1 = data_red[val1_key][0]
    v2 = data_red[val2_key][0]
    side1 = data_red[side_key][0]
    side2 = np.logical_not(side1)
    act1, xs = data_red.get_neural_activity(twin, t_beg, t_end, stepsize=tstep,
                                            time_zero_field=o1_on_key)
    act2, xs = data_red.get_neural_activity(twin, t_beg, t_end, stepsize=tstep,
                                            time_zero_field=o2_on_key)
    vs = np.concatenate((v1, v2), axis=0)
    act = np.concatenate((act1[0], act2[0]), axis=0)
    sides = np.concatenate((side1, side2), axis=0)
    sides[sides == 0] = -1
    return xs, act[:, neur_ind], vs, sides


def make_predictor_matrices(
        data,
        val1_key='subj_ev offer 1',
        val2_key='subj_ev offer 2',
        side_key='side of offer 1 (Left = 1 Right =0)',
        o1_on_key='Offer 1 on',
        o2_on_key='Offer 2 on',
        t_beg=100,
        t_end=1000,
        twin=None,
        tstep=None,
        norm_targets=True,
        shuffle_targets=False,
        norm_value=True,
        include_interaction=True,
        transform_value=False,
        discretize_value=False,
        use_session_value=True,
        n_value_bins=10,
        spline_knots=4,
        spline_degree=2,
        single_time=False,
        t1_only=False,
        t2_only=False,
        decrement=1,
        return_core_predictors=False,
        eps=1e-10,
):
    if discretize_value:
        v1 = np.concatenate(data[val1_key])
        v2 = np.concatenate(data[val2_key])
        v = np.concatenate((v1, v2))
        bins = np.linspace(
            np.min(v1), np.max(v2) + eps, n_value_bins + 1
        )
        all_vals = np.arange(1, n_value_bins + 1).reshape((-1, 1))
        use_val_transform = skp.StandardScaler().fit(all_vals)
    session_dicts = {}
    if tstep is not None and twin is not None:
        pops1, xs = data.get_neural_activity(
            twin, t_beg, t_end, stepsize=tstep, time_zero_field=o1_on_key
        )
        pops2, xs = data.get_neural_activity(
            twin, t_beg, t_end, stepsize=tstep, time_zero_field=o2_on_key
        )
    else:
        xs = np.array([(t_beg + t_end) / 2])
    for i, session in data.data.iterrows():
        timing = session['psth_timing']
        animal = session['animal']
        date = session['date']
        region = session.data['neur_regions'].iloc[0]
        val1 = np.expand_dims(session.data[val1_key], 1)
        val2 = np.expand_dims(session.data[val2_key], 1)
        if discretize_value:
            if use_session_value:
                v = np.concatenate((val1, val2))
                bins = np.linspace(np.min(v) - eps, np.max(v) + eps, n_value_bins + 1)
            val1 = np.digitize(val1, bins)
            val1 = use_val_transform.transform(val1)
            val2 = np.digitize(val2, bins)
            val2 = use_val_transform.transform(val2)
        side = session.data[side_key]
        sides1 = np.zeros((len(side), 1))
        sides1[side == 1, 0] = 1
        sides1[side == 0, 0] = -1

        sides2 = np.zeros((len(side), 1))
        sides2[side == 0, 0] = 1
        sides2[side == 1, 0] = -1

        p1 = np.concatenate((val1, sides1), axis=1)
        p2 = np.concatenate((val2, sides2), axis=1)
        core_predictors = np.concatenate((p1, p2), axis=0)

        if tstep is not None and twin is not None:
            resp1 = pops1[i]
            resp2 = pops2[i]
        else:
            o1_timing = session.data[o1_on_key]
            resp1 = _get_window(session.data.psth, o1_timing, t_beg, t_end,
                                timing, twin=twin, tstep=tstep)
            resp1 = np.expand_dims(resp1, -1)
            o2_timing = session.data[o2_on_key]
            resp2 = _get_window(session.data.psth, o2_timing, t_beg, t_end,
                                timing, twin=twin, tstep=tstep)
            resp2 = np.expand_dims(resp2, -1)
        targets = np.concatenate((resp1, resp2), axis=0)
        if norm_targets:
            targets = np.stack(
                list(skp.StandardScaler().fit_transform(targets[..., j])
                     for j in range(targets.shape[-1])),
                axis=-1,
            )

        predictors = form_predictors(
            core_predictors,
            transform_value=transform_value,
            norm_value=norm_value and not discretize_value,
            include_interaction=include_interaction,
            spline_knots=spline_knots,
            spline_degree=spline_degree,
        )
        half = int(predictors.shape[0] / 2)
        if single_time:
            targets = targets[half:]
            preds1 = predictors[:half]
            preds2 = predictors[half:]
            predictors = np.concatenate((preds1, preds2), axis=1)
            core_predictors = core_predictors[half:]
        elif t1_only:
            targets = targets[:half]
            predictors = predictors[:half]
            core_predictors = core_predictors[:half]
        elif t2_only:
            targets = targets[half:]
            predictors = predictors[half:]
            core_predictors = core_predictors[half:]
        elif decrement > 1:
            targets = targets[::decrement]
            predictors = predictors[::decrement]
            core_predictors = core_predictors[::decrement]
        if shuffle_targets:
            rng = np.random.default_rng()
            rng.shuffle(targets)
        add = (predictors, targets)
        if return_core_predictors:
            add = add + (core_predictors,)
        session_dicts[(region, animal, date)] = add        
    return session_dicts, xs


def form_predictors(
        predictors,
        transform_value=False,
        norm_value=True,
        include_interaction=True,
        spline_knots=4,
        spline_degree=2,
        use_val_transform=None,        
):
    val_mask = np.array([True, False])
    side_mask = np.array([False, True])
    if transform_value:
        st = skp.SplineTransformer(n_knots=spline_knots,
                                   degree=spline_degree,
                                   include_bias=False)
        new_val = st.fit_transform(predictors[:, val_mask])
        predictors = np.concatenate((new_val, predictors[:, side_mask]),
                                    axis=1)
        val_mask = np.array([True]*new_val.shape[1] + [False])
        side_mask = np.logical_not(val_mask)            
    if norm_value:
        if use_val_transform is None:
            use_val_transform = skp.StandardScaler()
        predictors[:, val_mask] = use_val_transform.fit_transform(
            predictors[:, val_mask]
        )
    if include_interaction:
        inter_term = predictors[:, val_mask]*predictors[:, side_mask]
        predictors = np.concatenate((predictors, inter_term),
                                    axis=1)
    return predictors


def split_and_save_subdicts(session_dict, folder, template='sd_{}.pkl',
                            save_size=1):
    items = list(session_dict.items())
    item_inds = range(0, len(session_dict), save_size)
    for i, start in enumerate(item_inds):
        sub_dict = dict(items[start:start+save_size])
        fname = template.format(i)
        full_path = os.path.join(folder, fname)
        pickle.dump(sub_dict, open(full_path, 'wb'))
    return i 


def _get_window(psth, tz, beg, end, times, twin=None, tstep=None):
    ct = np.expand_dims(times, 0) - np.expand_dims(tz, 1)
    mask = np.logical_and(ct >= beg, ct < end)
    psth_arr = np.stack(psth, axis=0)
    psth_arr = np.swapaxes(psth_arr, 1, 2)
    psth_masked = np.stack(list(
        psth_arr[i][mask_i] for i, mask_i in enumerate(mask)
    ), axis=0)
    counts = np.sum(psth_masked, axis=1)
    return counts


def _get_stim_reps(mat, inter, pred_func, val_ext=(-1.5, 1.5), n_pts=100,
                   link_function=None, zero_preds=None):
    vals = np.expand_dims(np.linspace(*val_ext, n_pts), 1)
    side1 = np.ones((n_pts, 1))
    side2 = -np.ones((n_pts, 1))

    args1 = np.concatenate((vals, side1), axis=1)
    args2 = np.concatenate((vals, side2), axis=1)

    pred1 = pred_func(args1)
    pred2 = pred_func(args2)

    pred1 = np.expand_dims(pred1, (0, 1))
    pred2 = np.expand_dims(pred2, (0, 1))

    if zero_preds is not None:
        if zero_preds == 'right':
            pred1 = np.concatenate((pred1, np.zeros_like(pred1)), axis=-1)
            pred2 = np.concatenate((pred2, np.zeros_like(pred2)), axis=-1)
        else:
            pred1 = np.concatenate((np.zeros_like(pred1), pred1), axis=-1)
            pred2 = np.concatenate((np.zeros_like(pred2), pred2), axis=-1)

    mat_use = np.expand_dims(mat, 2)
    if len(pred1.shape) < len(mat_use.shape):
        pred1 = np.expand_dims(pred1, -1)
        pred2 = np.expand_dims(pred2, -1)
        ax = -2
    else:
        ax = -1

    stim1 = np.swapaxes(np.sum(mat_use*pred1, axis=ax) + inter, 1, 2)
    stim2 = np.swapaxes(np.sum(mat_use*pred2, axis=ax) + inter, 1, 2)
    if link_function is not None:
        stim1 = link_function(stim1)
        stim2 = link_function(stim2)
    return stim1, stim2


nonother_div_dict = {
    'space':('subj_ev_left-subj_ev_right',
             'subj_ev_right-subj_ev_left'),
    # 'space 1':('subj_ev_left-subj_ev_right',),
    # 'space 2':('subj_ev_right-subj_ev_left',),
    'time':('subj_ev offer 1-subj_ev offer 2',
            'subj_ev offer 2-subj_ev offer 1'),
}
timeonly_div_dict = {
    'left offer':('subj_ev_left-subj_ev_left-1',),
    'right offer':('subj_ev_right-subj_ev_right-2',)
}
sideonly_div_dict = {
    'offer 1':('subj_ev_left-subj_ev_right-0',
               'subj_ev_right-subj_ev_left-0'),
    'offer 2':('subj_ev_left-subj_ev_right-3',
               'subj_ev_right-subj_ev_left-3')
}
sidetime_div_dict = {
    'offer 1':('subj_ev_left-subj_ev_right-1',
               'subj_ev_right-subj_ev_left-1'),
    'offer 2':('subj_ev_left-subj_ev_right-2',
               'subj_ev_right-subj_ev_left-2')
}
full_div_dict = {
    'time only':timeonly_div_dict,
    'side only':sideonly_div_dict,
    'side-time':sidetime_div_dict,
    'non other':nonother_div_dict,    
}


def normalize_pred_dimensions(
        data,
        target_dims=100,
        target_snr=5,
        scale_snr=True,
        scale_dims=False,
        div_dict=full_div_dict,
        region_order=('OFC', 'PCC', 'pgACC', 'VS', 'vmPFC', 'all'),
        use_other=True,
        other_key="non other",
        no_sem=True,
):
    if use_other:
        div_dict = div_dict[other_key]
    else:
        div_dict = div_dict['non other']
    for i, (div_name, dks) in enumerate(div_dict.items()):
        for j, region in enumerate(region_order):
            dl, dn, sig, sem = [], [], [], []
            for dk in dks:
                res = data[region][dk][0, 0]
                dl = res["d_l"][0, 0]
                dn = res["d_n"][0, 0]
                nn = res["n_neurs"][0, 0]
                sig = res["sigma"][0, 0]
                sem = res["sem"][0, 0]
                dn_s = np.sqrt(2) * dn

                if scale_dims:
                    dl = np.sqrt(target_dims * (dl ** 2) / nn)
                    dn_s = np.sqrt(target_dims * (dn_s ** 2) / nn)
                if scale_snr:
                    pwr = (1/2) * (dl**2 + dn_s**2)
                    factor = np.sqrt(target_snr / pwr)
                    dl = factor*dl
                    dn_s = factor*dn_s
                    sig = np.ones_like(sig)
                if no_sem:
                    sem = np.zeros_like(sem)
                dn = dn_s / np.sqrt(2)

                sem_nl = np.sqrt(dn_s**2 + sem**2)
                pred_ccgp = mrdt._compute_ccgp(
                    dl, dl, sem_nl, sig
                )
                pred_bin = mrdt._compute_bind(
                    dn_s, sig,
                )
                res["d_l"][0, 0] = dl
                res["d_n"][0, 0] = dn
                res["pred_ccgp"][0, 0] = 1 - pred_ccgp
                res["pred_bin"][0, 0] = 1 - pred_bin
                res["n_neurs"][0, 0] = np.ones_like(nn) * target_dims
                res['sem'][0, 0] = sem
                res['sigma'][0, 0] = sig

    return data


dec_cond_groups = {
    "side": (
        ('subj_ev_left', 'subj_ev_right'), ('subj_ev_right', 'subj_ev_left')
    ),
    "time": (
        ('subj_ev offer 1', 'subj_ev offer 2'), ('subj_ev offer 2', 'subj_ev offer 1')
    ),
}
def print_dec_differences(
        dec_dict, region_list=None, cond_groups=dec_cond_groups, t_comp=250, p_thr=.05,
):
    if region_list is None:    
        region_list = list(dec_dict.keys())
    comb_dict = {}
    for j, r in enumerate(region_list):
        comb_dict[r] = {}
        cond_dict = dec_dict[r]
        for i, (cond, ks) in enumerate(cond_groups.items()):
            decs = []
            gens = [] 
            for k in ks:
                res = cond_dict[k]
                dec, xs = res[:2]
                decs.append(dec)
                if len(res) > 2:
                    gen = res[-1]
                    gens.append(gen)
            dec = np.mean(decs, axis=(0, 2))
            if len(gens) > 0:
                gen = np.mean(gens, axis=(0, 2))
            comb_dict[r][cond] = (dec, gen)
    t_ind = np.argmin((xs - t_comp)**2)
    print("comparison time: {}".format(xs[t_ind]))
    n_comps = ss.comb(len(region_list), 2)
    adj_perc = 100 - 100*p_thr / n_comps
    for i, j in it.combinations(range(len(region_list)), 2):
        r1, r2 = region_list[i], region_list[j]
        for cond in cond_groups.keys():
            dec1, gen1 = comb_dict[r1][cond]
            dec2, gen2 = comb_dict[r2][cond]

            high, low = u.conf_interval(dec1 - dec2, withmean=True, perc=adj_perc)
            s = "{} - {}, {}, {:.2f} - {:.2f}".format(
                r1, r2, cond, low[t_ind], high[t_ind]
            )
            print(s)


default_test_diff = (
    "order flips, side-values same",  # temporal misbinding
    "order flips, side-values flip",  # spatial misbinding
)
def print_rdm_factorial_analysis(
        rdm_dict,
        test_diff=default_test_diff,
):
    diff_distrs = {}
    for k, (_, _, ambig_dists, _) in rdm_dict.items():
        diff = ambig_dists[test_diff[0]] - ambig_dists[test_diff[1]]
        diff_distrs[k] = diff
    n_comps = ss.comb(len(rdm_dict), 2)
    perc = 100 - 5 / n_comps
    for k1, k2 in it.combinations(rdm_dict.keys(), 2):
        
        high, low = u.conf_interval(
            diff_distrs[k1] - diff_distrs[k2], withmean=True, perc=perc,
        )[:, 0]
        print("{} - {}: {} - {}".format(k1, k2, low, high))
    return diff_distrs


def print_factorial_analysis(
        regions,
        *args,
        t_ind=-1,
        cond_names=None,
):
    all_diffs = []
    all_regions = []
    all_conds = []
    if cond_names is None:
        cond_names = list("cond {}".format(i) for i in range(len(args)))
    
    for i, data in enumerate(args):
        dec, xs, d1, d2, g1, g2, gen = data
        dec = np.mean(dec[..., t_ind:], axis=1)
        gen = np.mean(gen[..., t_ind:], axis=1)
        all_diffs.append(dec - gen)
        all_conds.append((cond_names[i],)*len(dec))
        all_regions.append(regions)
    diffs = np.concatenate(all_diffs)
    regions = np.concatenate(all_regions)
    conds = np.concatenate(all_conds)
    res_dict = {"diff": np.squeeze(diffs), "cond": conds, "region": regions}

    df = pd.DataFrame(res_dict)
    model = sm_formula.ols(
        "diff ~ C(region)*C(cond) - 1",
        data=df,
    ).fit()

    result = sm.stats.anova_lm(model)
    return result, model

null_pred = ft.partial(form_predictors,
                       norm_value=False,
                       include_interaction=False)
interaction_pred = ft.partial(form_predictors,
                              norm_value=False)

null_spline_pred = ft.partial(form_predictors,
                              norm_value=False,
                              include_interaction=False,
                              transform_value=True)
interaction_spline_pred = ft.partial(form_predictors,
                                     norm_value=False,
                                     transform_value=True)


def sample_fit_reps(
        fit_dict,
        pred_func,
        use_regions=None,
        use_monkeys=None,
        n_pts=100,
        **kwargs,
):
    if use_regions is None:
        use_regions = ('OFC', 'PCC', 'pgACC', 'VS', 'vmPFC')
    if use_monkeys is None:
        use_monkeys = ('Batman', 'Calvin', 'Hobbes', 'Pumbaa', 'Spock', 'Vader')
    mat1 = np.concatenate(list(v[0][0] for (r, m, _), v in fit_dict.items()
                               if r[0] in use_regions and m in use_monkeys),
                          axis=1)
    inter1 = np.concatenate(list(v[0][1] for (r, m, _), v in fit_dict.items()
                                 if r[0] in use_regions and m in use_monkeys),
                            axis=1)
    mat2 = np.concatenate(list(v[1][0] for (r, m, _), v in fit_dict.items()
                               if r[0] in use_regions and m in use_monkeys),
                          axis=1)
    inter2 = np.concatenate(list(v[1][1] for (r, m, _), v in fit_dict.items()
                                 if r[0] in use_regions and m in use_monkeys),
                            axis=1)
    k_list = list(
        list(zip((k[2],)*len(k[0]), range(len(k[0]))))
        for k in fit_dict.keys()
        if k[0][0] in use_regions and k[1] in use_monkeys
    )
    key_group = np.concatenate(k_list, axis=0)

    # side 1 and side 2 for models 1
    stim11, stim12 = _get_stim_reps(mat1, inter1, pred_func, n_pts=n_pts,
                                    **kwargs)

    # side 1 and side 2 for models 2
    stim21, stim22 = _get_stim_reps(mat2, inter2, pred_func, n_pts=n_pts,
                                    **kwargs)
    return (stim11, stim12), (stim21, stim22), key_group


default_key_mask = {
    "linear": ("null", "null_spline"),
    "interaction": ("interaction", "interaction_spline"),
    "noise": ("noise",),
}


def sample_all_data(
        data_dict,
        use_regions=None,
        n_samples=100,
        n_trls=5,
        use_monkeys=None,
):
    if use_regions is None:
        use_regions = ('OFC', 'PCC', 'pgACC', 'VS', 'vmPFC')
    if use_monkeys is None:
        use_monkeys = ('Batman', 'Calvin', 'Hobbes', 'Pumbaa', 'Spock', 'Vader')

    dat = list(v[1:] for (r, m, _), v in data_dict.items()
               if r[0] in use_regions and m in use_monkeys)
    data_ja = na.JaggedArray(*dat)
    cond_arrs = data_ja.split_on_element(1, require_trials=n_trls)
    samps = {
        k: cja.sample_arrays(n_samples, n_concat=n_trls)
        for k, cja in cond_arrs.items()
    }
    resp_samples = np.concatenate(
        list(v[0] for v in samps.values()),
        axis=1,
    )
    return resp_samples


def compute_split_halfs_model_mix(
        model_dict,
        model_fit_dict,
        pred_dict,
        *args,
        align_func=compute_corr,
        key_mask=default_key_mask,
        pearson_brown=False,
        full_data=None,
        n_full_data_trials=5,
        compute_null=True,
        model_combination="weighted_sum",
        size_thr=10**8,
        **kwargs,
):
    keys = list(model_dict.keys())
    stim11_samps, stim12_samps, stim21_samps, stim22_samps = [], [], [], []
    for ms in keys:
        fit_dict = model_dict[ms][0]
        pred_func = pred_dict[ms]
        out = sample_fit_reps(
            fit_dict, pred_func, *args, **kwargs
        )
        (stim11, stim12), (stim21, stim22), key_group = out
        stim11_samps.append(stim11)
        stim12_samps.append(stim12)
        stim21_samps.append(stim21)
        stim22_samps.append(stim22)
    shape = stim11.shape
    stim11_samps.append(np.zeros(shape))
    stim12_samps.append(np.zeros(shape))
    stim21_samps.append(np.zeros(shape))
    stim22_samps.append(np.zeros(shape))
    if full_data is not None and compute_null:
        full_reps = sample_all_data(
            full_data, n_trls=n_full_data_trials, **kwargs,
        )
    else:
        full_reps = None
    stim11_samps = np.stack(stim11_samps, axis=0)
    stim12_samps = np.stack(stim12_samps, axis=0)
    stim21_samps = np.stack(stim21_samps, axis=0)
    stim22_samps = np.stack(stim22_samps, axis=0)

    stim11 = np.zeros(stim11_samps.shape[1:])
    stim12 = np.zeros(stim12_samps.shape[1:])
    stim21 = np.zeros(stim21_samps.shape[1:])
    stim22 = np.zeros(stim22_samps.shape[1:])
    fk = keys + ["noise"]
    for i, (kg_str, ind) in enumerate(key_group):
        region, monkey = kg_str.split("-")[:2]
        new_key = (region, monkey, kg_str, int(ind))
        if model_combination in fk:
            ind = fk.index(model_combination)
            stim11[..., i, :] = stim11_samps[ind, ..., i, :]
            stim12[..., i, :] = stim12_samps[ind, ..., i, :]
            stim21[..., i, :] = stim21_samps[ind, ..., i, :]
            stim22[..., i, :] = stim22_samps[ind, ..., i, :]
        else:
            mf_kg = model_fit_dict.get(new_key)
            weights = mf_kg["weight"]
            simplex = {
                k: np.sum(list(weights[sm] for sm in v))
                for k, v in key_mask.items()
            }
            ms_weights = list(simplex[k] for k in fk)
            ms_weights = np.expand_dims(ms_weights, (1, 2, 3))
            stim11[..., i, :] = np.sum(ms_weights * stim11_samps[..., i, :], axis=0)
            stim12[..., i, :] = np.sum(ms_weights * stim12_samps[..., i, :], axis=0)
            stim21[..., i, :] = np.sum(ms_weights * stim21_samps[..., i, :], axis=0)
            stim22[..., i, :] = np.sum(ms_weights * stim22_samps[..., i, :], axis=0)

    if len(stim11.shape) == 4:
        dim = -2
    else:
        dim = -1
    if stim11.size > size_thr:
        rs_wi = np.zeros((stim11.shape[0], stim11.shape[-1]))
        rs_null = np.zeros_like(rs_wi)
        rs_ac1 = np.zeros_like(rs_wi)
        rs_ac2 = np.zeros_like(rs_wi)
        for i in range(stim11.shape[0]):
            rs_wi[i:i+1] = align_func(stim11[i:i+1], stim12[i:i+1], dim=dim)
            rs_null[i:i+1] = align_func(
                stim11[i:i+1],
                stim12[i:i+1],
                dim=dim,
                null=True,
                full_data=full_reps,
            )
            rs_ac1[i:i+1] = align_func(stim11[i:i+1], stim21[i:i+1], dim=dim)
            rs_ac2[i:i+1] = align_func(stim12[i:i+1], stim22[i:i+1], dim=dim)
    else:
        rs_wi = align_func(stim11, stim12, dim=dim)
        rs_null = align_func(stim11, stim12, null=True, full_data=full_reps, dim=dim)
        rs_ac1 = align_func(stim11, stim21, dim=dim)
        rs_ac2 = align_func(stim12, stim22, dim=dim)

    rs_ac = np.sqrt(rs_ac1*rs_ac2)
    if pearson_brown:
        rs_wi = 2*rs_wi/(1 + rs_wi)
        rs_ac = 2*rs_ac/(1 + rs_ac)
    return rs_wi, rs_ac, rs_null

def compute_split_halfs(
        fit_dict,
        pred_func,
        use_regions=None,
        pearson_brown=False,
        align_func=compute_corr,
        n_pts=100,
        **kwargs,
):
    out = sample_fit_reps(fit_dict, pred_func, n_pts=n_pts, use_regions=use_regions)
    (stim11, stim12), (stim21, stim22), key_group = out

    rs_wi = align_func(stim11, stim12)
    rs_ac1 = align_func(stim11, stim21)
    rs_ac2 = align_func(stim12, stim22)
    rs_ac = np.sqrt(rs_ac1*rs_ac2)

    rs_null1 = align_func(stim11, stim12, null=True)
    rs_null2 = align_func(stim21, stim22, null=True)
    rs_null = np.sqrt(rs_null1*rs_null2)
    if pearson_brown:
        rs_wi = 2*rs_wi/(1 + rs_wi)
        rs_ac = 2*rs_ac/(1 + rs_ac)
    return rs_wi, rs_ac, rs_null


def compute_angles(rdm):
    mat = rdm.get_matrices()[0]
    d01 = mat[(0, 1)]
    d02 = mat[(0, 2)]
    d12 = mat[(1, 2)]
    ang012 = np.arccos((d01**2 + d02**2 - d12**2)/(2*d01*d02))
    return ang012
    
def shared_subspace(vr1, vr2, dims=4, cv=None, n_folds=10, frac=.1):
    if cv is None:
        cv = skms.ShuffleSplit(n_folds, test_size=frac)
    vr1 = _nan_mask(vr1)
    vr2 = _nan_mask(vr2)

    print('vr1', vr1.shape)
    tot = np.concatenate((vr1, vr2), axis=0)
    ss = skp.StandardScaler().fit(tot)
    vr1 = ss.transform(vr1)
    vr2 = ss.transform(vr2)
    
    cv_gen1 = cv.split(vr1)
    cv_gen2 = cv.split(vr2)

    total_vars = np.zeros((n_folds, 2))
    trs_vars = np.zeros_like(total_vars)
    for i, (tr1_inds, te1_inds) in enumerate(cv_gen1):
        tr2_inds, te2_inds = next(cv_gen2)
        
        p1 = skd.PCA(dims)
        p1.fit(vr1[tr1_inds])
        p1_vr1 = p1.transform(vr1[te1_inds])
        p1_vr2 = p1.transform(vr2)
        p1_max = p1.transform(vr1[tr1_inds])
        print(p1_max.shape)
        print(np.var(p1_max, axis=0))
        print(p1.explained_variance_)
        
        p2 = skd.PCA(dims)
        p2.fit(vr2[tr2_inds])
        p2_vr2 = p2.transform(vr2[te2_inds])
        p2_vr1 = p2.transform(vr1)
        p2_max = p2.transform(vr2[tr2_inds])

        p1_max_sum = np.sum(np.var(p1_max, axis=0))
        p2_max_sum = np.sum(np.var(p2_max, axis=0))
        
        total_vars[i, 0] = np.sum(np.var(p1_vr1, axis=0))/p1_max_sum
        trs_vars[i, 0] = np.sum(np.var(p1_vr2, axis=0))/p1_max_sum
        total_vars[i, 1] = np.sum(np.var(p2_vr2, axis=0))/p2_max_sum
        trs_vars[i, 1] = np.sum(np.var(p2_vr1, axis=0))/p2_max_sum
        

    return total_vars, trs_vars
    
rand_splitter = skms.ShuffleSplit
def fit_gps(l_pops, r_pops, model=sklm.Ridge, norm=True,
            pca=None, rand_splitter=rand_splitter, test_prop=None,
            folds_n=20, shuffle=False, pre_pca=None, multi_task=True,
            num_inducing=40, max_fit=20,
            **model_kwargs):
    """ conds is the same length as pops, and gives the feature values """
    feats, acts = format_pops(l_pops, r_pops)
    if test_prop is None:
        test_prop = 1/folds_n
    if rand_splitter is None:
        splitter = skms.KFold(folds_n, shuffle=True)
        internal_splitter = skms.KFold(folds_n, shuffle=True)
    else:
        splitter = rand_splitter(folds_n, test_size=test_prop)
    models = np.zeros((feats.shape[0], folds_n, acts.shape[2]), dtype=object)
    te_perf = np.zeros_like(models, dtype=float)
    for i, feat in enumerate(feats[:1]):
        act = acts[i]
        not_nan_mask = ~np.isnan(act[0])
        act = act[:, not_nan_mask]
        m = model(**model_kwargs)
        for j, (tr_inds, te_inds) in enumerate(splitter.split(feat, act)):
            act_tr_i = act[tr_inds]
            feat_tr_i = feat[tr_inds]
            fit_range = min(act_tr_i.shape[1], max_fit)
            for k in range(act_tr_i.shape[1])[:fit_range]:
                m_oak = mu.oak_model(categorical_feature=[0],
                                     num_inducing=num_inducing,
                                     use_sparsity_prior=False)
                m_oak.fit(feat_tr_i, act_tr_i[:, k:k+1])

                pred = m_oak.predict(feat[te_inds])
                mu_mse = np.mean((pred - act[te_inds, k])**2)
                var = np.var(act[te_inds])
                te_perf[i, j, k] = 1 - mu_mse/var
                models[i, j, k] = m_oak
    return feats, acts, models, te_perf


def xor_analysis(data, tbeg, tend, dec1_field, dec2_field,
                 dead_perc=30, winsize=500, tstep=20,
                 pop_resamples=20, kernel='linear',
                 dec_tzf='offer_left_on',
                 gen_tzf='offer_right_on',
                 min_trials=160, pre_pca=.95,
                 shuffle_trials=True, c1_targ=2,
                 c2_targ=3, dec1_mask=None, dec2_mask=None,
                 **kwargs):
    out = _compute_masks(data, feat1, feat2, dead_perc=dead_perc,
                         use_split_dec=use_split_dec, dec_mask=f1_mask,
                         gen_mask=f2_mask, c1_targ=c1_targ, c2_targ=c2_targ)
    mask_f11, mask_f12, mask_f21, mask_f22 = out
    tzfs = (f1_tzf, f1_tzf, f2_tzf, f2_tzf)

    out = data.make_pseudo_pops(winsize, tbeg, tend, tstep,
                                mask_f11, mask_f12, mask_f21, mask_f22,
                                tzfs=tzfs,
                                regions=regions, min_trials=min_trials,
                                resamples=pop_resamples)
    xs, (pop_f11, pop_f12, pop_f21, pop_f22) = out
    if dead_perc is not None:
        out = _get_percentile_mask(data[dec1_field], dead_perc,
                                   data[dec2_field])
        mask_c11, mask_c12, mask_c21, mask_c22 = out
    else:
        mask_c11 = data[dec1_field] == c1_targ
        mask_c12 = data[dec1_field] == c2_targ
        mask_c21 = data[dec2_field] == c1_targ
        mask_c22 = data[dec2_field] == c2_targ
    if dec1_mask is not None:
        mask_c11 = mask_c11.rs_and(dec1_mask)
        mask_c12 = mask_c12.rs_and(dec1_mask)
    if dec2_mask is not None:
        mask_c21 = mask_c21.rs_and(dec2_mask)
        mask_c22 = mask_c22.rs_and(dec2_mask) 

    out = data.decode_masks(mask_c11, mask_c12, winsize, tbeg, tend, tstep, 
                            pseudo=True, time_zero_field=dec_tzf,
                            min_trials_pseudo=min_trials,
                            resample_pseudo=pop_resamples, ret_pops=True, 
                            shuffle_trials=shuffle_trials, pre_pca=pre_pca,
                            decode_tzf=gen_tzf, decode_m1=mask_c22,
                            decode_m2=mask_c21, # kernel=kernel,
                            combine=True, **kwargs)
    return out


def _compute_masks(data, dec_field, gen_field, dead_perc=30, use_split_dec=None,
                   dec_mask=None, gen_mask=None, c1_targ=2, c2_targ=3):
    if dead_perc is not None:
        if use_split_dec is not None:
            split_data_dec = data[use_split_dec]
        else:
            split_data_dec = None
        out = _get_percentile_mask(data[dec_field], dead_perc, data[gen_field],
                                   ref_data=split_data_dec)
        mask_c1, mask_c2, gen_mask_c1, gen_mask_c2 = out
    else:
        mask_c1 = data[dec_field] == c1_targ
        mask_c2 = data[dec_field] == c2_targ
        gen_mask_c1 = data[gen_field] == c1_targ
        gen_mask_c2 = data[gen_field] == c2_targ
    if dec_mask is not None:
        mask_c1 = mask_c1.rs_and(dec_mask)
        mask_c2 = mask_c2.rs_and(dec_mask)
    if gen_mask is not None:
        gen_mask_c1 = gen_mask_c1.rs_and(gen_mask)
        gen_mask_c2 = gen_mask_c2.rs_and(gen_mask)
    return mask_c1, mask_c2, gen_mask_c1, gen_mask_c2

def binding_analysis_noadd(data, tbeg, tend, feat1, feat2,
                           dead_perc=30, winsize=500, tstep=20,
                           pop_resamples=20, kernel='linear',
                           tzf='Offer 2 on', 
                           min_trials=160, pre_pca=None,
                           shuffle_trials=True, c1_targ=2, c2_targ=3,
                           f1_mask=None, f2_mask=None, use_split_dec=None,
                           regions=None, shuffle=False, mean=False,
                           n_folds=20, params=None, xor=False,
                           full_mask=None, gen_mask=None, **kwargs):
    if params is None:
        params = {'class_weight':'balanced'}
        # params.update(kwargs)
        
    out = _compute_masks(data, feat1, feat2, dead_perc=dead_perc,
                         use_split_dec=use_split_dec, dec_mask=f1_mask,
                         gen_mask=f2_mask, c1_targ=c1_targ, c2_targ=c2_targ)
    mask_f11, mask_f12, mask_f21, mask_f22 = out
    # left low, left high, right low, right high
    
    # left low AND right high
    c1_mask = mask_f11.rs_and(mask_f22)
    # left high AND right low
    c2_mask = mask_f12.rs_and(mask_f21)
    if gen_mask is not None:
        g1_mask = c1_mask.rs_and(gen_mask)
        g2_mask = c2_mask.rs_and(gen_mask)
        gens = (g1_mask, g2_mask)
        gen_tzfs = (tzf, tzf)
    else:
        gens = ()
        gen_tzfs = ()
    if full_mask is not None:
        c1_mask = c1_mask.rs_and(full_mask)
        c2_mask = c2_mask.rs_and(full_mask)
    tzfs = (tzf, tzf) + gen_tzfs
    masks = (c1_mask, c2_mask) + gens

    out = data.make_pseudo_pops(winsize, tbeg, tend, tstep,
                                *masks,
                                tzfs=tzfs,
                                regions=regions, min_trials=min_trials,
                                resamples=pop_resamples)
    if gen_mask is None:
        xs, (pop_c1, pop_c2) = out
        gen_c1 = (None,)*len(pop_c1)
        gen_c2 = (None,)*len(pop_c1)
    else:
        xs, (pop_c1, pop_c2, gen_c1, gen_c2) = out
    outs = np.zeros((len(pop_c1), n_folds, len(xs)))
    outs_gen = np.zeros((len(pop_c1), n_folds, len(xs)))
    print(pop_c1.shape, pop_c2.shape)
    for i, pop_c1_i in enumerate(pop_c1):
        out = na.fold_skl(pop_c1_i, pop_c2[i], n_folds, params=params, 
                          mean=mean, pre_pca=pre_pca, shuffle=shuffle,
                          impute_missing=False, gen_c1=gen_c1[i],
                          gen_c2=gen_c2[i],
                          **kwargs)
        if gen_mask is not None:
            out, out_gen = out
            outs_gen[i] = out_gen
        outs[i] = out
    if gen_mask is not None:
        full_out = (outs, xs, outs_gen)
    else:
        full_out = (outs, xs)
    return full_out

def binding_analysis(data, tbeg, tend, feat1, feat2,
                     dead_perc=30, winsize=500, tstep=20,
                     pop_resamples=20, kernel='linear',
                     f1_tzf='offer_left_on', f2_tzf='offer_right_on',
                     min_trials=160, pre_pca=None,
                     shuffle_trials=True, c1_targ=2, c2_targ=3,
                     f1_mask=None, f2_mask=None, use_split_dec=None,
                     regions=None, shuffle=False, mean=False,
                     n_folds=20, params=None, xor=False, **kwargs):
    if params is None:
        params = {'class_weight':'balanced'}
        # params.update(kwargs)            
    out = _compute_masks(data, feat1, feat2, dead_perc=dead_perc,
                         use_split_dec=use_split_dec, dec_mask=f1_mask,
                         gen_mask=f2_mask, c1_targ=c1_targ, c2_targ=c2_targ)
    mask_f11, mask_f12, mask_f21, mask_f22 = out
    tzfs = (f1_tzf, f1_tzf, f2_tzf, f2_tzf)

    out = data.make_pseudo_pops(winsize, tbeg, tend, tstep,
                                mask_f11, mask_f12, mask_f21, mask_f22,
                                tzfs=tzfs,
                                regions=regions, min_trials=min_trials,
                                resamples=pop_resamples)
    xs, (pop_f11, pop_f12, pop_f21, pop_f22) = out

    outs = np.zeros((len(pop_f11), n_folds, len(xs)))
    print(pop_f11.shape, pop_f12.shape, pop_f21.shape, pop_f22.shape)
    for i, pop_f11_i in enumerate(pop_f11):
        if xor:
            c1 = np.concatenate((pop_f11_i, pop_f22[i]), axis=2)
            c2 = np.concatenate((pop_f12[i], pop_f21[i]), axis=2)
        else:
            c1 = pop_f11_i + pop_f22[i]
            c2 = pop_f12[i] + pop_f21[i]
        out = na.fold_skl(c1, c2, n_folds, params=params, 
                          mean=mean, pre_pca=pre_pca, shuffle=shuffle,
                          impute_missing=False,
                          **kwargs)
        outs[i] = out[0]
    return outs, xs

def compute_within_across_corr(mus, masks):
    vecs = []
    for mask in masks:
        vecs.append(mus[mask] - mus[~mask])
    ang_dict = {}
    for (i, j) in it.product(range(len(vecs)), repeat=2):
        if i == j:
            angs = u.pairwise_cosine_similarity(vecs[i])
        else:
            angs = u.pairwise_cosine_similarity(vecs[i], vecs[j])
        ang_dict[(i, j)] = angs
    return ang_dict

def nearest_decoder_epoch_two(data, dec1_field, dec2_field,
                              dead_perc=20, use_split_dec=None,
                              min_trials=40, tzf='Offer 2 on',
                              include_opt_choice=False, choice_field='subj_ev',
                              **kwargs):
    left_mask = data['side of offer 1 (Left = 1 Right =0)'] == 1
    right_mask = data['side of offer 1 (Left = 1 Right =0)'] == 0
    out = _compute_masks(data, dec1_field, dec2_field, dead_perc=dead_perc,
                         use_split_dec=use_split_dec)
    mask_1hv, mask_1lv, mask_2hv, mask_2lv = out
    
    left_1hv_2hv = mask_1hv.rs_and(left_mask).rs_and(mask_2hv)
    left_1hv_2lv = mask_1hv.rs_and(left_mask).rs_and(mask_2lv)
    left_1lv_2hv = mask_1lv.rs_and(left_mask).rs_and(mask_2hv)
    left_1lv_2lv = mask_1lv.rs_and(left_mask).rs_and(mask_2lv)

    right_1hv_2hv = mask_1hv.rs_and(right_mask).rs_and(mask_2hv)
    right_1hv_2lv = mask_1hv.rs_and(right_mask).rs_and(mask_2lv)
    right_1lv_2hv = mask_1lv.rs_and(right_mask).rs_and(mask_2hv)
    right_1lv_2lv = mask_1lv.rs_and(right_mask).rs_and(mask_2lv)
    
    masks = (left_1hv_2hv, left_1hv_2lv, left_1lv_2hv, left_1lv_2lv,
             right_1hv_2hv, right_1hv_2lv, right_1lv_2hv, right_1lv_2lv)
    if include_opt_choice:
        c_field = choice_field + '_chosen'
        uc_field = choice_field + '_unchosen'
        opt_mask = data[c_field] >= data[uc_field]
        nonopt_mask = data[c_field] < data[uc_field]
        masks_optchoice = []
        masks_nonoptchoice = []
        for mask in masks:
            masks_optchoice.append(mask.rs_and(opt_mask))
            masks_nonoptchoice.append(mask.rs_and(nonopt_mask))
        masks = masks_optchoice + masks_nonoptchoice
    return nearest_decoder_epochs(data, masks, min_trials=min_trials, tzf=tzf,
                                  **kwargs)

def nearest_decoder_epoch_one(data, dec_field, dead_perc=20, use_split_dec=None,
                              min_trials=80, tzf='Offer 1 on',  **kwargs):
    left_mask = data['side of offer 1 (Left = 1 Right =0)'] == 1
    right_mask = data['side of offer 1 (Left = 1 Right =0)'] == 0
    out = _compute_masks(data, dec_field, dec_field, dead_perc=dead_perc,
                         use_split_dec=use_split_dec)
    mask_hv, mask_lv, _, _ = out
    left_hv = mask_hv.rs_and(left_mask)
    left_lv = mask_lv.rs_and(left_mask)
    right_hv = mask_hv.rs_and(right_mask)
    right_lv = mask_lv.rs_and(right_mask)

    masks = (left_hv, left_lv, right_hv, right_lv)
    return nearest_decoder_epochs(data, masks, min_trials=min_trials,
                                  tzf=tzf, **kwargs)


def nearest_decoder_epochs(data, masks, tbeg=100, tend=1000,
                           winsize=300, tstep=300, pop_resamples=20,
                           tzf='Offer 2 on', min_trials=80, 
                           regions=None, cv_runs=20, **kwargs):

    out = data.make_pseudo_pops(winsize, tbeg, tend, tstep, *masks,
                                tzfs=(tzf,)*len(masks),
                                min_trials=min_trials,
                                resamples=pop_resamples)
    xs, cond_pops = out
    dec_perf = np.zeros((pop_resamples, cv_runs, len(xs)))
    dec_info = np.zeros((pop_resamples, len(xs)), dtype=object)
    dec_confusion = np.zeros((pop_resamples, cv_runs, len(xs),
                              len(cond_pops), len(cond_pops)))
    for i in range(pop_resamples):
        use_pops = list(cp[i] for cp in cond_pops)
        out = na.nearest_decoder(*use_pops, norm=True, cv_runs=cv_runs,
                                 generate_confusion=True, **kwargs)
        dec_perf[i], dec_info[i], dec_confusion[i] = out
    return xs, dec_perf, dec_info, dec_confusion
    

def generalization_n_neurs(data, tbeg, tend, dec_field, gen_field,
                           dead_perc=30, winsize=500, tstep=20,
                           use_split_dec=None,
                           correct_only=False,
                           subsample_neurons=None,
                           c1_targ=2, c2_targ=3,
                           f1_mask=None, f2_mask=None,
                           regions=None,
                           **kwargs,):
    out = _compute_masks(data, dec_field, gen_field, dead_perc=dead_perc,
                         use_split_dec=use_split_dec, dec_mask=f1_mask,
                         gen_mask=f2_mask, c1_targ=c1_targ, c2_targ=c2_targ)
    mask_c1, mask_c2, gen_mask_c1, gen_mask_c2 = out
    if correct_only:
        choice_mask = data['subj_ev_chosen'] - data['subj_ev_unchosen'] > 0
        mask_c1 = mask_c1.rs_and(choice_mask)
        mask_c2 = mask_c2.rs_and(choice_mask)
    out = data.neuron_trial_tradeoff(
        mask_c1, mask_c2, regions=regions, time_zero_field="Offer 1 on",
    )
    return out

def make_trial_percentile_masks(data, split_perc, trl_field="trial_num", comb_mask=None):
    trl_nums = data[trl_field]
    tr_masks = []
    te_masks = []
    for i, tr in enumerate(trl_nums):
        thr_perc = np.percentile(tr, split_perc)
        tr_m = tr < thr_perc
        te_m = tr > thr_perc
        if comb_mask is not None:
            tr_m = np.logical_and(comb_mask[i], tr_m)
            te_m = np.logical_and(comb_mask[i], te_m)
        tr_masks.append(tr_m)
        te_masks.append(te_m)
    return tr_masks, te_masks


def compute_temporal_generalization(
        data,
        tbeg,
        tend,
        var,
        train_trl_perc=90,
        dec_offer=1,
        dead_perc=30,
        winsize=500,
        tstep=20,
        pop_resamples=20,
        kernel='linear',
        tzf='Offer {} on',
        min_trials=160,
        pre_pca=None,
        shuffle_trials=True,
        c1_targ=2,
        c2_targ=3,
        f1_mask=None,
        f2_mask=None,
        use_split_dec=None,
        correct_only=False,
        subsample_neurons=None,
        mask_var=None,
        mask_func=None,
        **kwargs,
):
    dec_suff = " offer {}".format(dec_offer)
    vm, = _make_var_masks(data, mask_var, mask_func, dec_suff)
    field = "{}{}".format(var, dec_suff)
    tzf = tzf.format(dec_offer)
    f1_mask, f2_mask = make_trial_percentile_masks(data, train_trl_perc, comb_mask=vm)
    
    out = _compute_masks(data, field, field, dead_perc=dead_perc,
                         use_split_dec=use_split_dec, dec_mask=f1_mask,
                         gen_mask=f2_mask, c1_targ=c1_targ, c2_targ=c2_targ)
    mask_c1, mask_c2, gen_mask_c1, gen_mask_c2 = out
    if correct_only:
        choice_mask = data['subj_ev_chosen'] - data['subj_ev_unchosen'] > 0
        mask_c1 = mask_c1.rs_and(choice_mask)
        mask_c2 = mask_c2.rs_and(choice_mask)
        gen_mask_c1 = gen_mask_c1.rs_and(choice_mask)
        gen_mask_c2 = gen_mask_c2.rs_and(choice_mask)
    out = data.decode_masks(
        mask_c1,
        mask_c2,
        winsize,
        tbeg,
        tend,
        tstep, 
        pseudo=True,
        time_zero_field=tzf,
        min_trials_pseudo=min_trials,
        resample_pseudo=pop_resamples,
        ret_pops=False,
        subsample_neurons=subsample_neurons,
        shuffle_trials=shuffle_trials,
        pre_pca=pre_pca,
        decode_tzf=tzf,
        decode_m1=gen_mask_c1, 
        decode_m2=gen_mask_c2,
        **kwargs
    )
    return out


def generalization_analysis(data, tbeg, tend, dec_field, gen_field,
                            dead_perc=30, winsize=500, tstep=20,
                            pop_resamples=20, kernel='linear',
                            f1_tzf='offer_left_on',
                            f2_tzf='offer_right_on',
                            min_trials=160, pre_pca=None,
                            shuffle_trials=True, c1_targ=2,
                            c2_targ=3, f1_mask=None, f2_mask=None,
                            use_split_dec=None,
                            correct_only=False,
                            subsample_neurons=None,
                            **kwargs):
    out = _compute_masks(data, dec_field, gen_field, dead_perc=dead_perc,
                         use_split_dec=use_split_dec, dec_mask=f1_mask,
                         gen_mask=f2_mask, c1_targ=c1_targ, c2_targ=c2_targ)
    mask_c1, mask_c2, gen_mask_c1, gen_mask_c2 = out
    if correct_only:
        choice_mask = data['subj_ev_chosen'] - data['subj_ev_unchosen'] > 0
        mask_c1 = mask_c1.rs_and(choice_mask)
        mask_c2 = mask_c2.rs_and(choice_mask)
        gen_mask_c1 = gen_mask_c1.rs_and(choice_mask)
        gen_mask_c2 = gen_mask_c2.rs_and(choice_mask)
    out = data.decode_masks(
        mask_c1,
        mask_c2,
        winsize,
        tbeg,
        tend,
        tstep, 
        pseudo=True,
        time_zero_field=f1_tzf,
        min_trials_pseudo=min_trials,
        resample_pseudo=pop_resamples,
        ret_pops=True,
        subsample_neurons=subsample_neurons,
        shuffle_trials=shuffle_trials,
        pre_pca=pre_pca,
        decode_tzf=f2_tzf,
        decode_m1=gen_mask_c1, 
        decode_m2=gen_mask_c2,
        **kwargs
    )
    return out

def regression_gen(pops_tr, rts_tr, pops_te, rts_te, model=sklm.Ridge,
                   norm=True, pca=None):
    outs = np.zeros(len(pops_tr))
    pipe = na.make_model_pipeline(model, norm=norm, pca=pca)
    for i, pop_tr in enumerate(pops_tr):
        rt_tr = rts_tr[i]
        pipe.fit(pop_tr, rt_tr)
        outs[i] = pipe.score(pops_te[i], rts_te[i])
    return outs

def discretize_classifier_cv(pops, rts, model=skc.SVC, **kwargs):
    new_rts = list(rts_i > np.median(rts_i) for rts_i in rts)
    return regression_cv(pops, new_rts, model=model, **kwargs)

def discretize_classifier_gen(pops_tr, rts_tr, pops_te, rts_te,
                              model=skc.SVC, **kwargs):
    meds = list(np.median(rt_tr_i) for rt_tr_i in rts_tr)
    new_rts_tr = list(rts_i > meds[i] for i, rts_i in enumerate(rts_tr))
    new_rts_te = list(rts_i > meds[i] for i, rts_i in enumerate(rts_te))
    return regression_gen(pops_tr, new_rts_tr, pops_te, new_rts_te,
                          model=model, **kwargs)

def reformat_dict(out_dict, keep_inds=(0, 1, -1),
                  keep_labels=('dec', 'xs', 'gen'),
                  save_dict=True, str_key=True,
                  keep_subset=True):
    new_dict = {}
    for key, val in out_dict.items():
        if keep_subset:
            if save_dict:
                sub_item = {}
                for i, keep_ind in enumerate(keep_inds):
                    sub_item[keep_labels[i]] = val[keep_ind]
            else:
                sub_item = []
                for i, keep_ind in enumerate(keep_inds):
                    sub_item.append(val[keep_ind])
        else:
            sub_item = val                    
        if str_key:
            nk = '-'.join(str(k_i) for k_i in key)
        else:
            nk = key
        new_dict[nk] = sub_item
    return new_dict

def reformat_generalization_loss(out_dict, save_dict=True):
    loss_dict = {}
    for (dec, gen), val in out_dict.items():
        p, g = val[0], val[-1]
        val_flip = out_dict[(gen, dec)]
        p_flip, g_flip = val_flip[0], val_flip[-1]
        xs = val_flip[1]
        if save_dict:
            loss_dict[dec] = {'dec':p, 'xs':xs, 'gen':g_flip}
        else:
            loss_dict[dec] = (p, xs, g_flip)
    return loss_dict


def _prob_side_tune(data, f1, f2, side=0, prob_less=1):
    prob_m = data[f1] < prob_less
    side_m = data[f2] == side
    return prob_m.rs_and(side_m)

cond_suffixes = (('_left', '_right', 'prob',
                  ['side of offer 1 (Left = 1 Right =0)']),
                 ('_right', '_left', 'prob',
                  ['side of offer 1 (Left = 1 Right =0)']),
                 ('_left', '_left', 'prob',
                  ['side of offer 1 (Left = 1 Right =0)']),
                 ('_right', '_right', 'prob',
                  ['side of offer 1 (Left = 1 Right =0)']))
cond_timing = (('offer_left_on', 'offer_right_on'),
               ('offer_right_on', 'offer_left_on'),
               ('offer_left_on', 'offer_left_on'),
               ('offer_right_on', 'offer_right_on'))
cond_funcs = (
    (ft.partial(_prob_side_tune, side=1),  # side 1/offer 1 then side 2/offer 2
     ft.partial(_prob_side_tune, side=1)),
    
    (ft.partial(_prob_side_tune, side=1),  # side 1/offer 1 then side 2/offer 1
     ft.partial(_prob_side_tune, side=0)),
    
    (ft.partial(_prob_side_tune, side=0),  # side 1/offer 2 then side 2/offer 2
     ft.partial(_prob_side_tune, side=1)),
    
    (ft.partial(_prob_side_tune, side=0),  # side 2/offer 1 then side 1/offer 2
     ft.partial(_prob_side_tune, side=0))
)
def compute_conditional_generalization(data, tbeg, tend, dec_var,
                                       conditional_funcs=cond_funcs,
                                       suffixes=cond_suffixes,
                                       timing=cond_timing, 
                                       compute_reverse=True,
                                       **kwargs):
    out_dict = {}
    for i, (dec_suff, gen_suff, mask_var, other_mask) in enumerate(suffixes):
        dec_field = dec_var + dec_suff
        gen_field = dec_var + gen_suff
        mask_dec_field = mask_var + dec_suff
        mask_gen_field = mask_var + gen_suff

        full_dec_mask = [mask_dec_field] + other_mask
        full_gen_mask = [mask_gen_field] + other_mask

        dec_tzf, gen_tzf = timing[i]
        print(dec_field, dec_tzf)
        print(gen_field, gen_tzf)
        for j, cf in enumerate(cond_funcs):
            conditional_func_dec, conditional_func_gen = cond_funcs[j]
            dec_mask = conditional_func_dec(data, *full_dec_mask)
            gen_mask = conditional_func_gen(data, *full_gen_mask)

            out = generalization_analysis(data, tbeg, tend, dec_field, gen_field,
                                          f1_mask=dec_mask, f2_mask=gen_mask,
                                          f1_tzf=dec_tzf, f2_tzf=gen_tzf,
                                          **kwargs)
            out_dict[(dec_field, gen_field, j)] = out
    return out_dict      


default_suffixes = (('_chosen', '_unchosen'), ('_left', '_right'),
                    (' offer 1', ' offer 2'))
default_suffixes = (('_left', '_right'),
                    (' offer 1', ' offer 2'))


def _make_var_masks(data, mask_var, mask_func, *mask_fields, suffs=True):
    if mask_var is None and mask_func is None:
        masks = list(None for mf in mask_fields)
    else:
        if suffs:
            mask_fields = list(mask_var + mf for mf in mask_fields)
        if mask_func is not None:
            masks = list(mask_func(data[mf]) for mf in mask_fields)
        else:
            masks = list(None for mf in mask_fields)
    return masks


default_timing = (('offer_chosen_on', 'offer_unchosen_on'),
                  ('offer_left_on', 'offer_right_on'),
                  ('Offer 1 on', 'Offer 2 on'))
default_timing = (('offer_left_on', 'offer_right_on'),
                  ('Offer 1 on', 'Offer 2 on'))
def _compute_all_funcs(data, tbeg, tend, dec_var, func,
                       suffixes=default_suffixes,
                       timing=default_timing, mask_func=None,
                       compute_reverse=True, mask_var=None,
                       tzf_key=False, **kwargs):
    out_dict = {}
    for i, (dec_suff, gen_suff) in enumerate(suffixes):
        dec_field = dec_var + dec_suff
        gen_field = dec_var + gen_suff
        dec_mask, gen_mask = _make_var_masks(
            data, mask_var, mask_func, dec_suff, gen_suff
        )
        dec_tzf, gen_tzf = timing[i]
        out = func(data, tbeg, tend, dec_field, gen_field,
                   f1_mask=dec_mask, f2_mask=gen_mask,
                   f1_tzf=dec_tzf, f2_tzf=gen_tzf, **kwargs)
        if tzf_key:
            key = ((dec_field, dec_tzf), (gen_field, gen_tzf))
        else:
            key = (dec_field, gen_field)
        out_dict[key] = out
        if compute_reverse:
            out = func(data, tbeg, tend, gen_field, dec_field,
                       f1_mask=gen_mask, f2_mask=dec_mask,
                       f1_tzf=gen_tzf, f2_tzf=dec_tzf, **kwargs)
            if tzf_key:
                key = ((gen_field, gen_tzf), (dec_field, dec_tzf))
            else:
                key = (gen_field, dec_field)
            out_dict[key] = out
    return out_dict  


def compute_all_binding(*args, **kwargs):
    return _compute_all_funcs(*args, binding_analysis,
                              **kwargs)

def compute_all_generalizations(*args, **kwargs):
    return _compute_all_funcs(*args, generalization_analysis,
                              **kwargs)

def compute_all_generalization_n_neurs(*args, **kwargs):
    return _compute_all_funcs(*args, generalization_n_neurs,
                              **kwargs)
    

def compute_all_xor(*args, **kwargs):
    return _compute_all_funcs(*args, xor_analysis,
                              **kwargs)

def compute_time_dec(*args, **kwargs):
    suffixes = (
        (" offer 1", " offer 2"),
        (" offer 1", " offer 1"),
    )
    timing = (
        ("Offer 2 on", "Offer 2 on"),
        ("Offer 1 on", "Offer 2 on"),
    )
    return _compute_all_funcs(
        *args,
        generalization_analysis,
        timing=timing,
        suffixes=suffixes,
        tzf_key=True,
        **kwargs,
    )


def compute_timewindow_dec(
        *args, winsize=500, offset=0, dec_pt=None, tstep=20, **kwargs
):
    if dec_pt is None:
        dec_pt = offset + winsize / 2
    beg_pt = dec_pt - (winsize + tstep) / 2
    end_pt = dec_pt + (winsize + tstep) / 2
    return compute_time_dec(
        *args,
        dec_beg=beg_pt,
        dec_end=end_pt,
        winsize=winsize,
        tstep=tstep,
        collapse_time=True,
        **kwargs,
    )


def _combine_pops(*args):
    n_pops = len(args[0])
    out = []
    for i in range(n_pops):
        for j in range(len(args)):
            if j == 0:
                out.append(args[j][i])
            else:
                out[i] = np.concatenate((out[i], args[j][i]), axis=0)
    return out           

def _estimate_params(pop, cond, est, scaler, splitter, folds_n,
                     multi_task=True):
    if multi_task:
        r2s = np.zeros(folds_n)
    else:
        r2s = np.zeros((folds_n, pop.shape[1]))
        r2s[:] = np.nan
    coefs = np.zeros((folds_n, pop.shape[1], cond.shape[1]))
    resids = np.zeros((folds_n, pop.shape[1]))
    scaler.fit(pop)
    for i, (tr, te) in enumerate(splitter.split(pop)):
        norm_tr = scaler.transform(pop[tr])
        norm_te = scaler.transform(pop[te])
        if multi_task:
            est.fit(cond[tr], norm_tr)
            r2s[i] = est.score(cond[te], norm_te)
            pred = est.predict(cond[te])
            coefs_i = est.coef_
        else:
            r2s_i = np.zeros(norm_tr.shape[1])
            pred = np.zeros_like(norm_te)
            coefs_i = np.zeros((norm_tr.shape[1], cond.shape[1]))
            for j in range(norm_tr.shape[1]):
                est.fit(cond[tr], norm_tr[:, j])
                r2s_i[j] = est.score(cond[te], norm_te[:, j])
                pred[:, j] = est.predict(cond[te])
                coefs_i[j] = est.coef_
            r2s[i, :r2s_i.shape[0]] = r2s_i
        resids[i, :pred.shape[1]] = np.mean((pred - norm_te)**2, axis=0)
        resids[i, pred.shape[1]:] = np.nan
        coefs[i, :pred.shape[1]] = coefs_i
        coefs[i, pred.shape[1]:] = np.nan
    return coefs, resids, r2s

def fit_simple_linear_models(pops, conds, model=sklm.LinearRegression,
                             norm=True, pre_pca=None, time_ind=None,
                             **model_kwargs):
    """ conds is the same length as pops, and gives the feature values """
    if time_ind is None:
        pops = list(mraux.accumulate_time(pop) for pop in pops)
    else:
        pops = list(pop[..., time_ind:time_ind+1] for pop in pops)

    ohe = skp.OneHotEncoder(sparse=False)
    ohe.fit(np.arange(len(conds)).reshape(-1, 1))
    scaler = na.make_model_pipeline(norm=True, pca=pre_pca)
    pops_full = np.concatenate(pops, axis=3)
    lin_conds = list((tuple(ci),)*pops[i].shape[3]
                     for i, ci in enumerate(conds))
    nonlin_conds = list((tuple(ohe.transform([[i]])[0]),)*pops[i].shape[3]
                        for i, ci in enumerate(conds))
    lin_full = np.concatenate(lin_conds, axis=0)
    lin_full = lin_full/np.sqrt(np.mean(np.sum(lin_full**2, axis=1)))
    nonlin_full = np.concatenate(nonlin_conds, axis=0)
    nonlin_full = nonlin_full/np.sqrt(np.mean(np.sum(nonlin_full**2, axis=1)))

    cond_full = np.concatenate((lin_full, nonlin_full), axis=1)
    n_pops, n_neurs = pops_full.shape[:2]
    coeffs = np.zeros((n_pops, n_neurs, cond_full.shape[1]))
    for i, pop in enumerate(pops_full):
        pop = np.squeeze(pop)
        if norm or pre_pca is not None:
            pipe = na.make_model_pipeline(norm=norm, pca=pre_pca)
            pop = pipe.fit_transform(pop.T).T
        for j in range(pop.shape[0]):
            m = model(fit_intercept=False)
            # use_pop = skp.StandardScaler().fit_transform(pop[j])
            print(cond_full.shape)
            print(pop[j].shape)
            m.fit(cond_full, pop[j])
            coeffs[i, j] = m.coef_
    u_conds = np.unique(cond_full, axis=0)
    return u_conds, pop, coeffs


rand_splitter = skms.ShuffleSplit
def fit_linear_models(pops, conds, model=sklm.MultiTaskElasticNetCV, norm=True,
                      pca=None, rand_splitter=rand_splitter, test_prop=None,
                      folds_n=20, shuffle=False, pre_pca=None, multi_task=True,
                      internal_cv=True, **model_kwargs):
    """ conds is the same length as pops, and gives the feature values """
    if test_prop is None:
        test_prop = 1/folds_n
    ohe = skp.OneHotEncoder(sparse=False)
    ohe.fit(np.arange(len(conds)).reshape(-1, 1))
    scaler = na.make_model_pipeline(norm=True, pca=pre_pca)
    pops_full = np.concatenate(pops, axis=3)
    lin_conds = list((tuple(ci),)*pops[i].shape[3]
                     for i, ci in enumerate(conds))
    nonlin_conds = list((tuple(ohe.transform([[i]])[0]),)*pops[i].shape[3]
                        for i, ci in enumerate(conds))
    lin_full = np.concatenate(lin_conds, axis=0)
    lin_full = lin_full/np.sqrt(np.mean(np.sum(lin_full**2, axis=1)))
    nonlin_full = np.concatenate(nonlin_conds, axis=0)
    nonlin_full = nonlin_full/np.sqrt(np.mean(np.sum(nonlin_full**2, axis=1)))

    cond_full = np.concatenate((lin_full, nonlin_full), axis=1)
    lin_mats = np.zeros((pops_full.shape[0], folds_n, pops_full.shape[1],
                         lin_full.shape[1],
                         pops_full.shape[-1]))
    nonlin_mats = np.zeros((pops_full.shape[0], folds_n, pops_full.shape[1],
                            nonlin_full.shape[1],
                            pops_full.shape[-1]))
    resid = np.zeros((pops_full.shape[0], folds_n, pops_full.shape[1],
                      pops_full.shape[-1]))
    if multi_task:
        r2 = np.zeros((pops_full.shape[0], folds_n,
                       pops_full.shape[-1]))
    else:
        r2 = np.zeros((pops_full.shape[0], folds_n,
                       pops_full.shape[1],
                       pops_full.shape[-1]))
    if rand_splitter is None:
        splitter = skms.KFold(folds_n, shuffle=True)
        internal_splitter = skms.KFold(folds_n, shuffle=True)
    else:
        splitter = rand_splitter(folds_n, test_size=test_prop)
        internal_splitter = rand_splitter(folds_n, test_size=test_prop)
    if internal_cv:
        m = model(fit_intercept=False, cv=internal_splitter, **model_kwargs)
    else:
        m = model(fit_intercept=False,  **model_kwargs)
    for i, pop in enumerate(pops_full):
        for j in range(pops_full.shape[-1]):
            out = _estimate_params(pops_full[i, :, 0, :, j].T,
                                   cond_full, m, scaler, splitter,
                                   folds_n, multi_task=multi_task)
            coef, resid_ij, r2_ij = out
            lin_mats[i, :, :, :, j] = coef[..., :len(conds[0])]
            nonlin_mats[i, :, :, :, j] = coef[..., len(conds[0]):]
            resid[i, :, :, j] = resid_ij
            r2[i, ..., j] = r2_ij
    return lin_mats, nonlin_mats, resid, r2


def predict_asymp_dists(lm, nm, sigma, k=None, n=2, n_stim=2):
    if len(lm.shape) == 2:
        lm = np.expand_dims(lm, -1)
    if len(nm.shape) == 2:
        nm = np.expand_dims(nm, -1)
    if len(sigma.shape) == 2:
        sigma = np.expand_dims(sigma, -1)
    
    d_ll = np.sqrt(np.mean(np.nansum((lm/sigma)**2, axis=0), axis=0))
    d_n =  np.sqrt(np.mean(np.nansum((nm/sigma)**2, axis=0), axis=0))
    f3 = -(d_ll**2)/(2*np.sqrt(d_ll**2 + d_n**2))
    gen = sts.norm(0, 1).cdf(f3)
    if k is None:
        k = lm.shape[1] + 1
        
    pwrs = d_ll**2 + d_n**2
    t = (d_ll**2)/pwrs
    arg_pwr = np.sqrt(pwrs)
    err_types = mrt.get_ccgp_error_tradeoff_theory(arg_pwr, k, n,
                                                   n_stim, t)
    bind = err_types[2][-1]
    return bind, gen

def predict_ccgp_binding_noformat(lm, nlm, resid_var=None, n=2, k=None,
                                  n_stim=2):
    if resid_var is None:
        resid_var = 1
    lin_pwr = np.mean(np.sum((lm/resid_var)**2, axis=0))
    nonlin_pwr = np.mean(np.sum((nlm/resid_var)**2, axis=0))
    if k is None:
        k = lm.shape[1]
    pwr = lin_pwr + nonlin_pwr
    t = lin_pwr / pwr
    pwr_s = np.sqrt(pwr)
    out = mrt.get_ccgp_error_tradeoff_theory(pwr_s, k, n, n_stim, t)
    
    err_theory, gen_theory, ts = out
    bind_avg = ts[-1][0, 0]
    gen_avg = np.squeeze(gen_theory)
    return bind_avg, gen_avg

def predict_ccgp_binding(lm, nlm, resid_var, n=2, n_stim=2):
    resid = np.nanmean(np.expand_dims(np.sqrt(resid_var), -2), axis=0)
    resid = np.expand_dims(np.sqrt(resid_var), -2)
    lm_sqr = np.nansum((lm/resid)**2, axis=2)
    lin_pwr = np.nanmean(lm_sqr, axis=(0, 1, 2))
    nlm_sqr = np.nansum((nlm/resid)**2, axis=2)
    nonlin_pwr = np.nanmean(nlm_sqr, axis=(0, 1, 2))

    k = lm.shape[3]
    pwr = lin_pwr + nonlin_pwr
    # print(lin_pwr, nonlin_pwr)
    t = lin_pwr / pwr
    pwr_s = np.sqrt(pwr)
    out = mrt.get_ccgp_error_tradeoff_theory(pwr_s, k, n, n_stim, t)
    
    err_theory, gen_theory, ts = out
    return np.squeeze(ts[-1]), np.squeeze(gen_theory)

def _get_percentile_chunks(*field_datas, n_chunks=10, ref_data=None):
    if ref_data is None:
        ref_data = field_datas[0]
    ref_data = np.concatenate(ref_data)
    chunk_bounds = np.linspace(0, 100, n_chunks + 1)
    cb_vals = np.percentile(ref_data, chunk_bounds)
    masks = list([] for _ in field_datas)
    for i, lb in enumerate(cb_vals[:-1]):
        ub = cb_vals[i + 1]
        for j, fd in enumerate(field_datas):
            mask_ij = (fd >= lb).rs_and(fd < ub)
            masks[j].append(mask_ij)
    return masks

def regression_data_pseudopop(data, tbeg, tend, targ1, targ2,
                              t1_tzf='offer_left_on', t2_tzf='offer_right_on',
                              winsize=500, min_trials=10, n_percentiles=10,
                              dec_mask=None, gen_mask=None, **kwargs):
    if dec_mask is not None:
        ref_data = (data.mask(dec_mask))[targ1]
    else:
        ref_data = None
    out = _get_percentile_chunks(data[targ1], data[targ2],
                                 n_chunks=n_percentiles,
                                 ref_data=ref_data)
    t1_masks, t2_masks = out
    if dec_mask is not None:
        t1_masks = list(cdm.rs_and(dec_mask) for cdm in t1_masks)
    if gen_mask is not None:
        t2_masks = list(cgm.rs_and(gen_mask) for cgm in t2_masks)

    out = data.regress_discrete_masks(chunk_dec_masks, winsize, tbeg, tend,
                                      winsize, time_zero_field=dec_tzf,
                                      min_trials_pseudo=min_trials,
                                      pseudo=True, decode_tzf=gen_tzf,
                                      decode_masks=chunk_gen_masks,
                                      **kwargs)




    
def regression_gen_pseudopop(data, tbeg, tend, dec_targ, gen_targ,
                             dec_tzf='offer_left_on', gen_tzf='offer_right_on',
                             winsize=500, min_trials=10, n_percentiles=10,
                             dec_mask=None, gen_mask=None, **kwargs):
    if dec_mask is not None:
        ref_data = (data.mask(dec_mask))[dec_targ]
    else:
        ref_data = None
    out = _get_percentile_chunks(data[dec_targ], data[gen_targ],
                                 n_chunks=n_percentiles,
                                 ref_data=ref_data)
    chunk_dec_masks, chunk_gen_masks = out
    if dec_mask is not None:
        chunk_dec_masks = list(cdm.rs_and(dec_mask) for cdm in chunk_dec_masks)
    if gen_mask is not None:
        chunk_gen_masks = list(cgm.rs_and(gen_mask) for cgm in chunk_gen_masks)
    out = data.regress_discrete_masks(chunk_dec_masks, winsize, tbeg, tend,
                                      winsize, time_zero_field=dec_tzf,
                                      min_trials_pseudo=min_trials,
                                      pseudo=True, decode_tzf=gen_tzf,
                                      decode_masks=chunk_gen_masks,
                                      **kwargs)
    return out
                                

def pop_regression(data, tbeg, tend, tzfs=('offer_left_on', 'offer_right_on'),
                   targs=('ev_left', 'ev_right'), min_neurs=10,
                   gen_func=regression_gen, cv_func=regression_cv,
                   do_classification=False):
    if do_classification:
        cv_func = discretize_classifier_cv
        gen_func = discretize_classifier_gen
    prod = it.product(range(len(tzfs)), repeat=2)
    pop_mask = data['n_neurs'] > min_neurs
    data = data.session_mask(pop_mask)
    n_pops = np.sum(pop_mask)
    out = np.zeros((len(tzfs), len(tzfs), n_pops))
    labels = np.zeros((len(tzfs), len(tzfs), 2), dtype=object)
    combs = {}
    for (i1, i2) in prod:
        if i1 == i2:
            pops = data.get_psth_window(tbeg, tend, time_zero_field=tzfs[i1])
            rtarg = data[targs[i1]]
            out[i1, i2] = cv_func(pops, rtarg)
        else:
            pops_i1 = data.get_psth_window(tbeg, tend, time_zero_field=tzfs[i1])
            pops_i2 = data.get_psth_window(tbeg, tend, time_zero_field=tzfs[i2])
            rtarg_i1 = data[targs[i1]]
            rtarg_i2 = data[targs[i2]]
            out[i1, i2] = gen_func(pops_i1, rtarg_i1, pops_i2, rtarg_i2)
            pops_comb = _combine_pops(pops_i1, pops_i2)
            rtarg_comb = _combine_pops(rtarg_i1, rtarg_i2)
            combs[(i1, i2)] = cv_func(pops_comb, rtarg_comb)
        labels[i1, i2] = (tzfs[i1], tzfs[i2])
    return out, combs, labels

