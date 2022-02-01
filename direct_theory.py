
import numpy as np
import sklearn.model_selection as skms
import scipy.stats as sts

import general.utility as u
import general.neural_analysis as na

def _accumulate_time(pop, keepdim=True, ax=1):
    out = np.concatenate(list(pop[..., i] for i in range(pop.shape[-1])),
                         axis=ax)
    if keepdim:
        out = np.expand_dims(out, -1)
    return out

def _est_noise(*pops, proj_ax=None, intercept=0, trl_ax=1, feat_ax=0,
               collapse=True):
    resids = []
    for pop in pops:
        if proj_ax is not None:
            pop = np.sum(proj_ax*(pop - intercept), axis=feat_ax, keepdims=True)
        # m_pop = np.mean(pop, axis=trl_ax, keepdims=True)
        # var = np.mean((pop - m_pop)**2, axis=trl_ax, keepdims=True)
        var = np.var(pop, axis=trl_ax, keepdims=True)
        if collapse:
            var = np.mean(var, axis=feat_ax)
        resids.append(var)
    return np.mean(resids, axis=0)

def _get_pair_ax(pop1, pop2, noise=1, trl_ax=1, feat_ax=0):
    intercept = np.mean(pop2, axis=trl_ax, keepdims=True)
    f1_l = (np.mean(pop1, axis=trl_ax, keepdims=True)
            - intercept)/noise
    d_pair = np.sqrt(np.nansum(f1_l**2, axis=feat_ax))
    f1_pair = u.make_unit_vector(f1_l.T).T
    f1_pair = np.expand_dims(f1_pair, 1)
    return f1_pair, d_pair, intercept

def _get_shared_ax(pops1, pops2, trl_ax=1, feat_ax=0, intercept=None):
    p1_mu = list(np.mean(pi, axis=trl_ax, keepdims=True)
                 for pi in pops1)
    p2_mu = list(np.mean(pi, axis=trl_ax, keepdims=True)
                         for pi in pops2)
    vec = ((p1_mu[0] - p2_mu[0]) + (p1_mu[1] - p2_mu[1]))/2
    if intercept is None:
        intercept = p2_mu
    vec_unit = np.expand_dims(u.make_unit_vector(vec.T).T, trl_ax)
    dists1 = _get_shared_dists(vec_unit, *pops1, intercept=intercept)
    dists2 = _get_shared_dists(vec_unit, *pops2, intercept=intercept)
    return vec_unit, dists1, dists2, intercept

def _get_shared_dists(vec, *pops, intercept=0, feat_ax=0):
    dists = []
    for i, p_i in enumerate(pops):
        dists.append(np.sum((p_i - intercept)*vec, axis=feat_ax,
                            keepdims=True))
    return dists    

def _proj_pops(ax, *pops, trl_ax=1, feat_ax=0, pre_mean=True,
               intercept=0):
    if pre_mean:
        new_pops = []
        for pop in pops:
            new_pops.append(np.mean(pop, axis=trl_ax, keepdims=True))
        pops = new_pops
    proj_dists = []
    for pop in pops:
        proj_dists.append(np.sum((pop - intercept)*ax, axis=feat_ax))
    return proj_dists        

def _proj_distance(pop1, pop2, ax, offset=0, intercept=0,
                   trl_ax=1, feat_ax=0, pre_mean=True):
    if pre_mean:
        pop1 = np.mean(pop1, axis=trl_ax, keepdims=True)
        pop2 = np.mean(pop2, axis=trl_ax, keepdims=True)
    diff_proj = np.sum((pop1 - pop2)*ax, axis=feat_ax)
    # p1_proj = np.sum((pop1 - intercept)*ax, axis=feat_ax) 
    # p2_proj = np.sum((pop2 - intercept)*ax, axis=feat_ax)
    # c1 = np.mean(p1_proj, axis=trl_ax - 1)
    # c2 = np.mean(p2_proj, axis=trl_ax - 1)
    return diff_proj # (c1 - c2)

def _reprojection_diffs(axs, pops, feat_ax=0, trl_ax=1):
    dists = []
    axs = np.array(axs)[..., 0]
    for pop in pops:
        pop_m = np.mean(pop, axis=trl_ax)
        pop_reproj = np.dot(axs.T, np.dot(axs, pop_m))
        dist = np.sqrt(np.sum((pop_m - pop_reproj)**2, axis=feat_ax))
        dists.append(dist)
    return np.mean(dists)

def _proj_diffs(ax1, ax2, *pds, intercept=0, trl_ax=1, feat_ax=0):
    ds = []
    for (pop, dist1, dist2) in pds:
        proj_cent1 = np.mean(ax1*dist1, axis=trl_ax, keepdims=True)
        proj_cent2 = np.mean(ax2*dist2, axis=trl_ax, keepdims=True)
        proj_cent = proj_cent1 + proj_cent2
        pop_cent = np.mean(pop - intercept, axis=trl_ax, keepdims=True)
        ds.append(np.sqrt(np.sum((pop_cent - proj_cent)**2, axis=feat_ax)))
    return ds

def _compute_ccgp(d_ll, d_lg, d_n, sigma):
    f3 = (d_ll*d_lg - .5*d_ll**2)/(sigma*np.sqrt(d_ll**2 + d_n**2))
    out = sts.norm(0, 1).cdf(-f3)
    return out

def direct_ccgp_bind_est_pops(train_pops, test_pops, n_folds=5, test_prop=.1,
                              **kwargs):
    n_pops = len(train_pops[0])
    bind_ests = np.zeros((n_pops, n_folds))
    gen_ests = np.zeros_like(bind_ests)
    d_ll, d_lg, d_n, sigma = list(np.zeros_like(bind_ests)
                                  for i in range(4))
    for i in range(n_pops):
        tr_ps = list(tp[i] for tp in train_pops)
        te_ps = list(tp[i] for tp in test_pops)
        # out = direct_ccgp_bind_est(tr_ps, te_ps, **kwargs)
        if test_prop == 0:
            gen_ests[i] = _mech_ccgp(tr_ps, te_ps, **kwargs)
        else:
            gen_ests[i] = very_direct_ccgp(tr_ps, te_ps, n_folds=n_folds,
                                           test_prop=test_prop, **kwargs)
        # d_ll[i], d_lg[i], d_n[i], sigma[i] = out
        # gen_ests[i] = _compute_ccgp(*out)
    return bind_ests, gen_ests, (d_ll, d_lg, d_n, sigma)

def _preprocess(*pops, fit=True, norm=True, pre_pca=.99, pipe=None,
                sigma_est=None, trl_ax=1, cent=None):
    if fit and pipe is None and (norm or pre_pca is not None):
        pipe = na.make_model_pipeline(norm=norm, pca=pre_pca)
    if fit and pipe is not None:
        full_pop = np.concatenate(pops, axis=trl_ax)
        pipe.fit(full_pop.T)
    if pipe is not None:
        preproc_pops = list(pipe.transform(p_i.T).T for p_i in pops)
    else:
        preproc_pops = pops
    if sigma_est is None:
        sigma_est_comp = np.sqrt(_est_noise(*preproc_pops, collapse=False))
    else:
        sigma_est_comp = sigma_est
    sigma_est_comp = 1 
    if cent is None:
        cent = np.mean(preproc_pops[0], axis=trl_ax, keepdims=True)
    out_pops = list((p_i - cent)/sigma_est_comp for p_i in preproc_pops)
    out = (out_pops, pipe, cent)
    if sigma_est is None:
        out = out + (sigma_est_comp,)
    return out

def _compute_cb_quantities(train_pops, test_pops, norm=True, pre_pca=.99,
                           trl_ax=1, feat_ax=0):
    out = _preprocess(*(train_pops + test_pops), norm=norm, pre_pca=pre_pca,
                      trl_ax=trl_ax)
    out_pops, pipe, cent, sigma_est = out
    train_pops, test_pops = out_pops[:2], out_pops[2:]
    out = _get_shared_ax((train_pops[0], test_pops[0]),
                         (train_pops[1], test_pops[1]))
    f1_full, c1_dists, c2_dists, intercept = out
    train_dists = (c1_dists[0], c2_dists[0])
    test_dists = (c1_dists[1], c2_dists[1])
    
    out = _get_shared_ax(train_pops, test_pops, intercept=intercept)
    f2_full, train_f2_dists, test_f2_dists, f2_intercept = out
    f1_spec, _, dec_intercept = _get_pair_ax(*train_pops)
    out = (pipe, f1_full, f2_full, f1_spec, dec_intercept, intercept, cent,
           (train_dists, test_dists, train_f2_dists, test_f2_dists),
           sigma_est)
    return out
    
def _compute_cv_quantities(train_pops, test_pops, pipe, f1_full, f2_full,
                           f1_spec, dists, dec_intercept=0, intercept=0, cent=0,
                           trl_ax=1, feat_ax=0, sigma_est=1):
    out = _preprocess(*(train_pops + test_pops), pipe=pipe, fit=False,
                      sigma_est=sigma_est, trl_ax=trl_ax, cent=cent)
    train_pops, test_pops = out[0][:len(train_pops)], out[0][len(train_pops):]

    d_ll = _proj_distance(*train_pops, f1_full, intercept=intercept)
    d_lg = _proj_distance(*test_pops, f1_full, intercept=intercept)
    
    d_nn = _reprojection_diffs((f1_full, f2_full), train_pops + test_pops)

    sigma = _est_noise(*(train_pops + test_pops), # proj_ax=f1_spec,
                       intercept=dec_intercept)
    sigma = np.squeeze(np.sqrt(sigma))
    d_ll = np.squeeze(d_ll)
    d_lg = np.squeeze(d_lg)
    d_nn = np.squeeze(d_nn)
    return d_ll, d_lg, d_nn, sigma

def _get_splitter_pops(splitters, pops, trl_ax=1, feat_ax=0):
    tr_pops, te_pops = [], []
    for i, spl_i in enumerate(splitters):
        pop_i = pops[i]
        tr_inds, te_inds = next(spl_i)
        tr_pops.append(pop_i[:, tr_inds])
        te_pops.append(pop_i[:, te_inds])
    return tr_pops, te_pops

def _mech_ccgp(a_pops, b_pops, norm=True, pre_pca=.99, trl_ax=1, feat_ax=0,
               accumulate_time=True):
    if accumulate_time:
        a_pops = list(np.squeeze(_accumulate_time(tp, ax=0))
                      for tp in a_pops)
        b_pops = list(np.squeeze(_accumulate_time(tp, ax=0))
                      for tp in b_pops)
    out = _preprocess(*(a_pops + b_pops), norm=norm, pre_pca=pre_pca,
                      trl_ax=trl_ax, sigma_est=None)
    out_pops, pipe, cent, sigma_est = out
    a_tr_i, b_tr_i = out_pops[:2], out_pops[2:]
    a_te_i, b_te_i = out_pops[:2], out_pops[2:]
    
    f1, d_ll_n, inter = _get_pair_ax(*a_tr_i)
    f1_g, d_lg_n, _ = _get_pair_ax(*b_tr_i)
        
    f1_common, _, _, i_common = _get_shared_ax((a_tr_i[0], b_tr_i[0]),
                                               (a_tr_i[1], b_tr_i[1]))

        
    dists = np.array(_proj_pops(f1, *b_tr_i, intercept=inter))    
    sigma = np.sqrt(_est_noise(*(a_tr_i + b_tr_i), proj_ax=f1,
                               intercept=inter))
    
    err1 = sts.norm(0, 1).cdf(-(dists[0] - d_ll_n/2)/(sigma))
    err2 = sts.norm(0, 1).cdf((dists[1] - d_ll_n/2)/(sigma))
    gen_err = (err1 + err2)/2 
    return gen_err

def very_direct_ccgp(a_pops, b_pops, accumulate_time=True, norm=True, pre_pca=.99,
                     n_folds=2, trl_ax=1, feat_ax=0,
                     splitter=skms.ShuffleSplit, test_prop=.1):
    """ R x D x """
    if accumulate_time:
        a_pops = list(np.squeeze(_accumulate_time(tp, ax=0))
                      for tp in a_pops)
        b_pops = list(np.squeeze(_accumulate_time(tp, ax=0))
                      for tp in b_pops)
    a_pops_splitters = list(
        splitter(n_folds, test_size=test_prop).split(a_pop_i.T)
        for i, a_pop_i in enumerate(a_pops))
    b_pops_splitters = list(
        splitter(n_folds, test_size=test_prop).split(b_pop_i.T)
        for i, b_pop_i in enumerate(b_pops))
    d_ll = np.zeros(n_folds)
    d_lg = np.zeros_like(d_ll)
    d_nn = np.zeros_like(d_ll)
    sigma = np.zeros_like(d_ll)
    gen_err = np.zeros_like(d_ll)
    for i in range(n_folds):
        a_tr_i, a_te_i = _get_splitter_pops(a_pops_splitters, a_pops)
        b_tr_i, b_te_i = _get_splitter_pops(b_pops_splitters, b_pops)

        out = _preprocess(*(a_tr_i + b_tr_i), norm=norm, pre_pca=pre_pca,
                          trl_ax=trl_ax, sigma_est=None)
        out_pops, pipe, cent, sigma_est = out
        # print(a_tr_i[0].shape, a_tr_i[1].shape,
        #       b_tr_i[0].shape, b_tr_i[1].shape)
        a_tr_i, b_tr_i = out_pops[:2], out_pops[2:]
        # print(a_tr_i[0].shape, a_tr_i[1].shape,
        #       b_tr_i[0].shape, b_tr_i[1].shape)

        out = _preprocess(*(a_te_i + b_te_i), norm=norm, pre_pca=pre_pca,
                          trl_ax=trl_ax, pipe=pipe, cent=cent,
                          fit=False, sigma_est=sigma_est)
        out_pops, pipe, cent = out
        a_te_i, b_te_i = out_pops[:2], out_pops[2:]

        f1, d_ll_n, inter = _get_pair_ax(*a_tr_i)
        f1_g, d_lg_n, _ = _get_pair_ax(*b_tr_i)

        f1_te, d_ll_te_n, inter_te = _get_pair_ax(*a_te_i)
        f1_g_te, d_lg_te_n, _ = _get_pair_ax(*b_te_i)
        
        f1_common, _, _, i_common = _get_shared_ax((a_tr_i[0], b_tr_i[0]),
                                                   (a_tr_i[1], b_tr_i[1]))

        p_ll = np.dot(f1.T, d_lg_n*f1_g)
        d_prod = p_ll*d_ll_n
        delt = np.sqrt(d_ll_n**2 - d_lg_n**2)
        
        d_lg_1 = .5*(np.sqrt(4*d_prod + delt**2) - delt)
        d_lg_2 = .5*(-np.sqrt(4*d_prod + delt**2) - delt)
        d_lg = max(d_lg_1, d_lg_2)
        d_ll = np.sqrt(d_lg**2 + delt**2)
        print(d_ll, d_lg, np.sqrt(d_prod))
        # print('d_prod', d_prod)
        # print('d_lg_n', d_ll_n)
        d_nn = np.sqrt(d_ll_n**2 - d_prod)
        
        dists = np.array(_proj_pops(f1, *b_tr_i, intercept=inter))
        dists_wi = np.array(_proj_pops(f1, *a_te_i, intercept=inter))
        
        sigma = np.sqrt(_est_noise(*(a_tr_i + b_tr_i), proj_ax=f1,
                                   intercept=inter))
        print('ccgp', 1 - _compute_ccgp(d_ll, d_lg, d_nn, sigma))
        print('ccgp', 1 - _compute_ccgp(np.sqrt(d_prod), np.sqrt(d_prod), d_nn, sigma))
        # print(d_ll_n, d_lg_n)
        # print(d_ll, d_lg, d_nn, sigma)
        
        # print(dists - d_ll_n/2)
        # print(dists, sigma)                          

        err1 = sts.norm(0, 1).cdf(-(dists[0] - d_ll_n/2)/(sigma))
        err2 = sts.norm(0, 1).cdf((dists[1] - d_ll_n/2)/(sigma))
        # print(1 - err1, 1 - err2, 1 - (err1 + err2)/2)
        gen_err[i] = (err1 + err2)/2 # sts.norm(0, 1).cdf(dists/2)
        print('err', 1 - gen_err[i])
    return gen_err
        

def direct_ccgp_bind_est(a_pops, b_pops, accumulate_time=True, n_folds=10,
                         norm=True, pre_pca=.99, trl_ax=1, feat_ax=0,
                         splitter=skms.ShuffleSplit, test_prop=.1):
    """ R x D x """
    if accumulate_time:
        a_pops = list(np.squeeze(_accumulate_time(tp, ax=0))
                      for tp in a_pops)
        b_pops = list(np.squeeze(_accumulate_time(tp, ax=0))
                      for tp in b_pops)
    a_pops_splitters = list(
        splitter(n_folds, test_size=test_prop).split(a_pop_i.T)
        for i, a_pop_i in enumerate(a_pops))
    b_pops_splitters = list(
        splitter(n_folds, test_size=test_prop).split(b_pop_i.T)
        for i, b_pop_i in enumerate(b_pops))
    d_ll = np.zeros(n_folds)
    d_lg = np.zeros_like(d_ll)
    d_nn = np.zeros_like(d_ll)
    sigma = np.zeros_like(d_ll)
    for i in range(n_folds):
        a_tr_i, a_te_i = _get_splitter_pops(a_pops_splitters, a_pops)
        b_tr_i, b_te_i = _get_splitter_pops(b_pops_splitters, b_pops)
        out = _compute_cb_quantities(a_tr_i, b_tr_i, norm=norm, pre_pca=pre_pca,
                                     trl_ax=trl_ax, feat_ax=feat_ax)
        pipe, f1_full, f2_full, f1_spec  = out[:4]
        dec_intercept, intercept, cent, dists, sigma_est = out[4:]

        out = _compute_cv_quantities(a_te_i, b_te_i, pipe, f1_full, f2_full,
                                     f1_spec, dists, dec_intercept=dec_intercept,
                                     intercept=intercept, cent=cent,
                                     sigma_est=sigma_est)
        d_ll[i], d_lg[i], d_nn[i], sigma[i] = out
        out = _compute_cv_quantities(a_tr_i, b_tr_i, pipe, f1_full, f2_full,
                                     f1_spec, dists, dec_intercept=dec_intercept,
                                     intercept=intercept, cent=cent,
                                     sigma_est=sigma_est)
        # d_ll[i], d_lg[i], d_nn[i], _ = out
        d_ll[i], d_lg[i], d_nn[i], sigma[i] = out
        # sigma[i] = 1
    
    return d_ll, d_lg, d_nn, sigma
