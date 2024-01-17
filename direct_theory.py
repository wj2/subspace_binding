
import numpy as np
import sklearn.model_selection as skms
import scipy.stats as sts
import itertools as it

import general.utility as u
import general.neural_analysis as na
import composite_tangling.code_analysis as ca
import scipy.optimize as sopt
import rsatoolbox as rsa

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

def _compute_bind(d_n, sigma, k=2, n=2, n_stim=2):
    n_swaps = ca.compute_feature_swaps_theory(k, n, n_stim)
    return n_swaps*sts.norm(0, 1).cdf(-np.sqrt(2)*d_n/(2*sigma))


def rsa_preproc(a_pops, b_pops, norm=True, pre_pca=.99, trl_ax=1, feat_ax=0,
                accumulate_time=True, ret_out_pops=False):
    if accumulate_time:
        a_pops = list(np.squeeze(_accumulate_time(tp, ax=feat_ax))
                      for tp in a_pops)
        b_pops = list(np.squeeze(_accumulate_time(tp, ax=feat_ax))
                      for tp in b_pops)

    out = _preprocess(*(a_pops + b_pops), norm=norm, pre_pca=pre_pca,
                      trl_ax=trl_ax, sigma_est=None,
                      sigma_transform=False)
    out_pops, pipe, cent, sigma_est = out
    if not ret_out_pops:
        out = out_pops[:2], out_pops[2:]
    else:
        out = out_pops
    return out


def _rsa_theory(a_pops, b_pops, **kwargs):
    a_pops_u, b_pops_u = rsa_preproc(a_pops, b_pops, **kwargs)
    out = estimate_lin_nonlin_distance(a_pops_u, b_pops_u)
    d_lins, d_nls, sigmas, sems, n_neurs = out

    sem_nl = np.sqrt(2*d_nls**2 + sems**2)
    print(d_lins.shape, d_nls.shape)
    # THIS SEEMS WRONG, probably should be [:, 0]
    pred_ccgp = _compute_ccgp(d_lins[0, 0], d_lins[0, 0], sem_nl,
                              sigmas)
    pred_bind = _compute_bind(np.sqrt(2)*d_nls, sigmas)
    return pred_ccgp, pred_bind, (d_lins[:, 0], d_nls, sigmas, sems, n_neurs)


def combined_ccgp_bind_est(train_pops, test_pops, use_rsa=True, schema=None, **kwargs):
    n_pops = len(train_pops[0])
    assert(use_rsa)
    if schema is None:
        schema = np.array([[0, 0],
                           [1, 0],
                           [0, 1],
                           [1, 1]])

    # make rdms
    rdms_all = np.zeros((n_pops, len(schema), len(schema)))
    sems = np.zeros(n_pops)
    sigmas = np.zeros_like(sems)
    sep_rdms = []
    for i in range(n_pops):
        tr_ps = list(tp[i] for tp in train_pops)
        te_ps = list(tp[i] for tp in test_pops)
        a_pops, b_pops = rsa_preproc(tr_ps, te_ps, **kwargs)
        rdm_mat, sigmas_i, sems_i, rdms = estimate_distances(a_pops, b_pops)
        sep_rdms.append(rdms)
        rdms_all[i] = rdm_mat
        sigmas[i] = sigmas_i
        sems[i] = sems_i
        
    d_lins = np.zeros((n_pops, schema.shape[1]))
    d_nls = np.zeros((n_pops, 1))
    n_neurs = rdms[0].descriptors['noise'].shape[0]
    print(np.mean(rdms_all*n_neurs, axis=0))
    print(rdms_all[0])
    for i in range(n_pops):
        n_neurs = sep_rdms[i][0].descriptors['noise'].shape[0]
        d_lins[i], d_nls[i], _ = decompose_all_distance_matrices(
            np.mean(rdms_all[i:i+1], axis=0, keepdims=True), schema, n_neurs
        )

    sem_nl = np.sqrt(2*d_nls**2 + sems**2)
    pred_ccgp = _compute_ccgp(d_lins[0, 0], d_lins[0, 0], sem_nl,
                              sigmas)
    pred_bind = _compute_bind(np.sqrt(2)*d_nls, sigmas)
    return pred_ccgp, pred_bind, (d_lins[:, 0], d_nls, sigmas, sems)


def direct_ccgp_bind_est_pops(
        *args, time_accumulate=True, **kwargs,
):
    if time_accumulate:
        out = direct_ccgp_bind_est_pops_cv(*args, **kwargs)
    else:
        out = direct_ccgp_bind_est_pops_tc(*args, **kwargs)
    return out


def direct_ccgp_bind_est_pops_tc(
        train_pops, test_pops, n_folds=5, test_prop=.1, **kwargs
):
    n_pops = len(train_pops[0])
    n_tc = train_pops[0][0].shape[-1]
    bind_ests = np.zeros((n_pops, n_tc))
    gen_ests = np.zeros_like(bind_ests)
    d_l, d_n, sigma, sem, n_neurs = list(np.zeros_like(bind_ests)
                                         for i in range(5))
    for i in range(n_pops):
        for j in range(n_tc):
            tr_ps = list(tp[i, ..., j] for tp in train_pops)
            te_ps = list(tp[i, ..., j] for tp in test_pops)
            out = _rsa_theory(tr_ps, te_ps, **kwargs)
            gen_ests[i, j], bind_ests[i, j] = out[0]
            d_l[i, j], d_n[i, j], sigma[i, j], sem[i, j], n_neurs[i, j] = out[1]
    return bind_ests, gen_ests, (d_l, d_n, sigma, sem, n_neurs)    


def direct_ccgp_bind_est_pops_cv(train_pops, test_pops, n_folds=5, test_prop=.1,
                              **kwargs):
    n_pops = len(train_pops[0])
    bind_ests = np.zeros((n_pops, n_folds))
    gen_ests = np.zeros_like(bind_ests)
    d_l, d_n, sigma, sem, n_neurs = list(np.zeros_like(bind_ests)
                                         for i in range(5))
    for i in range(n_pops):
        tr_ps = list(tp[i] for tp in train_pops)
        te_ps = list(tp[i] for tp in test_pops)
        out = _rsa_theory(tr_ps, te_ps, **kwargs)
        gen_ests[i], bind_ests[i], (d_l[i], d_n[i], sigma[i], sem[i], n_neurs[i]) = out
    return bind_ests, gen_ests, (d_l, d_n, sigma, sem, n_neurs)


def _preprocess(*pops, fit=True, norm=True, pre_pca=.99, pipe=None,
                sigma_est=None, trl_ax=1, cent=None, post_norm=False,
                sigma_transform=True):
    if fit and pipe is None and (norm or pre_pca is not None):
        pipe = na.make_model_pipeline(norm=norm, pca=pre_pca,
                                      post_norm=post_norm)
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
    if sigma_transform:
        out_pops = list((p_i - cent)/sigma_est_comp for p_i in preproc_pops)
    else:
        out_pops = preproc_pops
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


def estimate_distances(p1_group, p2_group, max_trials=None,
                       n_resamps=50, **kwargs):
    rng = np.random.default_rng()
    t_nums = list(p_i.shape[-1] for p_i in p1_group + p2_group)
    if max_trials is None:
        max_trials = np.min(t_nums)

    n_stim = len(p1_group) + len(p2_group)
    rdm_mat = np.zeros((n_resamps, n_stim, n_stim))
    sigmas = np.zeros(n_resamps)
    sems = np.zeros(n_resamps)
    p12_group = p1_group + p2_group
    rdm_list = []

    for j in range(n_resamps):
        inds = list(
            rng.choice(tn, max_trials, replace=False) for tn in t_nums
        )
        p_all_sep = list(
            p_c[..., inds[i]] for i, p_c in enumerate(p12_group)
        )

        p_all = np.concatenate(p_all_sep, axis=1).T
        p_mask = np.var(p_all, axis=0) > 0
        p_all = p_all[:, p_mask]

        stim = list(
            (str(i),)*max_trials for i, p_ind in enumerate(p12_group)
        )
        stim = np.concatenate(stim)
        data = rsa.data.Dataset(p_all, obs_descriptors={'stimulus': stim})

        rdm = rsa.rdm.calc_rdm(data, descriptor='stimulus', noise=None,
                               method='crossnobis')
        f1, d_ll_n, inter = _get_pair_ax(*p_all_sep[:2])
        sigmas[j] = np.sqrt(_est_noise(
            *p_all_sep, proj_ax=f1, intercept=inter,
        ))
        sems[j] = _est_sem(*p_all_sep, sub_ax=f1)

        rdm_mat[j] = rdm.get_matrices()[0]
        rdm_list.append(rdm)
    return rdm_mat, sigmas, sems, rdm_list


def make_simple_distance_matrix(d_lins2, d_nl2, schema, shape):
    new_mat = np.zeros(shape)
    ind_pairs = it.combinations(range(shape[0]), 2)
    for (i, j) in ind_pairs:
        d_ij = (np.sum(((schema[i] - schema[j])**2)*d_lins2)
                + 2*d_nl2*(i != j))
        new_mat[i, j] = d_ij
        new_mat[j, i] = d_ij
    return new_mat


def make_distance_matrix(d_lins2, d_nl2, schema, shape):
    new_mat = np.zeros(shape)
    ind_pairs = it.combinations(range(shape[0]), 2)
    for (i, j) in ind_pairs:
        d_ij = (np.sum(((schema[i] - schema[j])**2)*d_lins2)
                + (d_nl2[i] + d_nl2[j])*(i != j))
        new_mat[i, j] = d_ij
        new_mat[j, i] = d_ij
    return new_mat


def decompose_all_distance_matrices(mat, schema, n_neurs):
    mat = mat*n_neurs
    mult = .1 # np.max(mat) 
    dists_init = np.abs(mult*sts.norm(0, 1).rvs(schema.shape[1]
                                                + schema.shape[0]))
    dists_init = np.abs(mult*sts.norm(0, 1).rvs(schema.shape[1] + 1))

    def _mat_recon_func(dists):
        d_lins = dists[:schema.shape[1]]
        d_nl = dists[schema.shape[1]:]
        new_mat = make_simple_distance_matrix(d_lins, d_nl, schema, mat.shape[1:])
        # new_mat = make_distance_matrix(d_lins, d_nl, schema, mat.shape[1:])
        diff = np.sum(np.mean((np.expand_dims(new_mat, 0) - mat)**2, axis=0))
        return diff

    res = sopt.minimize(_mat_recon_func, dists_init,
                        bounds=((0, None),)*len(dists_init))
    x = np.sqrt(res.x)
    d_lins = x[:schema.shape[1]]
    d_nl = x[schema.shape[1]:]

    d_nl = np.median(d_nl)
    return d_lins, d_nl, res


def decompose_distance_matrix_nnls(mat, schema, n_neurs, indiv_dists=False):
    mat = mat*n_neurs
    mat[mat < 0] = 0

    design = []
    targ = []
    for i, j in it.combinations(range(len(schema)), 2):
        if indiv_dists:
            add_d = np.zeros(len(schema))
            add_d[i] = 1
            add_d[j] = 1
        else:
            add_d = (2,)
        design_ij = np.concatenate((np.abs(schema[i] - schema[j]), add_d))
        design.append(design_ij)
        targ.append(mat[i, j])
    out = sopt.nnls(np.array(design), np.array(targ))
    # out = np.linalg.lstsq(np.array(design), np.array(targ), rcond=None)
    x = out[0]

    d_lins = x[:schema.shape[1]]
    d_nl = x[schema.shape[1]:]

    d_nl = np.mean(d_nl)
    return d_lins, d_nl, None


def decompose_distance_matrix(mat, schema, n_neurs):
    mult = 10  # np.max(mat)
    dists_init = np.abs(mult*sts.norm(0, 1).rvs(schema.shape[1]
                                                + schema.shape[0]))
    dists_init = np.abs(mult*sts.norm(0, 1).rvs(schema.shape[1] + 1))
    mat = mat*n_neurs

    def _mat_recon_func(dists):
        d_lins = dists[:schema.shape[1]]
        d_nl = dists[schema.shape[1]:]
        new_mat = make_simple_distance_matrix(d_lins, d_nl, schema, mat.shape)
        # new_mat = make_distance_matrix(d_lins, d_nl, schema, mat.shape)
        diff = np.sum((new_mat - mat)**2)
        return diff

    res = sopt.minimize(_mat_recon_func, dists_init,
                        bounds=((0, None),)*len(dists_init))
    x = res.x

    x = np.sqrt(x)
    d_lins = x[:schema.shape[1]]
    d_nl = x[schema.shape[1]:]

    d_nl = np.mean(d_nl)
    return d_lins, d_nl, res


def estimate_lin_nonlin_distance(p1_group, p2_group, schema=None,
                                 rdm_mat=None, indiv_dists=False,
                                 **kwargs):
    if schema is None:
        schema = np.array([[0, 0],
                           [1, 0],
                           [0, 1],
                           [1, 1]])
    if rdm_mat is None:
        rdm_mat, sigmas, sems, rdms = estimate_distances(p1_group, p2_group,
                                                         **kwargs)
    d_lins = np.zeros((rdm_mat.shape[0], schema.shape[1]))
    d_nls = np.zeros((rdm_mat.shape[0], 1))
    for i, rdm_i in enumerate(rdm_mat):
        n_neurs = rdms[i].descriptors['noise'].shape[0]
        d_lins[i], d_nls[i], _ = decompose_distance_matrix_nnls(
            rdm_i, schema, n_neurs, indiv_dists=indiv_dists,
        )
    d_lins = np.median(d_lins, axis=0, keepdims=True)
    d_nls = np.median(d_nls, axis=0, keepdims=True)
    sigmas = np.mean(sigmas, keepdims=True)
    sems = np.mean(sems, keepdims=True)
    d_lins[d_lins < 0] = 0
    d_nls[d_nls < 0] = 0
    d_lins = np.sqrt(d_lins)
    d_nls = np.sqrt(d_nls)
    # print(d_lins, d_nls, sigmas, sems)

    return d_lins, d_nls, sigmas, sems, n_neurs


def _est_sem(*pops, sub_ax=None, mean=True):
    ests = np.zeros(len(pops))
    for i, pop in enumerate(pops):
        n_dims, n_trls = pop.shape
        v = np.var(pop, axis=1)
        l_sem = np.sqrt(np.sum(v/n_trls))
        
        if sub_ax is not None:
            sub_sem = np.var(np.sum(sub_ax*pop, axis=0))/n_trls
            l_sem = np.sqrt(l_sem**2 - sub_sem)
        ests[i] = l_sem
    if mean:
        full_est = np.sqrt(2)*np.mean(ests)
    else:
        full_est = np.sqrt(np.sum(ests**2))
    return full_est


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
