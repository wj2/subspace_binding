
import itertools as it
import functools as ft
import numpy as np
import sklearn.preprocessing as skp
import sklearn.linear_model as sklm
import sklearn.svm as skc
import sklearn.model_selection as skms
import sklearn.metrics as skm
import scipy.stats as sts

import general.utility as u
import general.neural_analysis as na
import multiple_representations.theory as mrt

def contrast_regression(data, tbeg, tend, binsize=None, binstep=None,
                        include_field='include',
                        contrast_left='contrastLeft',
                        contrast_right='contrastRight', regions=None,
                        tz_field='stimOn', model=sklm.Ridge, norm=True,
                        pca=None, t_ind=0, cache=True):
    if binsize is None:
        binsize = tend - tbeg
    pre_pipe_steps = []
    if norm:
        pre_pipe_steps.append(skp.StandardScaler())
    if pca is not None:
        pre_pipe_steps.append(skd.PCA(pca))
    pre_pipe = sklpipe.make_pipeline(*pre_pipe_steps)

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

        pop_pre_i = pre_pipe.fit_transform(pop_i[..., t_ind])
        score_l = skms.cross_validate(model(), pop_pre_i, c_l_i,
                                      return_estimator=True)
        score_r = skms.cross_validate(model(), pop_pre_i, c_r_i,
                                      return_estimator=True)
        coefs_l = np.array(list(e.coef_ for e in score_l['estimator']))
        coefs_r = np.array(list(e.coef_ for e in score_r['estimator']))
        wi_angs_l = u.pairwise_cosine_similarity(coefs_l)
        wi_angs_r = u.pairwise_cosine_similarity(coefs_r)
        ac_angs_lr = u.pairwise_cosine_similarity(coefs_l, coefs_r)

        scores_left.append(score_l)
        scores_right.append(score_r)
        wi_angs.append((wi_angs_l, wi_angs_r))
        ac_angs.append(ac_angs_lr)
    return scores_left, scores_right, wi_angs, ac_angs

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
    low_thr = np.percentile(np.concatenate(ref_data, axis=0), 
                            50 - dead_perc/2)
    high_thr = np.percentile(np.concatenate(ref_data, axis=0), 
                             50 + dead_perc/2)
    high_mask = field_data > high_thr
    low_mask = field_data < low_thr
    out = (high_mask, low_mask)
    if additional_data is not None:
        add_high = additional_data > high_thr
        add_low = additional_data < low_thr
        out = out + (add_high, add_low)
    return out

def full_lm_organization(data, tbeg, tend, dead_perc=30, winsize=500, tstep=20,
                         turn_feature=('ev_left', 'ev_right'),
                         tzf_1='Offer 1 on', tzf_2='Offer 2 on',
                         pre_pca=.95):
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
                              time_zero_field=tzf_1)
    pops2, xs = data.get_psth(binsize=winsize, begin=tbeg, end=tend,
                              time_zero_field=tzf_2)
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
    return cond_pop_pair, xs

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

def xor_analysis(data, tbeg, tend, dec1_field, dec2_field,
                 dead_perc=30, winsize=500, tstep=20,
                 pop_resamples=20, kernel='linear',
                 dec_tzf='offer_left_on',
                 gen_tzf='offer_right_on',
                 min_trials=160, pre_pca=.95,
                 shuffle_trials=True, c1_targ=2,
                 c2_targ=3, dec1_mask=None, dec2_mask=None,
                 **kwargs):
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

def binding_analysis(data, tbeg, tend, feat1, feat2,
                     dead_perc=30, winsize=500, tstep=20,
                     pop_resamples=20, kernel='linear',
                     f1_tzf='offer_left_on', f2_tzf='offer_right_on',
                     min_trials=160, pre_pca=None,
                     shuffle_trials=True, c1_targ=2, c2_targ=3,
                     f1_mask=None, f2_mask=None, use_split_dec='prob_chosen',
                     regions=None, shuffle=False, mean=False,
                     n_folds=20, params=None, **kwargs):
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
    for i, pop_f11_i in enumerate(pop_f11):
        c1 = pop_f11_i + pop_f22[i]
        c2 = pop_f12[i] + pop_f21[i]
        print(pop_f11_i.shape, c1.shape)
        out = na.fold_skl(c1, c2, n_folds, params=params, 
                          mean=mean, pre_pca=pre_pca, shuffle=shuffle,
                          impute_missing=False,
                          **kwargs)
        outs[i] = out[0]
    return outs, xs
    

def generalization_analysis(data, tbeg, tend, dec_field, gen_field,
                            dead_perc=30, winsize=500, tstep=20,
                            pop_resamples=20, kernel='linear',
                            f1_tzf='offer_left_on',
                            f2_tzf='offer_right_on',
                            min_trials=160, pre_pca=None,
                            shuffle_trials=True, c1_targ=2,
                            c2_targ=3, f1_mask=None, f2_mask=None,
                            use_split_dec='prob_chosen',
                            **kwargs):
    out = _compute_masks(data, dec_field, gen_field, dead_perc=dead_perc,
                         use_split_dec=use_split_dec, dec_mask=f1_mask,
                         gen_mask=f2_mask, c1_targ=c1_targ, c2_targ=c2_targ)
    mask_c1, mask_c2, gen_mask_c1, gen_mask_c2 = out
    
    out = data.decode_masks(mask_c1, mask_c2, winsize, tbeg, tend, tstep, 
                            pseudo=True, time_zero_field=f1_tzf,
                            min_trials_pseudo=min_trials,
                            resample_pseudo=pop_resamples, ret_pops=True, 
                            shuffle_trials=shuffle_trials, pre_pca=pre_pca,
                            decode_tzf=f2_tzf, decode_m1=gen_mask_c1, 
                            decode_m2=gen_mask_c2, **kwargs)
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
                  save_dict=True, str_key=True):
    new_dict = {}
    for (dec, gen), val in out_dict.items():
        if save_dict:
            sub_item = {}
            for i, keep_ind in enumerate(keep_inds):
                sub_item[keep_labels[i]] = val[keep_ind]
        else:
            sub_item = []
            for i, keep_ind in enumerate(keep_inds):
                sub_item.append(val[keep_ind])
        if str_key:
            nk = '-'.join((dec, gen))
        else:
            nk = (dec, gen)
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

default_suffixes = (('_chosen', '_unchosen'), ('_left', '_right'),
                    (' offer 1', ' offer 2'))
default_timing = (('offer_chosen_on', 'offer_unchosen_on'),
                  ('offer_left_on', 'offer_right_on'),
                  ('Offer 1 on', 'Offer 2 on'))
def _compute_all_funcs(data, tbeg, tend, dec_var, func,
                       suffixes=default_suffixes,
                       timing=default_timing, mask_func=None,
                       compute_reverse=True, **kwargs):
    out_dict = {}
    for i, (dec_suff, gen_suff) in enumerate(suffixes):
        dec_field = dec_var + dec_suff
        gen_field = dec_var + gen_suff
        if mask_func is not None:
            dec_mask = mask_func(data[dec_field])
            gen_mask = mask_func(data[gen_field])
        else:
            dec_mask = None
            gen_mask = None
        dec_tzf, gen_tzf = timing[i]
        print(dec_field, dec_tzf)
        print(gen_field, gen_tzf)
        out = func(data, tbeg, tend, dec_field, gen_field,
                   f1_mask=dec_mask, f2_mask=gen_mask,
                   f1_tzf=dec_tzf, f2_tzf=gen_tzf, **kwargs)
        out_dict[(dec_field, gen_field)] = out
        if compute_reverse:
            out = func(data, tbeg, tend, gen_field, dec_field,
                       f1_mask=gen_mask, f2_mask=dec_mask,
                       f1_tzf=gen_tzf, f2_tzf=dec_tzf, **kwargs)
            out_dict[(gen_field, dec_field)] = out
    return out_dict  

def compute_all_binding(*args, **kwargs):
    return _compute_all_funcs(*args, binding_analysis,
                              **kwargs)

def compute_all_generalizations(*args, **kwargs):
    return _compute_all_funcs(*args, generalization_analysis,
                              **kwargs)

def compute_all_xor(*args, **kwargs):
    return _compute_all_fucns(*args, xor_analysis,
                              **kwargs)
                                

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
        print(lb, ub)
        for j, fd in enumerate(field_datas):
            mask_ij = (fd >= lb).rs_and(fd < ub)
            masks[j].append(mask_ij)
    return masks

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
                                      None, time_zero_field=dec_tzf,
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

