
import os
import re
import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.stats as sts
import functools as ft
import pickle
import sklearn.linear_model as sklm
import sklearn.model_selection as skms

import general.plotting as gpl
import general.plotting_styles as gps
import general.paper_utilities as pu
import general.data_io as gio
import general.utility as u
import general.neural_analysis as na
import multiple_representations.analysis as mra
import multiple_representations.auxiliary as mraux
import multiple_representations.theory as mrt
import multiple_representations.visualization as mrv
import multiple_representations.direct_theory as mrdt

config_path = 'multiple_representations/figures.conf'

colors = np.array([(127,205,187),
                   (65,182,196),
                   (29,145,192),
                   (34,94,168),
                   (37,52,148),
                   (8,29,88)])/256

class MultipleRepFigure(pu.Figure):

    def _make_color_dict(self, ks):
        color_dict = {}
        for k in ks:
            color_dict[k] = self.params.getcolor(k + '_color')
        return color_dict
            
    def save_pickles(self, folder, suff='', ext='.mat'):
        unsaved_keys = []
        for key in self.panel_keys:
            panel_data = self.data.get(key)
            if panel_data is not None:
                filename = key.split('_', 1)[1] + suff + ext
                path = os.path.join(folder, filename)
                sio.savemat(path, panel_data)
            else:
                unsaved_keys.append(key)
        if len(unsaved_keys) > 0:
            print('the following keys did not exist and were not '
                  'saved:\n{}'.format(unsaved_keys))
    
def _accumulate_time(pop, keepdim=True, ax=1):
    out = np.concatenate(list(pop[..., i] for i in range(pop.shape[-1])),
                         axis=ax)
    if keepdim:
        out = np.expand_dims(out, -1)
    return out

def _subsample_pops(*pops, samp_pops=10):
    n_pops = pops[0].shape[0]
    inds = np.random.default_rng().choice(n_pops, size=samp_pops, replace=False)
    new_pops = list(pop_i[inds] for pop_i in pops)
    return new_pops

class DecodingFigure(MultipleRepFigure):

    def __init__(self, fig_key='decoding_figure', colors=colors, **kwargs):
        fsize = (8, 3)
        cf = u.ConfigParserColor()
        cf.read(config_path)
        
        params = cf[fig_key]
        self.fig_key = fig_key
        super().__init__(fsize, params, colors=colors, **kwargs)

    def make_gss(self):
        gss = {}        
        self.gss = gss

    def get_experimental_data(self, force_reload=False):
        if self.data.get('experimental_data') is None or force_reload:
            data_folder = self.params.get('data_folder')
            data = gio.Dataset.from_readfunc(mraux.load_fine_data, data_folder)
            
            exclude_list = self.params.getlist('exclude_datasets')
            session_mask = np.logical_not(np.isin(data['date'], exclude_list))
            self.data['experimental_data'] = data.session_mask(session_mask)
        return self.data['experimental_data']

    def _decoding_analysis(self, var, func, force_reload=True, **kwargs):
        tbeg = self.params.getint('tbeg')
        tend = self.params.getint('tend')
        winsize = self.params.getint('winsize')
        winstep = self.params.getint('winstep')
        prs = self.params.getint('resamples')
        pca_pre = self.params.getfloat('pca_pre')
        time_acc = self.params.getboolean('time_accumulate')
        regions = self.params.getlist('use_regions')

        exper_data = self.get_experimental_data(force_reload=force_reload)

        decoding_results = {}
        for region in regions:
            if region == 'all':
                use_regions = None 
            else:
                use_regions = (region,)
            out_r = func(exper_data, tbeg, tend, var,
                         winsize=winsize, pre_pca=pca_pre,
                         pop_resamples=prs, tstep=winstep,
                         time_accumulate=time_acc,
                         regions=use_regions, **kwargs)
            decoding_results[region] = out_r            
        return decoding_results

    def _fit_linear_models(self, dec_dict, force_reload=False):
        pca_pre = self.params.getfloat('pca_pre')
        l1_ratio = self.params.getfloat('l1_ratio')
        test_prop = self.params.getfloat('test_prop')
        multi_task = self.params.getboolean('multi_task')
        folds_n = self.params.getint('n_folds')
        samp_pops = self.params.getint('linear_fit_pops')
        internal_cv = self.params.getboolean('internal_cv')
        if internal_cv:
            model = sklm.ElasticNetCV
        else:
            model = sklm.ElasticNet
        conds = ((-1, 1), (1, 1), (-1, -1), (1, -1))
        lm_dict = {}
        done_keys = []
        for contrast, (_, _, p1, p2, p3, p4, _) in dec_dict.items():
            if set(contrast) not in done_keys and p1.shape[1] > 0:
                pops = list(_accumulate_time(p_i) for p_i in (p1, p2, p3, p4))
                pops = _subsample_pops(*pops, samp_pops=samp_pops)
                out = mra.fit_linear_models(pops, conds, folds_n=folds_n,
                                            model=model, multi_task=multi_task,
                                            l1_ratio=l1_ratio, pre_pca=pca_pre,
                                            test_prop=test_prop, max_iter=10000,
                                            internal_cv=internal_cv)
                lm, nlm, nv, r2 = out
                out_dict = {'lm':lm, 'nlm':nlm, 'nv':nv, 'r2':r2,}
                done_keys.append(set(contrast))
                lm_dict[contrast] = out_dict
        return lm_dict

    def _model_terms(self, key, dec_key, force_recompute=False):
        if (self.data.get(dec_key) is None and
            self.data.get(key) is None):
            self.panel_prob_generalization()
        dec = self.data[dec_key]
        use_regions = self.params.getlist('use_regions')
        if self.data.get(key) is None or force_recompute:
            model_dict = {}
            for region, dec_results in dec.items():
                if region in use_regions:
                    out = self._fit_linear_models(dec_results)
                    model_dict[region] = out
        self.data[key] = model_dict
        return self.data[key]

    def _model_predictions(self, term_key):
        terms = self.data[term_key]
        predictions = {}
        for region, term_contrasts in terms.items():
            predictions[region] = {}
            for contrast, term_dict in term_contrasts.items():
                lm = term_dict['lm']
                nlm = term_dict['nlm']
                nv = term_dict['nv']
                r2 = term_dict['r2']
                pred_out = mra.compute_ccgp_bin_prediction(lm, nlm, nv, ) # r2=r2)
                pred_ccgp, pred_bind = pred_out[:2]
                ret_dict = {'pred_ccgp':np.squeeze(pred_ccgp),
                            'pred_bind':np.squeeze(pred_bind)}
                predictions[region][contrast] = ret_dict
        return predictions

    def _direct_predictions(self, dec_key):
        decs = self.data.get(dec_key)
        use_regions = self.params.getlist('use_regions')
        model_dict = {}
        for region, dec_results in decs.items():
            if region in use_regions:
                model_dict[region] = {}
                for contrast, out in dec_results.items():
                    _, _, p1, p2, p3, p4, _ = out
                    if p1.shape[1] > 0:
                        out = mrdt.direct_ccgp_bind_est_pops(
                            (p1, p2), (p3, p4), test_prop=0,
                            empirical=False)
                        out_dict = {'pred_ccgp':1 - out[1],
                                    'pred_bin':1 - out[0],
                                    'd_l':out[2][0],
                                    'd_n':out[2][1],
                                    'sigma':out[2][2]}
                    else:
                        out_dict = None
                    model_dict[region][contrast] = out_dict
        return model_dict
                    
    def panel_rwd_direct_predictions(self, force_refit=False):
        key = 'rwd_direct_predictions'
        save_key = 'panel_rwd_direct_predictions'
        dec_key = 'rwd_generalization'
        if self.data.get(key) is None or force_refit:
            self.data[key] = self._direct_predictions(dec_key)
        if self.data.get(save_key) is None:
            self.make_dec_save_dicts(keys=(key,),
                                     loss=False,
                                     keep_subset=False)
        return self.data[key]
                                     
    def panel_prob_direct_predictions(self, force_refit=False):
        key = 'prob_direct_predictions'
        save_key = 'panel_prob_direct_predictions'
        dec_key = 'prob_generalization'
        if self.data.get(key) is None or force_refit:
            self.data[key] = self._direct_predictions(dec_key)
        if self.data.get(save_key) is None:
            self.make_dec_save_dicts(keys=(key,),
                                     loss=False,
                                     keep_subset=False)
        return self.data[key]

    def panel_ev_direct_predictions(self, force_refit=False):
        key = 'ev_direct_predictions'
        save_key = 'panel_ev_direct_predictions'
        dec_key = 'ev_generalization'
        if self.data.get(key) is None or force_refit:
            self.data[key] = self._direct_predictions(dec_key)
        if self.data.get(save_key) is None:
            self.make_dec_save_dicts(keys=(key,),
                                     loss=False,
                                     keep_subset=False)
        return self.data[key]

    def panel_ev_other_direct_predictions(self, force_refit=False):
        key = 'ev_other_direct_predictions'
        save_key = 'panel_ev_other_direct_predictions'
        dec_key = 'ev_generalization_other'
        if self.data.get(key) is None or force_refit:
            self.data[key] = self._direct_predictions(dec_key)
        if self.data.get(save_key) is None:
            self.make_dec_save_dicts(keys=(key,),
                                     loss=False,
                                     keep_subset=False)
        return self.data[key]

    def panel_prob_model_prediction(self, force_refit=False,
                                    force_recompute=False):
        key = 'panel_prob_model_prediction'
        term_key = 'prob_model_terms'
        dec_key = 'prob_generalization'
        if force_refit or (self.data.get(key) is None
                           and self.data.get(term_key) is None):
            self._model_terms(term_key, dec_key,
                              force_recompute=force_refit)
        if force_recompute or self.data.get(key) is None:
            self.data[key] = self._model_predictions(term_key)
        return self.data.get(key)

    def panel_rwd_model_prediction(self, force_refit=False,
                                   force_recompute=False):
        key = 'panel_rwd_model_prediction'
        term_key = 'rwd_model_terms'
        dec_key = 'rwd_generalization'
        if force_refit or (self.data.get(key) is None
                           and self.data.get(term_key) is None):
            self._model_terms(term_key, dec_key,
                              force_recompute=force_refit)
        if force_recompute or self.data.get(key) is None:
            self.data[key] = self._model_predictions(term_key)
        return self.data.get(key)

    def make_dec_save_dicts(self, keys=('prob_generalization',
                                        'rwd_generalization'),
                            loss=True, prefix='panel', **kwargs):
        for k in keys:
            reg_dict = {}
            if loss:
                reform_func = mra.reformat_generalization_loss
                new_key = prefix + '_loss_' + k
            else:
                reform_func = mra.reformat_dict
                new_key = prefix + '_' + k
            for region, data in self.data[k].items():
                reg_dict[region] = reform_func(self.data[k][region],
                                               **kwargs)
            self.data[new_key] = reg_dict

    def prob_generalization(self, force_reload=False,
                                  force_recompute=False):
        key = 'prob_generalization'
        dead_perc = self.params.getfloat('exclude_middle_percentiles')
        mask_func = lambda x: x < 1
        min_trials = self.params.getint('min_trials_prob')
        if self.data.get(key) is None or force_recompute:
            out = self._decoding_analysis('prob', mra.compute_all_generalizations,
                                          force_reload=force_reload,
                                          dead_perc=dead_perc, mask_func=mask_func,
                                          min_trials=min_trials)
            self.data[key] = out
        return self.data[key]

    def rwd_generalization(self, force_reload=False,
                                 force_recompute=False):
        key = 'rwd_generalization'
        dead_perc = None
        min_trials = self.params.getint('min_trials_rwd')
        if self.data.get(key) is None or force_recompute:
            out = self._decoding_analysis('rwd', mra.compute_all_generalizations,
                                          force_reload=force_reload,
                                          dead_perc=dead_perc,
                                          min_trials=min_trials)
            self.data[key] = out
        return self.data[key]

    def ev_generalization(self, force_reload=False,
                          force_recompute=False,
                          use_subj=True):
        data_field = 'ev'
        if use_subj:
            data_field = 'subj_' + data_field
        key = 'ev_generalization'
        dead_perc = self.params.getfloat('exclude_middle_percentiles')
        min_trials = self.params.getint('min_trials_ev')
        mask_func = lambda x: x < 1
        mask_var = 'prob'
        use_split_dec = None
        if self.data.get(key) is None or force_recompute:
            out = self._decoding_analysis(data_field, mra.compute_all_generalizations,
                                          force_reload=force_reload,
                                          dead_perc=dead_perc,
                                          min_trials=min_trials,
                                          use_split_dec=use_split_dec,
                                          mask_func=mask_func, mask_var=mask_var)
            self.data[key] = out
        return self.data[key]
    
    def ev_generalization_other(self, force_reload=False,
                                force_recompute=False,
                                use_subj=True):
        data_field = 'ev'
        if use_subj:
            data_field = 'subj_' + data_field
        key = 'ev_generalization_other'
        dead_perc = self.params.getfloat('exclude_middle_percentiles')
        min_trials = self.params.getint('min_trials_ev_other')
        use_split_dec = None
        if self.data.get(key) is None or force_recompute:
            out = self._decoding_analysis(data_field,
                                          mra.compute_conditional_generalization,
                                          force_reload=force_reload,
                                          dead_perc=dead_perc,
                                          min_trials=min_trials,
                                          use_split_dec=use_split_dec)
            self.data[key] = out
        return self.data[key]
    
    def _generic_dec_panel(self, key, func, *args, dec_key=None, loss=False,
                           **kwargs):
        func(*args, **kwargs)
        key_i = key.split('_', 1)[1]
        self.make_dec_save_dicts(keys=(key_i,), loss=loss)
        return self.data[key]

    def _plot_gen_results(self, dec_key, theory_key):
        dec_gen = self.data.get(dec_key)
        dec_gen_theory = self.data.get(theory_key)
        if dec_gen is not None:
            for region, dec_res in dec_gen.items():
                if dec_gen_theory is not None:
                    pred_dict = dec_gen_theory.get(region)
                else:
                    pred_dict = None
                axs = mrv.plot_all_gen(dec_res, prediction=pred_dict)
                axs[0].set_title(region)
        
    def plot_all_prob_generalization(self):
        dec_gen_key = 'prob_generalization'
        dec_gen_theory_key = 'prob_direct_predictions'
        return self._plot_gen_results(dec_gen_key, dec_gen_theory_key)

    def plot_all_rwd_generalization(self):
        dec_gen_key = 'rwd_generalization'
        dec_gen_theory_key = 'rwd_direct_predictions'
        return self._plot_gen_results(dec_gen_key, dec_gen_theory_key)

    def plot_all_ev_generalization(self):
        dec_gen_key = 'ev_generalization'
        dec_gen_theory_key = 'ev_direct_predictions'
        return self._plot_gen_results(dec_gen_key, dec_gen_theory_key)

    def plot_all_ev_generalization_other(self):
        dec_gen_key = 'ev_generalization_other'
        dec_gen_theory_key = 'ev_other_direct_predictions'
        return self._plot_gen_results(dec_gen_key, dec_gen_theory_key)

    def panel_prob_generalization(self, *args, **kwargs):
        key = 'panel_prob_generalization'
        return self._generic_dec_panel(key, self.prob_generalization, *args,
                                       **kwargs)

    def panel_rwd_generalization(self, *args, **kwargs):
        key = 'panel_rwd_generalization'
        return self._generic_dec_panel(key, self.rwd_generalization, *args,
                                       **kwargs)

    def panel_ev_generalization(self, *args, **kwargs):
        key = 'panel_ev_generalization'
        return self._generic_dec_panel(key, self.ev_generalization, *args,
                                       **kwargs)

    def panel_ev_generalization_other(self, *args, **kwargs):
        key = 'panel_ev_generalization_other'
        return self._generic_dec_panel(key, self.ev_generalization_other, *args,
                                       **kwargs)

    def panel_loss_prob_generalization(self, *args, **kwargs):
        key = 'panel_loss_prob_generalization'
        dec_key = 'prob_generalization'
        if self.data.get(dec_key) is None:
            self.panel_prob_generalization(*args, **kwargs)
        self.make_dec_save_dicts(keys=(dec_key,), loss=True)
        return self.data[key]
    
    def panel_loss_rwd_generalization(self, *args, **kwargs):
        key = 'panel_loss_rwd_generalization'
        dec_key = 'rwd_generalization'
        if self.data.get(dec_key) is None:
            self.panel_rwd_generalization(*args, **kwargs)
        self.make_dec_save_dicts(keys=(dec_key,), loss=True)
        return self.data[key]

    def panel_loss_ev_generalization(self, *args, **kwargs):
        key = 'panel_loss_ev_generalization'
        dec_key = 'ev_generalization'
        if self.data.get(dec_key) is None:
            self.panel_ev_generalization(*args, **kwargs)
        self.make_dec_save_dicts(keys=(dec_key,), loss=True)
        return self.data[key]


def _get_nonlinear_columns(coeffs):
    return _get_template_columns(coeffs, '.*\(nonlinear\)')

def _get_linear_columns(coeffs):
    return _get_template_columns(coeffs, '.*\(linear\)')
    
def _get_template_columns(coeffs, template, columns=None):
    mask = np.array(list(re.match(template, col) is not None
                         for col in columns))
    return np.array(coeffs)[:, mask]

def _get_rwd_lin(coeffs):
    return _get_template_columns(coeffs, 'reward \(linear\)')

def _get_prob_lin(coeffs):
    return _get_template_columns(coeffs, 'prob \(linear\)')

def _get_rwd_nonlin(coeffs):
    return _get_template_columns(coeffs, 'rwd:.* \(nonlinear\)')

def _get_prob_nonlin(coeffs):
    return _get_template_columns(coeffs, 'prob:.* \(nonlinear\)')

def _get_rwd_lin_boot(coeffs, columns):
    return _get_template_columns(coeffs, 'reward', columns=columns)

def _get_prob_lin_boot(coeffs, columns):
    return _get_template_columns(coeffs, 'probability', columns=columns)

def _get_rwd_nonlin_boot(coeffs,columns):
    return _get_template_columns(coeffs, 'rwd x.*', columns=columns)

def _get_prob_nonlin_boot(coeffs, columns):
    return _get_template_columns(coeffs, 'prob x.*', columns=columns)

def _get_nv_boot(coeffs, columns):
    return _get_template_columns(coeffs, 'MSE_TRAIN', columns=columns)

class TheoryFigure(MultipleRepFigure):

    def __init__(self, fig_key='theory_figure', colors=colors, **kwargs):
        fsize = (8, 3)
        cf = u.ConfigParserColor()
        cf.read(config_path)
        
        params = cf[fig_key]
        self.fig_key = fig_key
        self.saved_coefficients = None
        super().__init__(fsize, params, colors=colors, **kwargs)

    def make_gss(self):
        gss = {}
        self.gss = gss

    def get_coeffs_bootstrapped(self, force_reload=False, methods=('Boot-EN',),
                                key='ALL_MONKEY', col_key='variables'):
        if self.data.get('saved_coefficients') is None or force_reload:
            folder = self.params.get('coeff_folder')
            file_ = self.params.get('coeff_boot_file')
            use_regions = self.params.getlist('use_regions')
            time_ind = self.params.getint('coeff_time_ind')
            column_file = self.params.get('coeff_boot_columns')

            out_coeffs = {}
            for method in methods:
                use_folder = folder.format(method=method)
                columns = sio.loadmat(os.path.join(use_folder, column_file))
                columns = np.array(list(c[0] for c in columns[col_key][0]))

                out_coeffs[method] = {}
                coeff_data = sio.loadmat(os.path.join(use_folder, file_))
                coeff_data = coeff_data[key][0, 0]
                for region in list(coeff_data.dtype.names):
                    if region in use_regions:
                        coeff_mat = coeff_data[region][0, 0][0][0, time_ind]
                        out_coeffs[method][region] = (coeff_mat, columns)
                if 'all' in use_regions:
                    coeff_list = list(oc[0] for oc in out_coeffs[method].values())
                    coeff_all = np.concatenate(coeff_list, axis=0)
                    out_coeffs[method]['all'] = (coeff_all, columns)
            self.data['saved_coefficients'] = out_coeffs
        return self.data['saved_coefficients']
        
    def get_coeffs(self, force_reload=False):            
        if self.data.get('saved_coefficients') is None or force_reload:
            methods = self.params.getlist('methods')
            coeff_folder = self.params.get('coeff_folder')
            coeff_template = self.params.get('coeff_template')
            use_regions = self.params.getlist('use_regions')
            
            out_coeffs = {}
            for method in methods:
                out_coeffs[method] = {}
                method_folder = coeff_folder.format(method=method)
                fls = os.listdir(method_folder)
                for fl in fls:
                    m = re.match(coeff_template, fl)
                    if m is not None:
                        region, _ = m.groups()
                        if region in use_regions:
                            ci = pd.read_csv(os.path.join(method_folder, fl))
                            out_coeffs[method][region] = ci
                all_conc = pd.concat(out_coeffs[method].values())
                out_coeffs[method]['all'] = all_conc
            self.data['saved_coefficients'] = out_coeffs
        return self.data['saved_coefficients']
    
    def panel_coeff_prediction(self, force_recompute=False, boot=True):
        key = 'panel_coeff_prediction'
        if boot:
            coeffs = self.get_coeffs_bootstrapped(force_reload=force_recompute)
        else:
            coeffs = self.get_coeffs(force_reload=force_recompute)
        if self.data.get(key) is None or force_recompute:
            out_prediction = {}
            for method, coeffs_m in coeffs.items():
                out_prediction[method] = {}
                for region, cs in coeffs_m.items():
                    if boot:
                        coeffs, columns = cs
                        rwd_lin = _get_rwd_lin_boot(coeffs, columns=columns)
                        prob_lin = _get_prob_lin_boot(coeffs, columns=columns)
                        rwd_nonlin = _get_rwd_nonlin_boot(coeffs,
                                                          columns=columns)
                        prob_nonlin = _get_prob_nonlin_boot(coeffs,
                                                            columns=columns)
                        r2 = np.zeros((rwd_lin.shape[0], 1, 1))
                        resid_var = np.sqrt(_get_nv_boot(coeffs,
                                                         columns=columns))
                    else:
                        rwd_lin = _get_rwd_lin(cs)
                        prob_lin = _get_prob_lin(cs)
                        rwd_nonlin = _get_rwd_nonlin(cs)
                        prob_nonlin = _get_prob_nonlin(cs)
                        r2 = _get_template_columns(cs, 'pseudoR2')
                        if r2.shape[1] > 0:
                            resid_var = 1 - r2
                        else:
                            resid_var = np.expand_dims(1, (0, 1))
                    rwd_out = mra.predict_asymp_dists(
                        rwd_lin, rwd_nonlin, resid_var, k=2, n=2)
                    prob_out = mra.predict_asymp_dists(
                        prob_lin, prob_nonlin, resid_var, k=2, n=2)
                    rwd = {'bind_err':rwd_out[0], 'gen_err':rwd_out[1]}
                    prob = {'bind_err':prob_out[0], 'gen_err':prob_out[1]}
                    
                    out_prediction[method][region] = {'rwd':rwd,
                                                      'prob':prob}
            self.data[key] = out_prediction
        return self.data[key]

    def _get_subspace_correlations(self, force_reload=False,
                                   subspace_key='subspace_corr',
                                   linear_side='without_linear_side_incld'):
        if self.data.get('subspace_correlations') is None or force_reload:
            methods = self.params.getlist('methods')
            coeff_folder = self.params.get('coeff_folder')
            subspace_template = self.params.get('subspace_template')
            use_regions = self.params.getlist('use_regions')

            out_corrs = {}
            for method in methods:
                out_corrs[method] = {}
                method_folder = coeff_folder.format(method=method)
                fls = os.listdir(method_folder)
                for fl in fls:
                    m = re.match(subspace_template, fl)
                    if m is not None:
                        mat = sio.loadmat(os.path.join(method_folder, fl))
                        sk = mat[subspace_key][0, 0][linear_side][0, 0]
                        for region in sk.dtype.names:
                            if region in use_regions:
                                out_corrs[method][region] = sk[region][0, 0][0]
            self.data['subspace_correlations'] = out_corrs
        return self.data['subspace_correlations']

    def panel_subspace_corr_example(self, force_recompute=False):
        key = 'panel_subspace_corr_example'
        n_pwrs = 5
        n_trades = 101
        sum_d2s = np.linspace(5, 10, n_pwrs)**2
        trades = np.linspace(0, 1, n_trades)

        n_feats = self.params.getint('n_feats')
        n_vals = self.params.getint('n_vals')
        sigma = self.params.getfloat('sigma')
        if self.data.get(key) is None or force_recompute:
            gen_err = np.zeros((n_pwrs, n_trades))
            bind_err = np.zeros_like(gen_err)
            for i, sd2 in enumerate(sum_d2s):
                r1, err1 = mrt.vector_corr_ccgp(sd2, trades, sigma=sigma)
                r2, err2 = mrt.vector_corr_swap(sd2, n_feats, n_vals, trades,
                                                sigma=sigma)
                gen_err[i] = err1
                bind_err[i] = err2
            self.data[key] = {'pwrs':sum_d2s, 'r':r1, 'bind_err':bind_err,
                              'gen_err':gen_err}
        return self.data[key]
    
    def panel_subspace_corr_range(self, force_recompute=False):
        key = 'panel_subspace_corr_range'
        corrs = self._get_subspace_correlations(force_reload=force_recompute)
        pwr_params = self.params.getlist('pwr_params', typefunc=float)
        pwrs = np.linspace(*pwr_params[:2], int(pwr_params[2]))
        n_feats = self.params.getint('n_feats')
        n_vals = self.params.getint('n_vals')
        sigma = self.params.getfloat('sigma')
        if self.data.get(key) is None or force_recompute:
            out_arrs = {}
            for method, regions in corrs.items():
                out_arrs[method] = {}
                for region, arr in regions.items():
                    mr_bind = np.zeros(arr.shape + (len(pwrs),))
                    mr_gen = np.zeros_like(mr_bind)
                    for (i, j) in u.make_array_ind_iterator(arr[:, :2].shape):
                        out = mrt.pwr_range_corr(arr[i, j], pwrs, n_feats, n_vals,
                                                 sigma=sigma)
                        mr_bind[i, j] = out[0]
                        mr_gen[i, j] = out[1]
                        mr_bind[i, 2] = arr[i, 2]
                        mr_gen[i, 2] = arr[i, 2]
                    out_arrs[method][region] = {'corr_values':arr, 'bind_err':mr_bind,
                                                'gen_err':mr_gen}
            self.data[key] = {'probed_pwrs':pwrs, 'out':out_arrs}
        return self.data[key]
            
            
