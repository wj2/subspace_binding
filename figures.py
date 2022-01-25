
import os
import re
import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.stats as sts
import functools as ft
import pickle

import general.plotting as gpl
import general.plotting_styles as gps
import general.paper_utilities as pu
import general.utility as u
import multiple_representations.analysis as mra
import multiple_representations.theory as mrt

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
            
    # @property
    # def monkeys(self):
    #     return (self.params.get('monkey1'),
    #             self.params.get('monkey2'))

    # @property
    # def monkey_colors(self):
    #     return self._make_color_dict(self.monkeys)

    # @property
    # def bhv_outcomes(self):
    #     return (self.params.get('bhv_outcome1'),
    #             self.params.get('bhv_outcome2'),
    #             self.params.get('bhv_outcome3'))

    # @property
    # def bhv_colors(self):
    #     return self._make_color_dict(self.bhv_outcomes)
    
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

        # task_schem_grid = self.gs[:, :25]
        # gss['panel_task_schematic'] = self.get_axs((task_schem_grid,))
        
        # err_grid = pu.make_mxn_gridspec(self.gs, 2, 2,
        #                                 0, 100, 35, 50,
        #                                 10, 2)
        # gss['panel_err_distribution'] = self.get_axs(err_grid, sharex=True,
        #                                        sharey=True)

        # model_schem_grid = self.gs[:, 55:75]
        # gss['panel_model_schematic'] = self.get_axs((model_schem_grid,))

        # r_simp_gs = pu.make_mxn_gridspec(self.gs, 2, 1,
        #                                  0, 100, 80, 100,
        #                                  10, 0)
        # r_simp_3d = np.zeros_like(r_simp_gs, dtype=bool)
        # r_simp_3d[1] = False
        # err_rate_ax, simp_ax = self.get_axs(r_simp_gs, plot_3ds=r_simp_3d)

        # gss['panel_err_rates'] = err_rate_ax[0]
        # gss['panel_trial_simplex'] = simp_ax[0]
        
        self.gss = gss

    def get_experimental_data(self, force_reload=False):
        if self.data.get('experimental_data') is None or force_reload:
            datapath = self.params.get('datapath')
            data = gio.Dataset.from_readfunc(mraux.load_fine_data, data_folder)
            
            exclude_list = self.params.getlist('exclude_datasets')
            session_mask = np.logical_not(np.isin(data['date'], exclude_list))
            self.data['experimental_data'] = data.session_mask(session_mask)
        return self.data['experimental_data']

    def _decoding_analysis(self, var, func, force_reload=False, **kwargs):
        tbeg = self.params.getint('tbeg')
        tend = self.params.getint('tend')
        winsize = self.params.getint('winsize')
        winstep = self.params.getint('winstep')
        prs = self.params.getint('resamples')
        pca_pre = self.params.getfloat('pca_pre')
        time_acc = self.params.getbool('time_accumulate')

        exper_data = self.get_experimental_data(force_reload=force_reload)

        decoding_results = func(exper_data, tbeg, tend, var,
                                winsize=winsize, dead_perc=None,
                                pre_pca=pca_pre, pop_resamples=prs,
                                tstep=winstep, time_accumulate=time_acc,
                                **kwargs)

    def prob_generalization(self, force_reload=False):
        dead_perc = self.getfloat('exclude_middle_percentiles')
        mask_func = lambda x: x < 1
        if self.data.get('prob-gen') is None:
            out = self._decoding_analysis('prob', force_reload=force_reload,
                                          dead_perc=dead_perc, mask_func=mask_func)
            self.data['prob-gen'] = out
        return self.data['prob-gen']

    def rwd_generalization(self, force_reload=False):
        if self.data.get('rwd-gen') is None:
            out = self._decoding_analysis('rwd', force_reload=force_reload)
            self.data['rwd-gen'] = out
        return self.data['rwd-gen']


def _get_nonlinear_columns(coeffs):
    return _get_template_columns(coeffs, '.*\(nonlinear\)')

def _get_linear_columns(coeffs):
    return _get_template_columns(coeffs, '.*\(linear\)')
    
def _get_template_columns(coeffs, template):
    mask = np.array(list(re.match(template, col) is not None
                         for col in coeffs.columns))
    return np.array(coeffs)[:, mask]
    
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
                
    def panel_coeff_prediction(self, force_recompute=False):
        key = 'panel_coeff_prediction'
        coeffs = self.get_coeffs(force_reload=force_recompute)
        if self.data.get(key) is None or force_recompute:
            out_prediction = {}
            for method, coeffs_m in coeffs.items():
                out_prediction[method] = {}
                for region, cs in coeffs_m.items():
                    lin = _get_linear_columns(cs)
                    nonlin = _get_nonlinear_columns(cs)
                    r2 = _get_template_columns(cs, 'pseudoR2')
                    if r2.shape[1] > 0:
                        resid_var = 1 - r2
                    else:
                        resid_var = np.expand_dims(1, (0, 1))
                    out = mra.predict_ccgp_binding_noformat(lin, nonlin,
                                                            resid_var,
                                                            n=2)
                    bind_avg, gen_avg = out
                    out_prediction[method][region] = {'bind_err':bind_avg,
                                                      'gen_err':gen_avg}
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
            
    def save_pickles(self, folder, suff='', ext='.mat'):
        for key in self.panel_keys:
            panel_data = self.data[key]
            filename = key.split('_', 1)[1] + suff + ext
            path = os.path.join(folder, filename)
            sio.savemat(path, panel_data)
            
