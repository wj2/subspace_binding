import os
import re
import numpy as np
import pandas as pd
import scipy.io as sio
import sklearn.linear_model as sklm
import sklearn.preprocessing as skp
import matplotlib.pyplot as plt

import general.plotting as gpl
import general.paper_utilities as pu
import general.data_io as gio
import general.utility as u
import multiple_representations.analysis as mra
import multiple_representations.auxiliary as mraux
import multiple_representations.theory as mrt
import multiple_representations.visualization as mrv
import multiple_representations.direct_theory as mrdt

config_path = "multiple_representations/figures.conf"

colors = (
    np.array(
        [
            (127, 205, 187),
            (65, 182, 196),
            (29, 145, 192),
            (34, 94, 168),
            (37, 52, 148),
            (8, 29, 88),
        ]
    )
    / 256
)


class MultipleRepFigure(pu.Figure):

    def get_experimental_data(self, force_reload=False, data_folder=None):
        if self.data.get("experimental_data") is None or force_reload:
            if data_folder is None:
                data_folder = self.params.get("data_folder")
            data = gio.Dataset.from_readfunc(mraux.load_fine_data, data_folder)

            exclude_list = self.params.getlist("exclude_datasets")
            session_mask = np.logical_not(np.isin(data["date"], exclude_list))
            self.data["experimental_data"] = data.session_mask(session_mask)
        return self.data["experimental_data"]

    def _get_gen_emp_pred(self, dec_folder=None, pred_file=None, dec_file=None):
        if dec_folder is None:
            dec_folder = self.params.get("decoding_folder")
        if pred_file is None:
            pred_file = self.params.get("pred_file")
        if dec_file is None:
            dec_file = self.params.get("dec_file")
        preds = sio.loadmat(os.path.join(dec_folder, pred_file))
        emp = sio.loadmat(os.path.join(dec_folder, dec_file))
        return preds, emp

    def _make_color_dict(self, ks):
        color_dict = {}
        for k in ks:
            color_dict[k] = self.params.getcolor(k + "_color")
        return color_dict

    def get_exper_data(self):
        if self.data.get("exper_data") is None:
            data_folder = self.params.get("data_folder")
            data = gio.Dataset.from_readfunc(mraux.load_fine_data, data_folder)
            self.data["exper_data"] = data
        return self.data["exper_data"]

    def get_color_dict(self):
        ofc_color = self.params.getcolor("ofc_color")
        pcc_color = self.params.getcolor("pcc_color")
        pgacc_color = self.params.getcolor("pgacc_color")
        vmpfc_color = self.params.getcolor("vmpfc_color")
        vs_color = self.params.getcolor("vs_color")
        all_color = self.params.getcolor("all_color")
        color_dict = {
            "OFC": ofc_color,
            "PCC": pcc_color,
            "pgACC": pgacc_color,
            "vmPFC": vmpfc_color,
            "VS": vs_color,
            "all": all_color,
        }
        return color_dict

    def save_pickles(self, folder, suff="", ext=".mat"):
        unsaved_keys = []
        for key in self.panel_keys:
            panel_data = self.data.get(key)
            if panel_data is not None:
                filename = key.split("_", 1)[1] + suff + ext
                path = os.path.join(folder, filename)
                for k, v in panel_data.items():
                    if v is None:
                        k[v] = np.nan
                    else:
                        for kd, vd in v.items():
                            if vd is None:
                                v[kd] = np.nan

                sio.savemat(path, panel_data)
            else:
                unsaved_keys.append(key)
        if len(unsaved_keys) > 0:
            print(
                "the following keys did not exist and were not "
                "saved:\n{}".format(unsaved_keys)
            )

    @property
    def regions(self):
        regions = self.params.getlist("use_regions")
        return regions


def _accumulate_time(pop, keepdim=True, ax=1):
    out = np.concatenate(list(pop[..., i] for i in range(pop.shape[-1])), axis=ax)
    if keepdim:
        out = np.expand_dims(out, -1)
    return out


def _subsample_pops(*pops, samp_pops=10):
    n_pops = pops[0].shape[0]
    inds = np.random.default_rng().choice(n_pops, size=samp_pops, replace=False)
    new_pops = list(pop_i[inds] for pop_i in pops)
    return new_pops


class MonkeyDistances(MultipleRepFigure):
    def __init__(
            self,
            fig_key="monkey_dist_figure",
            colors=colors,
            **kwargs
    ):
        fsize = (6, 3)
        cf = u.ConfigParserColor()
        cf.read(config_path)

        params = cf[fig_key]
        self.fig_key = fig_key

        super().__init__(fsize, params, colors=colors, **kwargs)

    def make_gss(self):
        gss = {}

        n_regions = len(self.regions) - 1
        region_gs = pu.make_mxn_gridspec(self.gs, 2, n_regions, 0, 100, 0, 100, 8, 3)
        gss["panel_regions"] = self.get_axs(region_gs, squeeze=False, sharey="all")
        self.gss = gss

    def panel_regions(self):
        key = "panel_regions"
        axs = self.gss[key]
        if self.data.get(key) is None:
            runinds = self.params.getlist("runinds")
            comb_dict = mraux.load_monkey_region_runs(*runinds)
            self.data[key] = comb_dict
        comb_dict = self.data[key]

        l_color = self.params.getcolor("lin_color")
        n_color = self.params.getcolor("conj_color")
        regions = self.params.getlist("use_regions")[:-1]
        mrv.plot_monkey_pred_dict(
            comb_dict,
            axs=axs,
            l_color=[l_color],
            n_color=[n_color],
            region_list=regions,
        )
        for i, j in u.make_array_ind_iterator(axs.shape):
            if j == 0:
                axs[i, j].set_ylabel("estimated distance")


class SelectivityFigure(MultipleRepFigure):
    def __init__(
            self,
            fig_key="selectivity_figure",
            filter_performance=False,
            colors=colors,
            **kwargs
    ):
        fsize = (5, 8)
        cf = u.ConfigParserColor()
        cf.read(config_path)

        params = cf[fig_key]
        self.fig_key = fig_key
        self.filter_performance = filter_performance

        super().__init__(fsize, params, colors=colors, **kwargs)

    def get_exper_data(self):
        data = super().get_exper_data()
        if self.filter_performance:
            thr = self.params.getfloat("performance_thr")
            data = mraux.filter_low_performance(data, thr)
        return data

    def make_gss(self):
        gss = {}

        n_regions = len(self.regions) - 1
        n_egs = len(self.params.getlist("use_dates"))
        eg_neuron_gs = pu.make_mxn_gridspec(self.gs, 2, n_egs, 0, 25, 0, 100, 5, 5)
        gss["eg_neurons"] = self.get_axs(eg_neuron_gs, squeeze=False, sharey=True).T

        model_comp_gs = pu.make_mxn_gridspec(
            self.gs, 1, n_regions, 30, 45, 0, 100, 5, 1
        )
        gss["model_comp"] = self.get_axs(model_comp_gs, squeeze=True)

        subspace_corr_gs = pu.make_mxn_gridspec(self.gs, 2, 1, 50, 100, 50, 100, 5, 5)
        gss["subspace_corr"] = self.get_axs(subspace_corr_gs, squeeze=True)

        self.gss = gss



    def panel_eg_neurons(self):
        key = "eg_neurons"
        axs = self.gss[key]

        data = self.get_exper_data()
        use_dates = self.params.getlist("use_dates")
        use_neurs = self.params.getlist("neur_inds", typefunc=int)

        lv_color = self.params.getcolor("left_value_color")
        rv_color = self.params.getcolor("right_value_color")

        avg_beg = self.params.getint("avg_beg")
        avg_end = self.params.getint("avg_end")

        label_dict = {
            -1: ("left low value", "left high value"),
            1: ("right low value", "right high value"),
        }
        use_nl_val = self.params.getboolean("use_nl_value")
        pairs = zip(use_dates, use_neurs)
        for i, (date, neur_ind) in enumerate(pairs):
            xs, act, vs, sides = mra.make_psth_val_avg(data, date, neur_ind)
            region, monkey, num = date.split("-")
            neur_key = (region, monkey, num, neur_ind)
            if self.data.get(neur_key) is None:
                pred = np.stack((vs, sides), axis=1)
                pred = mra.form_predictors(pred, transform_value=use_nl_val)
                x_mask = np.logical_and(xs >= avg_beg, xs < avg_end)
                targ = np.mean(act[:, x_mask], axis=1, keepdims=True)
                scaler = skp.StandardScaler()
                scaler.fit(targ)
                targ = scaler.transform(targ)
                print(pred.shape, targ.shape)
                _, fits, diags = mra._make_dict_and_fit(pred, targ)
                self.data[neur_key] = fits[0], scaler

            fit, scaler = self.data[neur_key]

            side_colors = (lv_color, rv_color)
            mrv.plot_value_resp(
                xs, act, vs, sides, ax=axs[i, 0], side_colors=side_colors
            )
            mrv.plot_value_tuning(
                xs, act, vs, sides, ax=axs[i, 1], side_colors=side_colors
            )
            pp_act = np.mean(fit.posterior_predictive["err_hat"], axis=(0, 1))
            model_act = scaler.inverse_transform(np.expand_dims(pp_act, 1))
            mrv.plot_value_tuning(
                xs,
                model_act,
                vs,
                sides,
                ax=axs[i, 1],
                side_colors=side_colors,
                linestyle="dashed",
            )
            axs[i, 0].set_title(region)
        use_ind = int(np.floor(axs.shape[0] / 2))
        axs[use_ind, 0].set_xlabel("time from stimulus onset")
        axs[use_ind, 1].set_xlabel("stimulus value")
        axs[0, 0].set_ylabel("spikes/s")
        axs[0, 1].set_ylabel("spikes/s")

    def _get_model_comp_data(self, key="model_comp", use_folder=None):
        if self.data.get(key) is None:
            if use_folder is None:
                fit_folder = self.params.get("fit_folder")
            else:
                fit_folder = use_folder
            _, comp_dict = mraux.load_many_fits(fit_folder)
            self.data[key] = comp_dict
        return self.data[key]
        
    def panel_model_comp(self, use_folder=None):
        key = "model_comp"
        axs = self.gss[key]
        comp_dict = self._get_model_comp_data(use_folder=use_folder)
        ax_labels = ("noise", "linear", "interaction")
        use_regions = self.regions[:-1]

        use_dates = self.params.getlist("use_dates")
        use_neurs = self.params.getlist("neur_inds", typefunc=int)

        for i, r in enumerate(use_regions):
            r_color = self.get_color_dict()[r]
            if r == "all":
                r_list = None
            else:
                r_list = (r,)
            s_i = tuple(use_dates[i].split("-")[:2])
            d_i = use_dates[i]
            n_i = use_neurs[i]
            h_i = s_i + (d_i, n_i)
            highlight_inds = (h_i,)
            mrv.visualize_model_weights(
                comp_dict,
                ax_labels=ax_labels,
                use_regions=r_list,
                ax=axs[i],
                highlight_inds=highlight_inds,
                pt_col=r_color,
                ms=3,
            )

    def panel_subspace_corr(self, recompute=False): 
        key_gen = "subspace_corr_nominal"
        key_spec = "subspace_corr_orig"
        key_ax = "subspace_corr"
        axs = self.gss[key_ax]

        tb_on = self.params.getint("tbeg_on")
        te_on = self.params.getint("tend_on")
        tb_delay = self.params.getint("tbeg_delay")
        te_delay = self.params.getint("tend_delay")

        if self.data.get(key_gen) is None or recompute:
            ms_on_dict = self.fit_subspace_models(tb_on, te_on)
            ms_on_shuff = self.fit_subspace_models(tb_on, te_on, shuffle_targs=True)
            ms_delay_dict = self.fit_subspace_models(tb_delay, te_delay)
            ms_delay_shuff = self.fit_subspace_models(
                tb_delay, te_delay, shuffle_targs=True
            )
            self.data[key_gen] = (ms_on_dict, ms_on_shuff, ms_delay_dict, ms_delay_shuff)
        ms_list = self.data[key_gen]
        if self.data.get(key_spec) is None or recompute or True:
            r_dicts = list({r: {} for r in self.regions} for ms in ms_list)
            for i, r in enumerate(self.regions):
                use_m = "all"
                for j, ms_dict in enumerate(ms_list):
                    r_dicts[j][r] = r_dicts[j][r].get(use_m, {})
                    r_dicts[j][r][use_m] = self.compute_subspace_corr(
                        ms_dict,
                        r,
                        use_m,
                        model_combination="interaction",
                    )
            self.data[key_spec] = r_dicts

        r_on, r_on_shuff, r_delay, r_delay_shuff = self.data[key_spec]
        markers = {}
        self.plot_subspace_corr(axs[0], r_on, style_dict=markers, shuff=r_on_shuff)
        self.plot_subspace_corr(
            axs[1], r_delay, style_dict=markers, shuff=r_delay_shuff, time="DELAY"
        )
        if self.params.getboolean("use_nl_val"):
            y_label = "alignment index"
        else:
            y_label = "subspace correlation"
        axs[0].set_ylabel(y_label)
        axs[1].set_ylabel(y_label)
        axs[1].spines["bottom"].set_visible(False)
        gpl.clean_plot_bottom(axs[0])
        
    def panel_subspace_corr_subset(self, recompute=False): 
        key_gen = "subspace_corr"
        key_spec = "subspace_corr_subset"
        axs = self.gss[key_gen]

        tb_on = self.params.getint("tbeg_on")
        te_on = self.params.getint("tend_on")
        tb_delay = self.params.getint("tbeg_delay")
        te_delay = self.params.getint("tend_delay")

        if self.data.get(key_gen) is None or recompute:
            data = self.get_exper_data()
            min_neurs = self.params.getint("min_neurs_bhv")
            session_mask = data["n_neurs"] > min_neurs
            data_filt = data.session_mask(session_mask)
            ms_on_dict = self.fit_subspace_models(tb_on, te_on, data=data_filt)
            ms_on_shuff = self.fit_subspace_models(
                tb_on, te_on, data=data_filt, shuffle_targs=True
            )
            ms_delay_dict = self.fit_subspace_models(tb_delay, te_delay, data=data_filt)
            ms_delay_shuff = self.fit_subspace_models(
                tb_delay, te_delay, data=data_filt, shuffle_targs=True
            )
            self.data[key_gen] = (ms_on_dict, ms_on_shuff, ms_delay_dict, ms_delay_shuff)
        ms_list = self.data["subspace_corr"]
        use_model_comb = "interaction"
        if self.data.get(key_spec) is None or recompute:
            r_dicts = list({} for ms in ms_list)
            for i, r in enumerate(self.regions):
                use_m = "all"
                for j, ms_dict in enumerate(ms_list):
                    r_dicts[j][r] = r_dicts[j].get(r, {})
                    try:
                        r_dicts[j][r][use_m] = self.compute_subspace_corr(
                            ms_dict,
                            r,
                            use_m,
                            model_combination=use_model_comb,
                        )
                    except ValueError as e:
                        s = ("an error occurred for {} and in monkey {}, this might "
                             "be because there are no sessions meeting the subset "
                             "criteria")
                        print(
                            s.format(r, use_m)
                        )
                        print(e)
            self.data[key_spec] = r_dicts

        r_on, r_on_shuff, r_delay, r_delay_shuff = self.data[key_spec]
        markers = {}
        self.plot_subspace_corr(axs[0], r_on, style_dict=markers, shuff=r_on_shuff)
        self.plot_subspace_corr(
            axs[1], r_delay, style_dict=markers, shuff=r_delay_shuff, time="DELAY"
        )
        if self.params.getboolean("use_nl_val"):
            y_label = "alignment index"
        else:
            y_label = "subspace correlation"
        axs[0].set_ylabel(y_label)
        axs[1].set_ylabel(y_label)
        axs[1].spines["bottom"].set_visible(False)
        gpl.clean_plot_bottom(axs[0])
       
        
    def panel_subspace_corr_time(self, recompute=False):
        key_gen = "subspace_corr_time_models"
        key_spec = "subspace_corr_time"
        key_ax = "subspace_corr"
        axs = self.gss[key_ax]
        tb_on = self.params.getint("tbeg_on")
        te_on = self.params.getint("tend_on")
        tb_delay = self.params.getint("tbeg_delay")
        te_delay = self.params.getint("tend_delay")

        groups = {
            "full": {"t1_only": False, "t2_only": False, "decrement": 2},
            "offer 1": {"t1_only": True, "t2_only": False},
            "offer 2": {"t1_only": False, "t2_only": True},
        }
        if self.data.get(key_gen) is None or recompute:
            out_dict = {}
            for k, kw in groups.items():
                out_on = self.fit_subspace_models(tb_on, te_on, **kw)
                out_on_shuff = self.fit_subspace_models(
                    tb_on, te_on, shuffle_targs=True,
                )
                out_delay = self.fit_subspace_models(tb_delay, te_delay, **kw)
                out_delay_shuff = self.fit_subspace_models(
                    tb_delay, te_delay, shuffle_targs=True,
                )
                out_dict[k] = (out_on, out_on_shuff, out_delay, out_delay_shuff)
            self.data[key_gen] = out_dict

        ms_group_dict = self.data[key_gen]
        if self.data.get(key_spec) is None or recompute:
            r_group_dict = {}
            for k, ms_list in ms_group_dict.items():
                r_dicts = list({r: {} for r in self.regions} for ms in ms_list)
                for i, r in enumerate(self.regions):
                    use_m = "all"
                    for j, ms_dict in enumerate(ms_list):
                        r_dicts[j][r] = r_dicts[j][r].get(use_m, {})
                        r_dicts[j][r][use_m] = self.compute_subspace_corr(
                            ms_dict,
                            r,
                            use_m,
                            model_combination="interaction",
                        )
                r_group_dict[k] = r_dicts
            self.data[key_spec] = r_group_dict

        r_group_dict = self.data[key_spec]
        null_color = self.params.getcolor("null_corr_color")
        style_kw = {
            "offer 1": {"all": {"markerstyles": "o"}},
            "offer 2": {"all": {"markerstyles": "s"}},
            # "full": {"markerstyles": "s"},
        }
        offsets = {
            "offer 1": -.2,
            "offer 2": 0,
            "full": .2
        }
        for k, r_list in r_group_dict.items():
            r_on, r_on_shuff, r_delay, r_delay_shuff = r_list
            print(k)
            self.plot_subspace_corr(
                axs[0],
                r_on,
                style_dict=style_kw.get(k),
                shuff=r_on_shuff,
                offset=offsets.get(k, 0),
            )
            self.plot_subspace_corr(
                axs[1],
                r_delay,
                offset=offsets.get(k, 0),
                style_dict=style_kw.get(k),
                shuff=r_delay_shuff,
                time="DELAY",
            )
        if self.params.getboolean("use_nl_val"):
            y_label = "alignment index"
        else:
            y_label = "subspace correlation"
        axs[0].set_ylabel(y_label)
        axs[1].set_ylabel(y_label)
        axs[1].spines["bottom"].set_visible(False)
        gpl.clean_plot_bottom(axs[0])


    def fit_subspace_models(
            self,
            t_on,
            t_off,
            tstep=None,
            tsize=None,
            discretize_value=False,
            norm_value=True,
            data=None,
            shuffle_targs=False,
            **kwargs,
    ):
        if data is None:
            data = self.get_exper_data()

        model_specs = {
                "linear": {"include_interaction": False},
                "interaction": {"include_interaction": True},
        }
        n_value_bins = self.params.getint("n_value_bins")
        use_nl_val = self.params.getboolean("use_nl_value")

        ms_dict = {}
        for ms_k, ms in model_specs.items():
            out_sessions, xs = mra.make_predictor_matrices(
                data,
                t_beg=t_on,
                t_end=t_off,
                tstep=tstep,
                twin=tsize,
                transform_value=use_nl_val,
                return_core_predictors=True,
                norm_value=norm_value,
                discretize_value=discretize_value,
                n_value_bins=n_value_bins,
                shuffle_targets=shuffle_targs,
                **ms,
                **kwargs,
            )
            boots = mra.fit_bootstrap_models(out_sessions)
            ms_dict[ms_k] = boots, xs, out_sessions
        return ms_dict

    def compute_subspace_corr(
            self,
            ms_dict,
            region,
            monkey,
            model_combination="interaction",
            discretize_value=False,
            out_sessions=None,
            use_folder=None,
    ):
        if model_combination == "weighted_sum":
            comp_dict = self._get_model_comp_data(use_folder=use_folder)
        else:
            comp_dict = None

        use_nl_val = self.params.getboolean("use_nl_value")
        if use_nl_val:
            align_func = mra.compute_alignment_index
            pred_dict = {
                "linear": mra.null_spline_pred,
                "interaction": mra.interaction_spline_pred,
            }
        else:
            align_func = mra.compute_corr
            pred_dict = {
                "linear": mra.null_pred,
                "interaction": mra.interaction_pred,
            }
        if region == "all":
            r_list = None
        else:
            r_list = (region,)
        if monkey == "all" or monkey is None:
            m_list = None
        else:
            m_list = (monkey,)
        n_value_trials = self.params.getint("n_value_trials")

        out_on = mra.compute_split_halfs_model_mix(
            ms_dict,
            comp_dict,
            pred_dict,
            align_func=align_func,
            use_regions=r_list,
            full_data=out_sessions,
            n_full_data_trials=n_value_trials,
            use_monkeys=m_list,
            compute_null=discretize_value,
            model_combination=model_combination,
        )
        return out_on

    def plot_subspace_corr_tc(
            self,
            ax,
            r_dict,
            xs,
            style_dict=None,
            color_dict=None,
            shuff=None,
            normalize=False,
            x_label="time from offer onset (ms)"
    ):
        if style_dict is None:
            style_dict = {}
        if not u.check_list(ax):
            axs = (ax,)*len(self.regions)
        else:
            axs = ax
        null_color = self.params.getcolor("null_corr_color")
        for i, r in enumerate(self.regions):
            r_color = self.get_color_dict()[r]
            for j, (m, (rs_wi, rs_ac, rs_null)) in enumerate(r_dict[r].items()):
                ax = axs[i]
                if shuff is not None:
                    null = shuff[r][m][0]
                else:
                    null = np.zeros_like(rs_wi)
                if normalize:
                    normed = (rs_wi - null) / (rs_ac - null)
                    gpl.plot_trace_werr(xs, normed, color=r_color, conf95=True, ax=ax)
                else:
                    gpl.plot_trace_werr(xs, rs_wi, color=r_color, conf95=True, ax=ax)
                    gpl.plot_trace_werr(xs, rs_ac, color=null_color, conf95=True, ax=ax)
                    gpl.plot_trace_werr(xs, null, color=null_color, conf95=True, ax=ax)
                ax.set_xlabel(x_label)
        return axs

    def plot_subspace_corr(
            self, ax, r_dict, style_dict=None, offset=.5, shuff=None, time="ON",
    ):
        if style_dict is None:
            style_dict = {}
            
        null1_color = self.params.getcolor("null_corr_color")
        null2_color = self.params.getcolor("null_corr_color")
        for i, r in enumerate(self.regions):
            r_color = self.get_color_dict()[r]
            n_items = len(r_dict[r])
            for j, (m, (rs_wi, rs_ac, rs_null)) in enumerate(r_dict[r].items()):
                if shuff is not None:
                    rs_null = shuff[r][m][0]
                mrv.plot_split_halfs_only(
                    rs_wi,
                    rs_ac,
                    rs_null,
                    ax=ax,
                    pt=i + (j - n_items/2)*offset/n_items,
                    colors=(r_color, null1_color, null2_color),
                    **style_dict.get(m, {})
                )
                high, low = u.conf_interval(
                    rs_ac - rs_wi, withmean=True, perc=90,
                )[:, 0]
                s1 = ("{region}, {monkey}, {time}: {low:.2f} to {high:.2f} "
                      "difference from ceiling")
                print(s1.format(
                    monkey=m, region=r, time=time, low=low, high=high,
                ))
                
                high, low = u.conf_interval(
                    rs_wi - rs_null, withmean=True, perc=90,
                )[:, 0]
                s = ("{region}, {monkey}, {time}: {low:.2f} to {high:.2f} "
                     "difference from floor")
                print(s.format(
                    monkey=m, region=r, time=time, low=low, high=high,
                ))
                    
        gpl.clean_plot(ax, 0)
        gpl.add_hlines(0, ax)
        ax.set_xticks(np.arange(len(self.regions)))
        ax.set_xticklabels(self.regions)
        ax.set_ylabel("subspace correlation")

    def panel_subspace_corr_tc(self, recompute=False, new_axes=True, fwid=2):
        key_gen = "subspace_corr_tc_models"
        key_spec = "subspace_corr_tc"
        key_ax = "subspace_corr"
        if new_axes:
            n_panels = len(self.regions)
            f, ax = plt.subplots(
                1, n_panels, figsize=(fwid*n_panels, fwid), sharex=True, sharey=True
            )
        else:
            ax = self.gss[key_ax][0]
            f = None
        tc_start = self.params.getint("tc_tbeg")
        tc_end = self.params.getint("tc_tend")
        twin = self.params.getint("tc_winsize")
        tstep = self.params.getint("tc_winstep")
        
        if self.data.get(key_gen) is None or recompute:
            ms_dict = self.fit_subspace_models(tc_start, tc_end, tstep, twin)
            ms_shuff = self.fit_subspace_models(
                tc_start, tc_end, tstep, twin, shuffle_targs=True,
            )
            
            self.data[key_gen] = (ms_dict, ms_shuff)
        ms_dict, ms_shuff = self.data[key_gen]
        xs = ms_dict["linear"][1]
        if self.data.get(key_spec) is None or recompute:
            r_dict = {}
            r_shuff = {}
            for i, r in enumerate(self.regions):
                r_dict[r] = {}
                r_shuff[r] = {}
                r_dict[r]["full"] = self.compute_subspace_corr(
                    ms_dict, r, None, model_combination="interaction",
                )
                r_shuff[r]["full"] = self.compute_subspace_corr(
                    ms_shuff, r, None, model_combination="interaction",
                )
            self.data[key_spec] = r_dict, r_shuff
        r_dict, r_shuff = self.data[key_spec]

        axs = self.plot_subspace_corr_tc(ax, r_dict, xs, shuff=r_shuff)
        if self.params.getboolean("use_nl_val"):
            y_label = "alignment index"
        else:
            y_label = "subspace correlation"
        axs[0].set_ylabel(y_label)
        if not new_axes:
            gpl.clean_plot_bottom(axs[0])
        return f
        
    def panel_subspace_corr_monkey(self, recompute=False):
        key_gen = "subspace_corr"
        key_spec = "subspace_corr_monkey"
        axs = self.gss[key_gen]

        tb_on = self.params.getint("tbeg_on")
        te_on = self.params.getint("tend_on")
        tb_delay = self.params.getint("tbeg_delay")
        te_delay = self.params.getint("tend_delay")

        if self.data.get(key_gen) is None or recompute:
            ms_on_dict = self.fit_subspace_models(tb_on, te_on)
            ms_on_shuff = self.fit_subspace_models(tb_on, te_on, shuffle_targs=True)
            ms_delay_dict = self.fit_subspace_models(tb_delay, te_delay)
            ms_delay_shuff = self.fit_subspace_models(
                tb_delay, te_delay, shuffle_targs=True
            )
            self.data[key_gen] = (ms_on_dict, ms_on_shuff, ms_delay_dict, ms_delay_shuff)
        ms_list = self.data["subspace_corr"]
        if self.data.get(key_spec) is None or recompute:
            r_list = list({r: {} for r in self.regions} for ms in ms_list)
            for i, r in enumerate(self.regions):
                monkeys = mraux.region_monkey_dict[r]
                for use_m in monkeys:
                    for j, ms in enumerate(ms_list):
                        r_list[j][r][use_m] = self.compute_subspace_corr(
                        ms,
                        r,
                        use_m,
                        model_combination="interaction",
                    )
            self.data[key_spec] = r_list

        r_on, r_on_shuff, r_delay, r_delay_shuff = self.data[key_spec]
        markers = {
            "Batman": {"markerstyles": "s"},
            "Calvin": {"markerstyles": "o"},
            "Hobbes": {"markerstyles": "x"},
            "Pumbaa":{"markerstyles": "*"},
            "Spock": {"markerstyles": "v"},
            "Vader": {"markerstyles": "D"},
        }
        self.plot_subspace_corr(axs[0], r_on, style_dict=markers, shuff=r_on_shuff)
        self.plot_subspace_corr(
            axs[1], r_delay, style_dict=markers, shuff=r_delay_shuff, time="DELAY"
        )
        if self.params.getboolean("use_nl_val"):
            y_label = "alignment index"
        else:
            y_label = "subspace correlation"
        axs[0].set_ylabel(y_label)
        axs[1].set_ylabel(y_label)
        axs[1].spines["bottom"].set_visible(False)
        gpl.clean_plot_bottom(axs[0])
        
    def panel_subspace_corr_model_mix(self, recompute=False, use_folder=None):
        key_gen = "subspace_corr"
        key_spec = "subspace_corr_model_mix"
        axs = self.gss[key_gen]

        tb_on = self.params.getint("tbeg_on")
        te_on = self.params.getint("tend_on")
        tb_delay = self.params.getint("tbeg_delay")
        te_delay = self.params.getint("tend_delay")

        if self.data.get(key_gen) is None or recompute:
            ms_on_dict = self.fit_subspace_models(tb_on, te_on)
            ms_on_shuff = self.fit_subspace_models(tb_on, te_on, shuffle_targs=True)
            ms_delay_dict = self.fit_subspace_models(tb_delay, te_delay)
            ms_delay_shuff = self.fit_subspace_models(
                tb_delay, te_delay, shuffle_targs=True
            )
            self.data[key_gen] = (ms_on_dict, ms_on_shuff, ms_delay_dict, ms_delay_shuff)
        ms_list = self.data[key_gen]
        model_combination = "weighted_sum"
        monkey = "all"
        if self.data.get(key_spec) is None or recompute:
            r_list = list({r: {} for r in self.regions} for ms in ms_list)
            for i, r in enumerate(self.regions):
                for j, ms in enumerate(ms_list):
                    r_list[j][r][monkey] = self.compute_subspace_corr(
                        ms,
                        r,
                        monkey,
                        model_combination=model_combination,
                        use_folder=use_folder,
                    )
            self.data[key_spec] = r_list

        r_on, r_on_shuff, r_delay, r_delay_shuff = self.data[key_spec]
        markers = {
            "Batman": {"markerstyles": "s"},
            "Calvin": {"markerstyles": "o"},
            "Hobbes": {"markerstyles": "x"},
            "Pumbaa":{"markerstyles": "*"},
            "Spock": {"markerstyles": "v"},
            "Vader": {"markerstyles": "D"},
        }
        self.plot_subspace_corr(axs[0], r_on, style_dict=markers, shuff=r_on_shuff)
        self.plot_subspace_corr(
            axs[1], r_delay, style_dict=markers, shuff=r_delay_shuff, time="DELAY"
        )
        if self.params.getboolean("use_nl_val"):
            y_label = "alignment index"
        else:
            y_label = "subspace correlation"
        axs[0].set_ylabel(y_label)
        axs[1].set_ylabel(y_label)
        axs[1].spines["bottom"].set_visible(False)
        gpl.clean_plot_bottom(axs[0])


class DecodingFigure(MultipleRepFigure):
    def __init__(self, fig_key="decoding_figure", colors=colors, **kwargs):
        fsize = (8, 3)
        cf = u.ConfigParserColor()
        cf.read(config_path)

        params = cf[fig_key]
        self.fig_key = fig_key
        super().__init__(fsize, params, colors=colors, **kwargs)

    def make_gss(self):
        gss = {}
        self.gss = gss

    def _decoding_analysis(self, var, func, force_reload=True, **kwargs):
        tbeg = self.params.getint("tbeg")
        tend = self.params.getint("tend")
        winsize = self.params.getint("winsize")
        winstep = self.params.getint("winstep")
        prs = self.params.getint("resamples")
        pca_pre = self.params.getfloat("pca_pre")
        time_acc = self.params.getboolean("time_accumulate")
        regions = self.params.getlist("use_regions")
        correct_only = self.params.getboolean("correct_only")
        subsample_neurons = self.params.getint("subsample_neurons")

        exper_data = self.get_experimental_data(force_reload=force_reload)

        decoding_results = {}
        for region in regions:
            if region == "all":
                use_regions = None
            else:
                use_regions = (region,)
            print(kwargs)
            print(use_regions, subsample_neurons, correct_only)
            print(var)
            out_r = func(
                exper_data,
                tbeg,
                tend,
                var,
                winsize=winsize,
                pre_pca=pca_pre,
                pop_resamples=prs,
                tstep=winstep,
                time_accumulate=time_acc,
                regions=use_regions,
                correct_only=correct_only,
                subsample_neurons=subsample_neurons,
                **kwargs
            )
            decoding_results[region] = out_r
        return decoding_results

    def _fit_linear_models(self, dec_dict, force_reload=False):
        pca_pre = self.params.getfloat("pca_pre")
        l1_ratio = self.params.getfloat("l1_ratio")
        test_prop = self.params.getfloat("test_prop")
        multi_task = self.params.getboolean("multi_task")
        folds_n = self.params.getint("n_folds")
        samp_pops = self.params.getint("linear_fit_pops")
        internal_cv = self.params.getboolean("internal_cv")
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
                out = mra.fit_linear_models(
                    pops,
                    conds,
                    folds_n=folds_n,
                    model=model,
                    multi_task=multi_task,
                    l1_ratio=l1_ratio,
                    pre_pca=pca_pre,
                    test_prop=test_prop,
                    max_iter=10000,
                    internal_cv=internal_cv,
                )
                lm, nlm, nv, r2 = out
                out_dict = {
                    "lm": lm,
                    "nlm": nlm,
                    "nv": nv,
                    "r2": r2,
                }
                done_keys.append(set(contrast))
                lm_dict[contrast] = out_dict
        return lm_dict

    def _model_terms(self, key, dec_key, force_recompute=False):
        if self.data.get(dec_key) is None and self.data.get(key) is None:
            self.panel_prob_generalization()
        dec = self.data[dec_key]
        use_regions = self.params.getlist("use_regions")
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
                lm = term_dict["lm"]
                nlm = term_dict["nlm"]
                nv = term_dict["nv"]
                # r2 = term_dict["r2"]
                pred_out = mra.compute_ccgp_bin_prediction(
                    lm,
                    nlm,
                    nv,
                )  # r2=r2)
                pred_ccgp, pred_bind = pred_out[:2]
                ret_dict = {
                    "pred_ccgp": np.squeeze(pred_ccgp),
                    "pred_bind": np.squeeze(pred_bind),
                }
                predictions[region][contrast] = ret_dict
        return predictions

    def _direct_predictions(self, dec_key, decs=None, use_regions=None, time_acc=None):
        if decs is None:
            decs = self.data.get(dec_key)
        if use_regions is None:
            use_regions = self.params.getlist("use_regions")
        if time_acc is None:
            time_acc = self.params.getboolean("time_accumulate")
        model_dict = {}
        for region, dec_results in decs.items():
            if region in use_regions:
                model_dict[region] = {}
                for contrast, out in dec_results.items():
                    _, _, p1, p2, p3, p4, _ = out
                    if p1.shape[1] > 0:
                        out = mrdt.direct_ccgp_bind_est_pops(
                            (p1, p2),
                            (p3, p4),
                            test_prop=0,
                            empirical=False,
                            time_accumulate=time_acc,
                        )
                        out_dict = {
                            "pred_ccgp": 1 - out[1],
                            "pred_bin": 1 - out[0],
                            "d_l": out[2][0],
                            "d_n": out[2][1],
                            "sigma": out[2][2],
                            "sem": out[2][3],
                            "n_neurs": out[2][4],
                        }
                    else:
                        out_dict = None
                    model_dict[region][contrast] = out_dict
        return model_dict

    def panel_rwd_direct_predictions(self, force_refit=False):
        key = "rwd_direct_predictions"
        save_key = "panel_rwd_direct_predictions"
        dec_key = "rwd_generalization"
        if self.data.get(key) is None or force_refit:
            self.data[key] = self._direct_predictions(dec_key)
        if self.data.get(save_key) is None:
            self.make_dec_save_dicts(keys=(key,), loss=False, keep_subset=False)
        return self.data[key]

    def panel_prob_direct_predictions(self, force_refit=False):
        key = "prob_direct_predictions"
        save_key = "panel_prob_direct_predictions"
        dec_key = "prob_generalization"
        if self.data.get(key) is None or force_refit:
            self.data[key] = self._direct_predictions(dec_key)
        if self.data.get(save_key) is None:
            self.make_dec_save_dicts(keys=(key,), loss=False, keep_subset=False)
        return self.data[key]

    def panel_ev_direct_predictions(self, force_refit=False):
        key = "ev_direct_predictions"
        save_key = "panel_ev_direct_predictions"
        dec_key = "ev_generalization"
        if self.data.get(key) is None or force_refit:
            self.data[key] = self._direct_predictions(dec_key)
        if self.data.get(save_key) is None or force_refit:
            self.make_dec_save_dicts(keys=(key,), loss=False, keep_subset=False)
        return self.data[key]

    def panel_ev_other_direct_predictions(self, force_refit=False):
        key = "ev_other_direct_predictions"
        save_key = "panel_ev_other_direct_predictions"
        dec_key = "ev_generalization_other"
        if self.data.get(key) is None or force_refit:
            self.data[key] = self._direct_predictions(dec_key)
        if self.data.get(save_key) is None or force_refit:
            self.make_dec_save_dicts(keys=(key,), loss=False, keep_subset=False)
        return self.data[key]

    def panel_prob_model_prediction(self, force_refit=False, force_recompute=False):
        key = "panel_prob_model_prediction"
        term_key = "prob_model_terms"
        dec_key = "prob_generalization"
        if force_refit or (
            self.data.get(key) is None and self.data.get(term_key) is None
        ):
            self._model_terms(term_key, dec_key, force_recompute=force_refit)
        if force_recompute or self.data.get(key) is None:
            self.data[key] = self._model_predictions(term_key)
        return self.data.get(key)

    def panel_rwd_model_prediction(self, force_refit=False, force_recompute=False):
        key = "panel_rwd_model_prediction"
        term_key = "rwd_model_terms"
        dec_key = "rwd_generalization"
        if force_refit or (
            self.data.get(key) is None and self.data.get(term_key) is None
        ):
            self._model_terms(term_key, dec_key, force_recompute=force_refit)
        if force_recompute or self.data.get(key) is None:
            self.data[key] = self._model_predictions(term_key)
        return self.data.get(key)

    def make_dec_save_dicts(
        self,
        keys=("prob_generalization", "rwd_generalization"),
        loss=True,
        prefix="panel",
        **kwargs
    ):
        for k in keys:
            reg_dict = {}
            if loss:
                reform_func = mra.reformat_generalization_loss
                new_key = prefix + "_loss_" + k
            else:
                reform_func = mra.reformat_dict
                new_key = prefix + "_" + k
            for region, data in self.data[k].items():
                reg_dict[region] = reform_func(self.data[k][region], **kwargs)
            self.data[new_key] = reg_dict

    def prob_generalization(self, force_reload=False, force_recompute=False):
        key = "prob_generalization"
        dead_perc = self.params.getfloat("exclude_middle_percentiles")
        def mask_func(x): return x < 1
        min_trials = self.params.getint("min_trials_prob")
        if self.data.get(key) is None or force_recompute:
            out = self._decoding_analysis(
                "prob",
                mra.compute_all_generalizations,
                force_reload=force_reload,
                dead_perc=dead_perc,
                mask_func=mask_func,
                min_trials=min_trials,
            )
            self.data[key] = out
        return self.data[key]

    def rwd_generalization(self, force_reload=False, force_recompute=False):
        key = "rwd_generalization"
        dead_perc = None
        min_trials = self.params.getint("min_trials_rwd")
        if self.data.get(key) is None or force_recompute:
            out = self._decoding_analysis(
                "rwd",
                mra.compute_all_generalizations,
                force_reload=force_reload,
                dead_perc=dead_perc,
                min_trials=min_trials,
            )
            self.data[key] = out
        return self.data[key]

    def ev_generalization(
        self, force_reload=False, force_recompute=False, use_subj=True
    ):
        data_field = "ev"
        if use_subj:
            data_field = "subj_" + data_field
        key = "ev_generalization"
        dead_perc = self.params.getfloat("exclude_middle_percentiles")
        min_trials = self.params.getint("min_trials_ev")
        exclude_safe = self.params.getboolean("exclude_safe")
        if exclude_safe:
            def mask_func(x): return x < 1
            mask_var = "prob"
        else:
            mask_func = None
            mask_var = None
        use_split_dec = None
        dec_less = self.params.getboolean("dec_less")
        if self.data.get(key) is None or force_recompute:
            out = self._decoding_analysis(
                data_field,
                mra.compute_all_generalizations,
                force_reload=force_reload,
                dead_perc=dead_perc,
                min_trials=min_trials,
                use_split_dec=use_split_dec,
                mask_func=mask_func,
                mask_var=mask_var,
                dec_less=dec_less,
            )
            self.data[key] = out
        return self.data[key]

    def ev_generalization_other(
        self, force_reload=False, force_recompute=False, use_subj=True
    ):
        data_field = "ev"
        if use_subj:
            data_field = "subj_" + data_field
        key = "ev_generalization_other"
        dead_perc = self.params.getfloat("exclude_middle_percentiles")
        min_trials = self.params.getint("min_trials_ev_other")
        dec_less = self.params.getboolean("dec_less")
        use_split_dec = None
        if self.data.get(key) is None or force_recompute:
            out = self._decoding_analysis(
                data_field,
                mra.compute_conditional_generalization,
                force_reload=force_reload,
                dead_perc=dead_perc,
                min_trials=min_trials,
                use_split_dec=use_split_dec,
                dec_less=dec_less,
            )
            self.data[key] = out
        return self.data[key]

    def _generic_dec_panel(self, key, func, *args, dec_key=None, loss=False, **kwargs):
        func(*args, **kwargs)
        key_i = key.split("_", 1)[1]
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
        dec_gen_key = "prob_generalization"
        dec_gen_theory_key = "prob_direct_predictions"
        return self._plot_gen_results(dec_gen_key, dec_gen_theory_key)

    def plot_all_rwd_generalization(self):
        dec_gen_key = "rwd_generalization"
        dec_gen_theory_key = "rwd_direct_predictions"
        return self._plot_gen_results(dec_gen_key, dec_gen_theory_key)

    def plot_all_ev_generalization(self):
        dec_gen_key = "ev_generalization"
        dec_gen_theory_key = "ev_direct_predictions"
        return self._plot_gen_results(dec_gen_key, dec_gen_theory_key)

    def plot_all_ev_generalization_other(self):
        dec_gen_key = "ev_generalization_other"
        dec_gen_theory_key = "ev_other_direct_predictions"
        return self._plot_gen_results(dec_gen_key, dec_gen_theory_key)

    def panel_prob_generalization(self, *args, **kwargs):
        key = "panel_prob_generalization"
        return self._generic_dec_panel(key, self.prob_generalization, *args, **kwargs)

    def panel_rwd_generalization(self, *args, **kwargs):
        key = "panel_rwd_generalization"
        return self._generic_dec_panel(key, self.rwd_generalization, *args, **kwargs)

    def panel_ev_generalization(self, *args, **kwargs):
        key = "panel_ev_generalization"
        return self._generic_dec_panel(key, self.ev_generalization, *args, **kwargs)

    def panel_ev_generalization_other(self, *args, **kwargs):
        key = "panel_ev_generalization_other"
        return self._generic_dec_panel(
            key, self.ev_generalization_other, *args, **kwargs
        )

    def panel_loss_prob_generalization(self, *args, **kwargs):
        key = "panel_loss_prob_generalization"
        dec_key = "prob_generalization"
        if self.data.get(dec_key) is None:
            self.panel_prob_generalization(*args, **kwargs)
        self.make_dec_save_dicts(keys=(dec_key,), loss=True)
        return self.data[key]

    def panel_loss_rwd_generalization(self, *args, **kwargs):
        key = "panel_loss_rwd_generalization"
        dec_key = "rwd_generalization"
        if self.data.get(dec_key) is None:
            self.panel_rwd_generalization(*args, **kwargs)
        self.make_dec_save_dicts(keys=(dec_key,), loss=True)
        return self.data[key]

    def panel_loss_ev_generalization(self, *args, **kwargs):
        key = "panel_loss_ev_generalization"
        dec_key = "ev_generalization"
        if self.data.get(dec_key) is None:
            self.panel_ev_generalization(*args, **kwargs)
        self.make_dec_save_dicts(keys=(dec_key,), loss=True)
        return self.data[key]


def _get_nonlinear_columns(coeffs):
    return _get_template_columns(coeffs, ".*\(nonlinear\)")


def _get_linear_columns(coeffs):
    return _get_template_columns(coeffs, ".*\(linear\)")


def _get_template_columns(coeffs, template, columns=None):
    mask = np.array(list(re.match(template, col) is not None for col in columns))
    return np.array(coeffs)[:, mask]


def _get_rwd_lin(coeffs):
    return _get_template_columns(coeffs, "reward \(linear\)")


def _get_prob_lin(coeffs):
    return _get_template_columns(coeffs, "prob \(linear\)")


def _get_rwd_nonlin(coeffs):
    return _get_template_columns(coeffs, "rwd:.* \(nonlinear\)")


def _get_prob_nonlin(coeffs):
    return _get_template_columns(coeffs, "prob:.* \(nonlinear\)")


def _get_rwd_lin_boot(coeffs, columns):
    return _get_template_columns(coeffs, "reward", columns=columns)


def _get_prob_lin_boot(coeffs, columns):
    return _get_template_columns(coeffs, "probability", columns=columns)


def _get_rwd_nonlin_boot(coeffs, columns):
    return _get_template_columns(coeffs, "rwd x.*", columns=columns)


def _get_prob_nonlin_boot(coeffs, columns):
    return _get_template_columns(coeffs, "prob x.*", columns=columns)


def _get_nv_boot(coeffs, columns):
    return _get_template_columns(coeffs, "MSE_TRAIN", columns=columns)


class SIDecodingFigure(MultipleRepFigure):
    def __init__(self, fig_key="si_decoding", colors=colors, **kwargs):
        fsize = (4, 6)
        cf = u.ConfigParserColor()
        cf.read(config_path)

        params = cf[fig_key]
        self.fig_key = fig_key
        super().__init__(fsize, params, colors=colors, **kwargs)

    def make_gss(self):
        gss = {}

        gs_contrasts = pu.make_mxn_gridspec(self.gs, 2, 1, 0, 100, 0, 100, 5, 5)
        axs_contrasts = self.get_axs(
            gs_contrasts, sharey=True, sharex=True, squeeze=True
        )

        gss["panel_dec_perf"] = axs_contrasts
        self.gss = gss

    def panel_dec_perf(self):
        key = "panel_dec_perf"
        axs = self.gss[key]

        preds, decs = self._get_gen_emp_pred()
        regions = self.params.getlist("use_regions")

        use_contrasts = self.params.getlist("use_contrasts")
        if use_contrasts is None:
            contrasts = preds[regions[0]].dtype.names
        else:
            contrasts = use_contrasts
        cnames = self.params.getlist("contrast_names")
        if cnames is None:
            cnames = contrasts
        for i, region in enumerate(regions):
            for j, contrast in enumerate(contrasts):
                pred_ij = preds[region][contrast]
                dec_ij = decs[region][contrast]
                color = self.get_color_dict()[region]
                try:
                    mrv.plot_dec_gen_pred(
                        pred_ij[0, 0], dec_ij[0, 0], i, ax=axs[j], color=color
                    )
                except IndexError:
                    s = "did not plot {} for {} (likely a nan)"
                    print(s.format(region, contrast))
                axs[j].set_title(cnames[j])

        axs[-1].set_xticks(range(len(regions)))
        axs[-1].set_xticklabels(regions)
        gpl.clean_plot(axs[0], 0)
        gpl.clean_plot_bottom(axs[0])
        gpl.clean_plot(axs[1], 0)
        axs[1].set_ylabel("decoding performance")
        axs[0].set_ylabel("decoding performance")
        gpl.add_hlines(0.5, axs[0])
        gpl.add_hlines(0.5, axs[1])


class OfferDecFigure(MultipleRepFigure):
    def __init__(self, fig_key="offer_dec_figure", colors=colors, **kwargs):
        fsize = (4, 3)
        cf = u.ConfigParserColor()
        cf.read(config_path)

        params = cf[fig_key]
        self.fig_key = fig_key
        self.saved_coefficients = None
        super().__init__(fsize, params, colors=colors, **kwargs)

    def make_gss(self):
        gss = {}

        dec_grid = pu.make_mxn_gridspec(self.gs, 2, 1,
                                        0, 100, 0, 80,
                                        2, 10)
        dec_axs = self.get_axs(
            dec_grid, sharey='all', sharex='all', squeeze=True,
        )
        decs_all_grid = pu.make_mxn_gridspec(self.gs, 2, 1,
                                             0, 100,
                                             90, 100,
                                             2, 10)
        dec_all_axs = self.get_axs(
            decs_all_grid, sharey='all', sharex='all', squeeze=True,
        )

        gss['panel_dec'] = (dec_axs, dec_all_axs)

    def panel_dec(self, force_reload=False, recompute=False):
        key = 'panel_dec'
        axs_regions, axs_all = self.gss[key]

        tbeg = self.params.getint("tbeg")
        tend = self.params.getint("tend")
        winsize = self.params.getint("winsize")
        winstep = self.params.getint("winstep")
        prs = self.params.getint("resamples")
        pre_pca = self.params.getfloat("pca_pre")
        regions = self.params.getlist("use_regions")
        min_trials = self.params.getint("min_trials_dec")

        use_pairs = ('offer 1', 'offer 2')
        if self.data.get(key) is None or recompute:
            exper_data = self.get_experimental_data(force_reload=force_reload)
            out_dec = mra.estimate_decoding_regions(
                exper_data,
                tbeg,
                tend,
                winsize=winsize,
                tstep=winstep,
                time_accumulate=True,
                pop_resamples=prs,
                pre_pca=pre_pca,
                min_trials=min_trials,
                region_list=regions,
                only_pairs=use_pairs,
            )
            out_dec_theor = mra.make_prediction_pops(out_dec)
            self.data[key] = out_dec, out_dec_theor

        out_dec, out_dec_theor = self.data[key]
        minor_tick = .1
        for i, pair in enumerate(use_pairs):
            decs = []
            gens = []
            colors = []
            for r in regions[:-1]:
                decs.append(
                    np.mean(list(p[0] for p in out_dec[r][pair]), axis=(0, 2))[..., -1]
                )
                gens.append(
                    np.mean(list(p[-1] for p in out_dec[r][pair]), axis=(0, 2))[..., -1]
                )
                colors.append(self.get_color_dict()[r])

            gpl.clean_plot(axs_regions[i], 0)
            kl = i == len(use_pairs) - 1
            gpl.clean_plot_bottom(axs_regions[i], keeplabels=kl)
            ticks = np.arange(len(regions))
            gpl.violinplot(decs, ticks - minor_tick,
                           color=colors,
                           ax=axs_regions[i])
            gpl.add_hlines(.5, axs_regions[i])
            axs_regions[i].set_ylabel('decoding\nperformance')
        axs_regions[-1].set_xticks(ticks)
        axs_regions[-1].set_xticklabels(regions)

        for i, pair in enumerate(use_pairs):
            dec = np.mean(list(p[0] for p in out_dec["all"][pair]),
                          axis=(0, 2))[..., -1]

            gen = np.mean(list(p[-1] for p in out_dec["all"][pair]),
                          axis=(0, 2))[..., -1]
            color = self.get_color_dict()["all"]
            # print(dec.shape)
            # print(out_dec[pair][0][0].shape)
            # nl = np.concatenate((out_dec_theor[pair][0][0],
            #                      out_dec_theor[pair][1][0]))

            gpl.clean_plot(axs_all[i], 0)
            kl = i == len(use_pairs) - 1
            gpl.clean_plot_bottom(axs_all[i], keeplabels=kl)
            gpl.violinplot([dec], [0 - minor_tick],
                           color=[color],
                           ax=axs_all[i])
            gpl.add_hlines(.5, axs_all[i])
        axs_all[-1].set_xticks([0])
        axs_all[-1].set_xticklabels(["all"])
        yl0 = axs_all[-1].get_ylim()[0]
        yl1 = axs_regions[-1].get_ylim()[1]

        axs_regions[0].set_ylim((yl0, yl1))

class SubsampledNeuronsFigure(MultipleRepFigure):
    def __init__(self, fig_key="subsample_figure", colors=colors, **kwargs):
        fsize = (6, 2)
        cf = u.ConfigParserColor()
        cf.read(config_path)

        params = cf[fig_key]
        self.fig_key = fig_key
        self.saved_coefficients = None
        super().__init__(fsize, params, colors=colors, **kwargs)

    def make_gss(self):
        gss = {}

        dist_grids = pu.make_mxn_gridspec(
            self.gs, 1, 2, 0, 100, 0, 100, 5, 10)
        dist_axs = self.get_axs(
            dist_grids, sharey='all', sharex='all', squeeze=True,
        )
        gss['panel_dists'] = dist_axs
        self.gss = gss

    def panel_dists(self):
        key = "panel_dists"
        axs = self.gss[key]

        ri = self.params.get("runind")
        l_color = self.params.getcolor("lin_color")
        n_color = self.params.getcolor("conj_color")
        regions = self.params.getlist("use_regions")
        
        comb_dict = mraux.load_decoding_runs(ri)
        mrv.plot_pred_dict(
            comb_dict["predictions"],
            axs=axs,
            clean_bottom=False,
            l_color=l_color,
            n_color=n_color,
            regions=regions,
        )
        axs[0].set_ylabel("estimated distance")
        axs[0].set_title("space")
        axs[1].set_title("time")
        
class TrialNeuronTradeoff(MultipleRepFigure):
    def __init__(self, fig_key="trial_neuron_figure", colors=colors, **kwargs):
        fsize = (6, 4)
        cf = u.ConfigParserColor()
        cf.read(config_path)

        params = cf[fig_key]
        self.fig_key = fig_key
        self.saved_coefficients = None
        super().__init__(fsize, params, colors=colors, **kwargs)

    def make_gss(self):
        gss = {}

        n_regions = len(self.params.getlist("use_regions"))
        trade_grids = pu.make_mxn_gridspec(
            self.gs, 2, int(n_regions/2), 0, 100, 0, 100, 15, 10)
        trade_axs = self.get_axs(
            trade_grids, sharex='all', squeeze=True,
        )
        gss['panel_tradeoffs'] = trade_axs.flatten()
        self.gss = gss

    def panel_tradeoffs(self):
        key = "panel_tradeoffs"
        axs = self.gss[key]

        min_line = self.params.getint("required_trials_line")
        regions = self.params.getlist("use_regions")
        if self.data.get(key) is None:
            data = self.get_exper_data()
            out_dict = {}
            for region in regions:
                if region == "all":
                    r = None
                else:
                    r = (region,)
                out = mra.compute_all_generalization_n_neurs(
                    data, 0, 500, "subj_ev", regions=r,
                )
                out_dict[region] = out
            self.data[key] = out_dict

        out_dict = self.data[key]
        cond_key = ('subj_ev_left','subj_ev_right')
        for i, region in enumerate(regions):
            thr, n_neurs = out_dict[region][cond_key]
            axs[i].plot(thr, n_neurs, color=self.get_color_dict()[region])
            gpl.add_vlines(min_line, axs[i])
            gpl.clean_plot(axs[i], 0)
            axs[i].set_xlabel("required trials")
            axs[i].set_ylabel("available neurons")
    

class TemporalChangeDecoding(MultipleRepFigure):
    def __init__(self, fig_key="temporal_change_figure", colors=colors, **kwargs):
        fsize = (6, 3)
        cf = u.ConfigParserColor()
        cf.read(config_path)

        params = cf[fig_key]
        self.fig_key = fig_key
        self.saved_coefficients = None
        super().__init__(fsize, params, colors=colors, **kwargs)

    def make_gss(self):
        gss = {}

        dec_tc_grid = pu.make_mxn_gridspec(
            self.gs, 1, 1, 0, 100, 0, 45, 5, 5)
        dec_tc_ax = self.get_axs(
            dec_tc_grid, sharey='all', sharex='all',
        )[0, 0]
        gss['panel_dec_tc'] = dec_tc_ax

        dec_region_grid = pu.make_mxn_gridspec(
            self.gs, 1, 1, 0, 100, 55, 100, 5, 5)
        dec_region_ax = self.get_axs(
            dec_region_grid, sharey='all', sharex='all',
        )[0, 0]
        gss['panel_dec_region'] = dec_region_ax
        
        self.gss = gss


    def temporal_generalization(
            self,
            tbeg,
            tend,
            winsize,
            winstep,
            var="subj_ev",
            time_accumulate=False,
            train_trl_perc=None,
            **kwargs,
    ):
        data = self.get_exper_data()
        prs = self.params.getint("resamples")
        pca_pre = self.params.getfloat("pca_pre")
        regions = self.params.getlist("use_regions")
        correct_only = self.params.getboolean("correct_only")
        subsample_neurons = self.params.getint("subsample_neurons")
        dead_perc = self.params.getfloat("exclude_middle_percentiles")
        min_trials = self.params.getint("min_trials")
        exclude_safe = self.params.getboolean("exclude_safe")
        if train_trl_perc is None:
            train_trl_perc = self.params.getfloat("train_trial_perc")
        if exclude_safe:
            def mask_func(x): return x < 1
            mask_var = "prob"
        else:
            mask_func = None
            mask_var = None
        use_split_dec = None
        dec_less = self.params.getboolean("dec_less")
        
        out_dict = {}
        for region in regions:
            if region == "all":
                use_regions = None
            else:
                use_regions = (region,)
            out_r = mra.compute_temporal_generalization(
                data,
                tbeg,
                tend,
                var,
                train_trl_perc=train_trl_perc,
                pop_resamples=prs,
                winsize=winsize,
                pre_pca=pca_pre,
                tstep=winstep,
                time_accumulate=time_accumulate,
                regions=use_regions,
                correct_only=correct_only,
                subsample_neurons=subsample_neurons,
                dead_perc=dead_perc,
                min_trials=min_trials,
                use_split_dec=use_split_dec,
                dec_less=dec_less,
                mask_var=mask_var,
                mask_func=mask_func,
                **kwargs,
            )
            out_dict[region] = out_r
        return out_dict
        
    def panel_dec_tc(self, recompute=False, **kwargs):
        key = "panel_dec_tc"
        ax = self.gss[key]

        if self.data.get(key) is None or recompute:
            tbeg = self.params.getfloat("tc_tbeg")
            tend = self.params.getfloat("tc_tend")
            winsize = self.params.getfloat("tc_winsize")
            winstep = self.params.getfloat("tc_winstep")
            out = self.temporal_generalization(
                tbeg, tend, winsize, winstep, time_accumulate=False, **kwargs,
            )
            self.data[key] = out
        out = self.data[key]

        region = self.params.get("tc_region")
        color = self.get_color_dict()[region]
        mrv.plot_temporal_tc_dict(out[region], ax=ax, color=color)

    def panel_dec_region(self, recompute=False, **kwargs,):
        key = "panel_dec_region"
        ax = self.gss[key]

        if self.data.get(key) is None or recompute:
            tbeg = self.params.getfloat("st_tbeg")
            tend = self.params.getfloat("st_tend")
            winsize = self.params.getfloat("st_winsize")
            winstep = self.params.getfloat("st_winstep")
            out = self.temporal_generalization(
                tbeg, tend, winsize, winstep, time_accumulate=True, **kwargs,
            )
            self.data[key] = out
        out = self.data[key]

        region_list = self.params.getlist("use_regions")
        mrv.plot_temporal_region_dict(
            out, color_dict=self.get_color_dict(), ax=ax, region_list=region_list,
        )
        gpl.clean_plot(ax, 0)            


class BehavioralConsistency(MultipleRepFigure):
    def __init__(self, fig_key="consistency_figure", colors=colors, **kwargs):
        fsize = (6, 6)
        cf = u.ConfigParserColor()
        cf.read(config_path)

        params = cf[fig_key]
        self.fig_key = fig_key
        self.saved_coefficients = None
        super().__init__(fsize, params, colors=colors, **kwargs)

    def make_gss(self):
        gss = {}
        n_rows = len(self.get_color_dict()) - 1
        n_cols = np.max(list(len(v) for k, v in mraux.region_monkey_dict.items()
                             if k != "all"))
        bhv_gs = pu.make_mxn_gridspec(self.gs, n_rows, n_cols, 0, 100, 0, 100, 2, 5)
        bhv_ax = self.get_axs(bhv_gs, sharey="all", sharex="all")
        gss['panel_consistency'] = bhv_ax
        self.gss = gss

    def panel_consistency(self):
        key = "panel_consistency"
        axs = self.gss[key]
        
        data = self.get_exper_data()
        if self.data.get(key) is None:
            corrs = data["subj_ev_chosen"] - data["subj_ev_unchosen"] > 0
            trl_nums = data["trial_num"]
            self.data[key] = (corrs, trl_nums)
        corrs, trl_nums = self.data[key]

        filter_len = self.params.getint("filter_len")
        perf_thr = self.params.getfloat("performance_thr")
        use_regions = self.params.getlist("use_regions")

        for i, trl in enumerate(trl_nums):
            corr = corrs[i]
            region = data["neur_regions"][i].iloc[0][0]
            monkey = data["animal"][i]
            j = mraux.region_monkey_dict[region].index(monkey)
            k = use_regions.index(region)
            filt = np.ones(filter_len)/filter_len
            corr_filt = np.convolve(corr, filt, mode="valid")
            trl_filt = np.convolve(trl, filt, mode="valid")
            if np.any(corr_filt < perf_thr):
                color = "k"
            else:
                color = self.get_color_dict()[region]
            if np.all(corr_filt == 0):
                corr_filt[:] = np.nan
            axs[k, j].plot(
                trl_filt,
                corr_filt,
                color=color,
                lw=.5,
            )
            axs[k, j].set_title(monkey[:1])

        for i, j in u.make_array_ind_iterator(axs.shape):
            gpl.add_hlines(.5, axs[i, j])
            gpl.clean_plot(axs[i, j], j)
            if i == len(axs) - 1:
                axs[i, j].set_xlabel("trial number")
            else:
                gpl.clean_plot_bottom(axs[i, j])
            if j == 0:
                axs[i, j].set_ylabel("proportion of trials\nwith optimal choice")


class CombinedRepFigure(MultipleRepFigure):
    def __init__(self, fig_key="combined_rep_figure", colors=colors, **kwargs):
        fsize = (8, 6)
        cf = u.ConfigParserColor()
        cf.read(config_path)

        params = cf[fig_key]
        self.fig_key = fig_key
        self.saved_coefficients = None
        super().__init__(fsize, params, colors=colors, **kwargs)

    def make_gss(self):
        gss = {}

        top = 40
        gap = 10

        all_split = 42
        all_gap = 3
        end = all_split + 8
        vis_grid = self.gs[:top, :30]
        vis_ax = self.get_axs(
            (vis_grid,), sharey='all', sharex='all', plot_3ds=np.array(((True,),)),
        )
        gss['panel_vis'] = vis_ax[0]

        sub_corr_grid = pu.make_mxn_gridspec(self.gs, 1, 1,
                                             0, top - 10, 40, 75,
                                             2, 10)
        sub_corr_axs = self.get_axs(
            sub_corr_grid, sharey='all', sharex='all',
        )
        gss['subspace_corr'] = sub_corr_axs[0, 0]

        ds_regions_gs = pu.make_mxn_gridspec(self.gs, 2, 1,
                                             top + gap, 100,
                                             0, all_split, 5, 10)
        ds_regions_axs = self.get_axs(ds_regions_gs, squeeze=False,
                                      sharey="vertical")
        ds_all_gs = pu.make_mxn_gridspec(self.gs, 2, 1,
                                         top + gap, 100,
                                         all_split + all_gap, end,
                                         5, 10)
        ds_all_axs = self.get_axs(ds_all_gs, squeeze=False,
                                  sharey="vertical")

        dist_mat_ax = self.get_axs((self.gs[0:top - 10, 80:],),)

        gss['panel_dists'] = (
            dist_mat_ax,
            ds_regions_axs,
            ds_all_axs,
        )

        choice_bhv_gs = pu.make_mxn_gridspec(self.gs, 1, 2,
                                             80 + 3, 100,
                                             65, 100, 2, 10)
        choice_bhv_axs = self.get_axs(choice_bhv_gs)
        gss['panel_choice_bhv'] = choice_bhv_axs

        dec_bhv_gs = pu.make_mxn_gridspec(self.gs, 1, 2,
                                          top + gap, 80 - gap,
                                          65, 100, 2, 10)
        dec_bhv_axs = self.get_axs(dec_bhv_gs, sharex="all", sharey="all")
        gss['panel_dec_bhv'] = dec_bhv_axs
        self.gss = gss

    # combined rep fig
    def panel_subspace_corr(self, recompute=False):
        key = "subspace_corr"
        ax = self.gss[key]
        data = self.get_exper_data()
        tb = self.params.getint("tbeg")
        te = self.params.getint("tend")
        regions = self.params.getlist('use_regions')

        use_nl_val = self.params.getboolean("use_nl_value")
        if use_nl_val:
            align_func = mra.compute_alignment_index
            pred_func = mra.interaction_spline_pred
        else:
            align_func = mra.compute_corr
            pred_func = mra.interaction_pred

        if self.data.get(key) is None or recompute:
            out_sessions = mra.make_predictor_matrices(
                data,
                t_beg=tb,
                t_end=te,
                transform_value=use_nl_val,
                single_time=True,
            )
            boots_full = mra.fit_bootstrap_models(out_sessions[0])

            out_sessions = mra.make_predictor_matrices(
                data,
                t_beg=tb,
                t_end=te,
                transform_value=use_nl_val,
                t1_only=True,
            )
            boots_t1 = mra.fit_bootstrap_models(out_sessions[0])

            self.data[key] = (boots_full, boots_t1)

        boots_full, boots_t1 = self.data[key]
        null_color = self.params.getcolor("null_corr_color")
        for i, r in enumerate(regions):
            r_color = self.get_color_dict()[r]
            if r == "all":
                r_list = None
            else:
                r_list = (r,)
            mrv.plot_split_halfs_full(
                boots_full,
                pred_func,
                use_regions=r_list,
                ax=ax,
                wi_pt=i,
                ac_pt=i,
                wi_d_pt=i,
                wi_color=r_color,
                ac_color=null_color,
                align_func=align_func,
                zp1='right',
                zp2='left',
            )
        gpl.clean_plot(ax, 0)
        gpl.clean_plot_bottom(ax, keeplabels=True)
        gpl.add_hlines(0, ax)
        ax.set_xticks(np.arange(len(regions)))
        ax.set_xticklabels(regions)

        if use_nl_val:
            y_label = "alignment index"
        else:
            y_label = "subspace correlation"
        ax.set_ylabel(y_label)
        
    def panel_vis(self, force_reload=False):
        key = 'panel_vis'
        ax = self.gss[key][0]
        prs = self.params.getint("resamples")
        tbeg_vis = self.params.getint("tbeg_vis")
        tend_vis = self.params.getint("tend_vis")
        min_trials = self.params.getint("min_trials_conds")
        if self.data.get(key) is None:
            exper_data = self.get_experimental_data(force_reload=force_reload)        
            out_sub = mra.estimate_submanifolds(
                exper_data,
                tbeg_vis,
                tend_vis,
                time_accumulate=True,
                pop_resamples=prs,
                min_trials=min_trials,
            )
            self.data[key] = out_sub

        out_sub = self.data[key]
        samp_ind = 25
        po2 = out_sub[0][samp_ind][:4]
        po1 = out_sub[0][samp_ind][4:8]
        po1e = out_sub[0][samp_ind][8:]

        # po2_color = self.params.getcolor('offer 2 color')
        # po1_color = self.params.getcolor('offer 1 color')
        # po1e_color = self.params.getcolor('original offer 1 color')

        rv_color = self.params.getcolor('right_value_color')
        lv_color = self.params.getcolor('left_value_color')
        rv_o1_color = self.params.getcolor('right_value_color_o1')
        lv_o1_color = self.params.getcolor('left_value_color_o1')
        space_color = self.params.getcolor('lin_color')
        style_same = self.params.get("timing_line_style")

        colors = ((rv_color, lv_color),
                  (rv_color, lv_color),
                  (rv_o1_color, lv_o1_color))
        styles = (None, style_same, None)
        mrv.plot_submanifolds(
            po2,
            po1e,
            po1,
            colors=colors,
            linestyles=styles,
            space_color=space_color,
            ax=ax,
        )

    def panel_dists(self, force_reload=False, recompute=False):
        key = 'panel_dists'
        ax_vis, ax_rs, ax_alls = self.gss[key]

        tbeg = self.params.getint("tbeg")
        tend = self.params.getint("tend")
        winsize = self.params.getint("winsize")
        winstep = self.params.getint("winstep")
        prs = self.params.getint("resamples")
        pca_pre = self.params.getfloat("pca_pre")
        regions = self.params.getlist("use_regions_rdm")
        min_trials = self.params.getint("min_trials_conds")
        exclude_safe = self.params.getboolean("exclude_safe")
        normalize_embedding_power = self.params.getboolean("normalize_embedding_power")
        dead_perc = self.params.getfloat("exclude_middle_percentiles")

        if self.data.get(key) is None or recompute:
            exper_data = self.get_experimental_data(force_reload=force_reload)
            if exclude_safe:
                def mask_func(x): return x < 1
                mask_var = "prob"
            else:
                mask_func = None
                mask_var = None

            out_rdm_dict = mra.estimate_rdm_regions(
                exper_data,
                tbeg,
                tend,
                dead_perc=dead_perc,
                winsize=winsize,
                winstep=winstep,
                time_accumulate=True,
                pop_resamples=prs,
                min_trials=min_trials,
                pca_pre=pca_pre,
                region_list=regions,
                mask_var=mask_var,
                mask_func=mask_func,
            )

            self.data[key] = out_rdm_dict
        out_rdm_dict = self.data[key]

        if normalize_embedding_power:
            out_rdm_dict = mra.normalize_embedding_power(out_rdm_dict)

        use_pairs_mb = (
            "order flips, side-values same",  # temporal misbinding
            "order flips, side-values flip",  # spatial misbinding
            "same order, side-values flip",  # temporal and spatial misbinding
        )
        use_labels_mb = (
            "temporal misbinding",
            "spatial misbinding",
            "spatial-temporal misbinding",
        )
        conf1_color = self.params.getcolor("temp_spatial_color")
        conf2_color = self.params.getcolor("spatial_color")
        conf3_color = self.params.getcolor("temporal_color")
        colors_mb = (conf3_color, conf2_color, conf1_color,)

        mra.print_rdm_factorial_analysis(out_rdm_dict)
        
        ax_r_mb, ax_a_mb = ax_rs[0, 0], ax_alls[0, 0]
        mrv.plot_dists_region(out_rdm_dict, use_pairs_mb, regions, colors_mb,
                              axs=(ax_r_mb, ax_a_mb),
                              labels=use_labels_mb)

        use_pairs_ud = (
            "low",
            "high",
        )
        use_labels_ud = (
            "low-value misbinding",
            "high-value misbinding",
        )
        ll_color = self.params.getcolor("low_color")
        hh_color = self.params.getcolor("high_color")
        colors_ud = (ll_color, hh_color,)
        ax_r_ud, ax_a_ud = ax_rs[1, 0], ax_alls[1, 0]
        mrv.plot_dists_region(out_rdm_dict, use_pairs_ud, regions, colors_ud,
                              axs=(ax_r_ud, ax_a_ud),
                              labels=use_labels_ud)

        highlight_colors = {
            "same order, side-values flip": conf1_color,
            "order flips, side-values flip": conf2_color,
            "order flips, side-values same": conf3_color,
            "high": hh_color,
            "low": ll_color,
        }
        highlights = {k: mra.default_ambiguities[k] for k in highlight_colors.keys()}
        use_rdm = np.mean(out_rdm_dict["all"][0], axis=0)
        cond_labels = (
            r"-L $\rightarrow$ -R",
            r"-L $\rightarrow$ +R",
            r"+L $\rightarrow$ -R",
            r"+L $\rightarrow$ +R",
            r"-R $\rightarrow$ -L",
            r"+R $\rightarrow$ -L",
            r"-R $\rightarrow$ +L",
            r"+R $\rightarrow$ +L",
        )
        cmap = plt.get_cmap('Greys')
        mrv.plot_dist_mat(use_rdm, labels=cond_labels, ax=ax_vis[0, 0], cmap=cmap,
                          highlights=highlights, highlight_colors=highlight_colors,
                          fig=self.f)
        ax_vis[0, 0].set_aspect('equal')

    def panel_choice_bhv(self):
        key = "panel_choice_bhv"
        ax_psy, ax_rel = self.gss[key][0]

        data = self.get_exper_data()
        m_color = self.params.getcolor("mixed_color")
        l_color = self.params.getcolor("low_color")
        h_color = self.params.getcolor("high_color")
        colors = (m_color, l_color, h_color)
        dead_perc = self.params.getfloat("exclude_middle_percentiles")

        corr_dict, out_dict = mrv.plot_choice_sensitivity(
            data, dead_perc=dead_perc, ax=ax_psy, colors=colors,
        )
        self.data[key] = (corr_dict, out_dict)
        if self.data.get("panel_dists") is None:
            self.panel_dists()
        out_rdm_dict = self.data.get("panel_dists")
        corr = (
            (corr_dict["mixed high-low"],),
            (corr_dict["low only"],),
            (corr_dict["high only"],),
        )
        pairs = (("same order, side-values flip",
                  "order flips, side-values flip",
                  "order flips, side-values same",),
                 ("low",),
                 ("high",),)
        for i, pair in enumerate(pairs):
            comb_dists = np.mean(
                list(out_rdm_dict["all"][2][p] for p in pair),
                axis=0,
            )
            dists = np.expand_dims(comb_dists, 1)
            corr_mu = np.mean(corr[i])
            gpl.plot_trace_werr([corr_mu], dists, ax=ax_rel, conf95=True,
                                color=colors[i], fill=False, points=True)
        ax_rel.set_xlim([.65, 1])
        ax_rel.set_xlabel('correct response\nrate')
        ax_rel.set_ylabel('estimated distance')

    def additional_dec_bhv_sweep(self, force_reload=False, recompute=False):
        key = "additional_dec_bhv_sweep"
        min_neurs_range = np.arange(2, 20, 4)
        if self.data.get(key) is None or recompute:
            min_trials = self.params.getint("min_trials_bhv")
            tbeg = self.params.getint("tbeg")
            tend = self.params.getint("tend")
            dead_perc = self.params.getfloat("dead_perc_bhv")
            pre_pca = self.params.getfloat("pca_pre")
            n_folds = self.params.getint("n_folds_bhv")
            test_prop = self.params.getfloat("test_frac_bhv")

            data = self.get_experimental_data(force_reload=force_reload)
            out_dict = {}
            for i, mn in enumerate(min_neurs_range):
                session_mask = data['n_neurs'] > mn
                data_filt = data.session_mask(session_mask)

                out_decbhv = mra.estimate_bhv_corr(
                    data_filt,
                    tbeg,
                    tend,
                    min_trials=min_trials,
                    dead_perc=dead_perc,
                    time_accumulate=True,
                    pre_pca=pre_pca,
                    n_folds=n_folds,
                    test_prop=test_prop,
                    ret_projections=True,
                    use_time=True,
                )
                regions = list(x[0][0] for x in data_filt['neur_regions'])
                out_dict[mn] = (regions, out_decbhv)
            self.data[key] = out_dict
        ks = (
            'left-higher vs right-higher -- no order',
            'first-higher vs second-higher -- no side',
        )
        out_dict = self.data[key]
        regions_all = np.unique(
            np.concatenate(list(v[0] for v in out_dict.values()))
        )
        f, axs = plt.subplots(
            2,
            len(regions_all),
            figsize=(1*len(regions_all), 2),
            sharex="all",
            sharey="all",
        )

        r_list = self.params.getlist("use_regions")
        mrv.plot_bhv_dec_line(
            out_dict,
            ks,
            min_neurs_range,
            axs=axs,
            u_rs=regions_all,
            color_dict=self.get_color_dict(),
            region_list=r_list,
        )
        for i, j in u.make_array_ind_iterator(axs.shape):
            gpl.clean_plot(axs[i, j], j)
            gpl.add_hlines(0, axs[i, j])
        return f

    def panel_dec_bhv(self, force_reload=False, recompute=False):
        key = "panel_dec_bhv"
        ax_lr, ax_fs = self.gss[key][0]

        if self.data.get(key) is None or recompute:
            min_neurs = self.params.getint("min_neurs_bhv")
            min_trials = self.params.getint("min_trials_bhv")
            tbeg = self.params.getint("tbeg")
            tend = self.params.getint("tend")
            dead_perc = self.params.getfloat("dead_perc_bhv")
            pre_pca = self.params.getfloat("pca_pre")
            n_folds = self.params.getint("n_folds_bhv")
            test_prop = self.params.getfloat("test_frac_bhv")

            data = self.get_experimental_data(force_reload=force_reload)
            session_mask = data['n_neurs'] > min_neurs
            data_filt = data.session_mask(session_mask)

            out_decbhv = mra.estimate_bhv_corr(
                data_filt,
                tbeg,
                tend,
                min_trials=min_trials,
                dead_perc=dead_perc,
                time_accumulate=True,
                pre_pca=pre_pca,
                n_folds=n_folds,
                test_prop=test_prop,
                ret_projections=True,
                use_time=True,
            )
            self.data[key] = data_filt, out_decbhv
        data_filt, out_decbhv = self.data[key]

        k1 = 'left-higher vs right-higher -- no order'
        k2 = 'first-higher vs second-higher -- no side'
        
        dec, xs, d1, d2, g1, g2, gen = out_decbhv[k1]
        color_dict = self.get_color_dict()
        regions = list(x[0][0] for x in data_filt['neur_regions'])
        
        mra.print_factorial_analysis(regions, out_decbhv[k1], out_decbhv[k2])

        mrv.plot_bhv_dec(
            dec,
            gen,
            regions,
            dec_pops=(d1, d2),
            gen_pops=(g1, g2),
            color_dict=color_dict,
            ax=ax_lr,
            add_lines=False
        )

        dec, xs, d1, d2, g1, g2, gen = out_decbhv[k2]
        mrv.plot_bhv_dec(
            dec,
            gen,
            regions,
            dec_pops=(d1, d2),
            gen_pops=(g1, g2),
            color_dict=color_dict,
            ax=ax_fs,
            add_lines=False,
        )
        null_line = 0
        gpl.add_hlines(null_line, ax_lr)
        gpl.add_hlines(null_line, ax_fs)
        gpl.add_vlines(null_line, ax_lr)
        gpl.add_vlines(null_line, ax_fs)

        name_str = "decoder projection"
        gen_str = "generalization projection"
        ax_fs.set_xlabel('{}\n(optimal trials)'.format(name_str))
        ax_lr.set_xlabel('{}\n(optimal trials)'.format(name_str))
        ax_lr.set_ylabel('{}\n(non-optimal trials)'.format(gen_str))


class GeneralTheoryFigure(MultipleRepFigure):
    def __init__(self, fig_key="general_theory_figure", colors=colors, **kwargs):
        fsize = (6, 6)
        cf = u.ConfigParserColor()
        cf.read(config_path)

        params = cf[fig_key]
        self.fig_key = fig_key
        self.saved_coefficients = None
        super().__init__(fsize, params, colors=colors, **kwargs)

    def make_gss(self):
        gss = {}

        snr_grid = pu.make_mxn_gridspec(self.gs, 1, 3, 0, 20, 0, 100, 10, 15)
        axs_snr = self.get_axs(snr_grid, sharey='all', sharex='all')
        gss['panel_snr'] = axs_snr


        o_grid = pu.make_mxn_gridspec(self.gs, 2, 2, 30, 100, 0, 100, 10, 15)
        axs_other = self.get_axs(o_grid)

        gss['panel_k'] = axs_other[0, 0]
        gss['panel_n'] = axs_other[0, 1]

        gss['panel_recovery'] = axs_other[1]
    
        self.gss = gss

    def panel_snr(self, recompute=False):
        key = 'panel_snr'
        axs = self.gss[key][0]

        n_values = self.params.getint('n_vals')
        n_feats = self.params.getint('n_feats')
        n_stim = self.params.getint('n_stim')
        tradeoffs = self.params.getlist('eg_trades', typefunc=float)
        p_range = self.params.getlist('pwr_range', typefunc=float)
        n_ps = self.params.getint('n_pwrs')
        pwrs_root = np.linspace(*p_range, n_ps)

        if self.data.get(key) is None or recompute:
            out_manypwrs = mrt.get_ccgp_error_dep(
                pwrs_root, tradeoffs, n_feats, n_values, n_stim,
                hamming_reps=5000,
            )

            self.data[key] = out_manypwrs
        out = self.data[key]

        conj_color = self.params.getcolor("conj_color")
        lin_color = self.params.getcolor("lin_color")
        cmap = gpl.make_linear_cmap(conj_color, lin_color)
        for i, trade in enumerate(tradeoffs):
            plot_pwrs = list(x[i] for x in out)
            err_theor, err_emp, swap_theor, gen_theor, gen_emp = plot_pwrs

            color = cmap(trade)
            l_ = gpl.plot_trace_werr(
                pwrs_root, 1 - err_emp.T, ax=axs[0], conf95=True, log_y=True,
                label='r = {:.1f}'.format(trade), color=color,
            )
            gpl.plot_trace_werr(pwrs_root, err_theor, ax=axs[0], color=l_[0].get_color(),
                                plot_outline=True, linestyle='dashed')
            gpl.plot_trace_werr(pwrs_root, swap_theor, ax=axs[1],
                                plot_outline=True, linestyle='dashed',
                                color=color,)

            l_ = gpl.plot_trace_werr(pwrs_root, 1 - gen_emp.T, ax=axs[2], conf95=True,
                                    color=color)
            gpl.plot_trace_werr(pwrs_root, gen_theor, ax=axs[2], color=l_[0].get_color(),
                                plot_outline=True, linestyle='dashed')
        axs[0].set_ylabel('error rate')
        axs[1].set_ylabel('misbinding\nerror rate')
        axs[2].set_ylabel('generalization\nerror rate')
        axs[0].set_xlabel('SNR')
        axs[1].set_xlabel('SNR')
        axs[2].set_xlabel('SNR')

    def panel_k(self):
        key = 'panel_k'
        ax = self.gss[key]

        pwr_root = np.array([self.params.getfloat('pwr_eg')])

        n_trades = np.linspace(0, 1, self.params.getint('n_trades'))

        n_feats = self.params.getlist('k_egs', typefunc=int)
        n_vals = self.params.getint('n_vals')
        n_stim = self.params.getint('n_stim')

        b_cmap = plt.get_cmap(self.params.get('bind_cmap'))
        g_cmap = plt.get_cmap(self.params.get('gen_cmap'))

        col_space = np.linspace(.2, .8, len(n_feats))
        for i, nf in enumerate(n_feats):
            pwr_use = pwr_root # *nf
            out = mrt.get_ccgp_error_dep(pwr_use, n_trades, nf, n_vals, n_stim,
                                         empirical=False)
            err, _, swap, gen, _ = out

            b_color = b_cmap(col_space[i])
            g_color = g_cmap(col_space[i])

            if i == len(n_feats) - 1:
                swap_label = 'misbinding'
                gen_label = 'generalization'
            else:
                swap_label = ''
                gen_label = ''
            gpl.plot_trace_werr(
                n_trades, np.squeeze(swap), ax=ax, color=b_color, zorder=-1,
                label=swap_label,
            )
            gpl.plot_trace_werr(
                n_trades, np.squeeze(gen), ax=ax, color=g_color, label=gen_label,
            )
        ax.set_ylim([0, .6])
        ax.set_xlabel('subspace correlation (r)')
        ax.set_ylabel('error rate')

    def panel_n(self):
        key = 'panel_n'
        ax = self.gss[key]

        pwr_root = np.array([self.params.getfloat('pwr_eg')])

        n_trades = np.linspace(0, 1, self.params.getint('n_trades'))

        n_feats = self.params.getint('n_feats')
        n_vals = self.params.getlist('nv_egs', typefunc=int)
        n_stim = self.params.getint('n_stim')

        b_cmap = plt.get_cmap(self.params.get('bind_cmap'))
        g_cmap = plt.get_cmap(self.params.get('gen_cmap'))

        col_space = np.linspace(.2, .8, len(n_vals))
        for i, nv in enumerate(n_vals):
            pwr_use = pwr_root # *nv

            out = mrt.get_ccgp_error_dep(pwr_use, n_trades, n_feats, nv, n_stim,
                                         empirical=False)
            err, _, swap, gen, _ = out

            b_color = b_cmap(col_space[i])
            g_color = g_cmap(col_space[i])
            if i == len(n_vals) - 1:
                swap_label = 'misbinding'
                gen_label = 'generalization'
            else:
                swap_label = ''
                gen_label = ''
            gpl.plot_trace_werr(
                n_trades, np.squeeze(swap), ax=ax, color=b_color, zorder=-1,
                label=swap_label,
            )
            gpl.plot_trace_werr(
                n_trades, np.squeeze(gen), ax=ax, color=g_color,
                label=gen_label,
            )
        ax.set_ylim([0, .6])
        ax.set_xlabel('subspace correlation (r)')
        ax.set_ylabel('error rate')

    def panel_recovery(self):
        key = 'panel_recovery'
        ax_dl, ax_dn = self.gss[key]

        n_feats = self.params.getint('n_feats')
        n_vals = self.params.getint('n_vals')
        n_units = self.params.getint('n_units')
        pwrs = self.params.getlist('recovery_pwrs', typefunc=float)
        tradeoffs = self.params.getlist('recovery_tradeoffs', typefunc=float)
        n_boots = self.params.getint('n_boots')
        n_samps = self.params.getint('n_samples')

        if self.data.get(key) is None:
            out = mra.estimate_distances(
                pwrs, tradeoffs, n_feats, n_vals, n_units=n_units, n_boots=n_boots,
                n_samples=n_samps
            )
            self.data[key] = out
        dist_ests = self.data[key]
        cmap = plt.get_cmap('RdPu')
        
        colors = cmap(np.linspace(.3, .9, len(pwrs)))
        for i, pwr in enumerate(pwrs):

            dl_ests = np.squeeze(dist_ests['dl'][1][i])
            dl_trues = dist_ests['dl'][0][i]
            l_ = gpl.plot_trace_werr(tradeoffs, dl_trues.T, ax=ax_dl,
                                    linestyle='dashed',
                                    plot_outline=True,
                                    color=colors[i])
            col = l_[0].get_color()
            gpl.plot_trace_werr(tradeoffs, dl_ests.T, ax=ax_dl,
                                color=col, central_tendency=np.nanmedian,
                                conf95=True)

            dn_ests = np.squeeze(dist_ests['dn'][1][i])
            dn_trues = dist_ests['dn'][0][i]
            gpl.plot_trace_werr(tradeoffs, dn_trues.T, ax=ax_dn,
                                color=col, linestyle='dashed',
                                plot_outline=True,
                                label='total power = {}'.format(pwr))
            gpl.plot_trace_werr(tradeoffs, dn_ests.T, ax=ax_dn,
                                central_tendency=np.nanmedian,
                                conf95=True, color=col)
        ax_dl.set_ylabel('linear distance')
        ax_dn.set_ylabel('nonlinear distance')
        ax_dn.set_xlabel('subspace correlation (r)')
        ax_dl.set_xlabel('subspace correlation (r)')


class DecodingCurrentPastFigure(MultipleRepFigure):
    def __init__(self, fig_key="combined_rep_tc_figure", colors=colors, **kwargs):
        fsize = (8, 5)
        cf = u.ConfigParserColor()
        cf.read(config_path)

        params = cf[fig_key]
        self.fig_key = fig_key
        self.saved_coefficients = None
        super().__init__(fsize, params, colors=colors, **kwargs)    

    def make_gss(self):
        gss = {}
        n_times = 2
        n_targs = 4

        gs_tc = pu.make_mxn_gridspec(
            self.gs, n_targs, n_times,
            0, 100, 0, 45, 5, 2
        )
        axs_tc = self.get_axs(gs_tc, sharey="all", sharex="all")
        gss["panel_dec_tc"] = axs_tc

        gs_regions = pu.make_mxn_gridspec(
            self.gs, n_targs, n_times,
            0, 100, 52, 100, 5, 2
        )
        axs_regions = self.get_axs(gs_regions, sharey="all", sharex="all")
        gss["panel_regions"] = axs_regions

        self.gss = gss

    def panel_dec_tc(self, reload=False):
        key = "panel_dec_tc"
        axs = self.gss[key]
        
        ri = self.params.get("tc_run_ind")
        eg_region = self.params.get("tc_eg_region")
        if self.data.get(key) is None or reload:
            comb_dict = mraux.load_decoding_runs(ri)
            self.data[key] = comb_dict
        comb_dict = self.data[key]
        mrv.plot_current_past_dict(
            comb_dict,
            plot_regions=(eg_region,),
            axs=axs,
            color_dict=self.get_color_dict(),
        )

    def panel_regions(self, reload=False):
        key = "panel_regions"
        axs = self.gss[key]
        
        ri = self.params.get("regions_run_ind")
        if self.data.get(key) is None or reload:
            comb_dict = mraux.load_decoding_runs(ri)
            self.data[key] = comb_dict
        comb_dict = self.data[key]
        mrv.plot_current_past_regions_dict(
            comb_dict,
            axs=axs,
            color_dict=self.get_color_dict(),
            plot_regions=self.params.getlist("use_regions"),
        )

        
class DecodingTCFigure(MultipleRepFigure):
    def __init__(self, fig_key="dec_tc_figure", colors=colors, **kwargs):
        fsize = (4.5, 8)
        cf = u.ConfigParserColor()
        cf.read(config_path)

        params = cf[fig_key]
        self.fig_key = fig_key
        self.saved_coefficients = None
        super().__init__(fsize, params, colors=colors, **kwargs)

    def make_gss(self):
        gss = {}
        n_conds = 2
        n_regions = len(self.regions)

        gs_decs = pu.make_mxn_gridspec(
            self.gs, n_regions, n_conds,
            0, 100, 0, 100, 5, 15
        )
        axs_decs = self.get_axs(gs_decs, sharey="all", sharex="all")

        gss["panel_dec_tc"] = axs_decs
        self.gss = gss

    def panel_dec_tc(self, reload=False):
        key = "panel_dec_tc"
        axs = self.gss[key]

        if self.data.get(key) is None or reload:
            comb_dict = mraux.load_decoding_runs("11370117")
            self.data[key] = comb_dict
        dec_dict = self.data[key]["decoding"]

        mrv.plot_dec_dict(
            dec_dict,
            axs=axs.T,
            color_dict=self.get_color_dict(),
            region_list=self.regions,
            y_label="decoding performance",
        )
        mra.print_dec_differences(dec_dict, t_comp=250)
        mra.print_dec_differences(dec_dict, t_comp=500)
        mra.print_dec_differences(dec_dict, t_comp=750)
        
        axs[-1, 0].set_xlabel("time from offer onset")
        axs[-1, 1].set_xlabel("time from offer onset")
        axs[0, 0].set_title("space")
        axs[0, 1].set_title("time")
    
        
class TheoryFigure(MultipleRepFigure):
    def __init__(self, fig_key="theory_figure", colors=colors, **kwargs):
        fsize = (4.5, 8)
        cf = u.ConfigParserColor()
        cf.read(config_path)

        params = cf[fig_key]
        self.fig_key = fig_key
        self.saved_coefficients = None
        super().__init__(fsize, params, colors=colors, **kwargs)

    def make_gss(self):
        gss = {}
        use_contrasts = self.params.getlist("use_contrasts")
        if use_contrasts is None:
            preds = self._get_gen_emp_pred()[0]
            n_conds = len(preds["all"].dtype.names)
        else:
            n_conds = len(use_contrasts)

        gs_dists = pu.make_mxn_gridspec(self.gs, 1, 2, 0, 18, 0, 100, 10, 15)
        axs_dists = self.get_axs(gs_dists, sharey=True, sharex=True)

        gs12 = pu.make_mxn_gridspec(self.gs, 2, n_conds, 28, 70, 0, 100, 5, 15)
        axs12 = self.get_axs(gs12, sharex=True, sharey="horizontal")

        axs1 = axs12[0:1]
        axs2 = axs12[1:2]

        gs3 = pu.make_mxn_gridspec(self.gs, 1, n_conds, 80, 100, 0, 100, 10, 15)
        axs3 = self.get_axs(gs3, sharey=True)

        gs_inset = pu.make_mxn_gridspec(self.gs, 1, n_conds, 80, 90, 20, 100, 0, 35)
        axs_inset = self.get_axs(gs_inset, sharey=True, sharex=True)

        gss["panel_theory_dec_plot"] = (
            axs_dists, np.concatenate((axs1, axs2, axs_inset)).T
        )
        gss["panel_theory_comparison"] = axs3
        self.gss = gss

    def get_coeffs_bootstrapped(
        self,
        force_reload=False,
        methods=("Boot-EN",),
        key="ALL_MONKEY",
        col_key="variables",
    ):
        if self.data.get("saved_coefficients") is None or force_reload:
            folder = self.params.get("coeff_folder")
            file_ = self.params.get("coeff_boot_file")
            use_regions = self.params.getlist("use_regions")
            time_ind = self.params.getint("coeff_time_ind")
            column_file = self.params.get("coeff_boot_columns")

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
                if "all" in use_regions:
                    coeff_list = list(oc[0] for oc in out_coeffs[method].values())
                    coeff_all = np.concatenate(coeff_list, axis=0)
                    out_coeffs[method]["all"] = (coeff_all, columns)
            self.data["saved_coefficients"] = out_coeffs
        return self.data["saved_coefficients"]

    def get_coeffs(self, force_reload=False):
        if self.data.get("saved_coefficients") is None or force_reload:
            methods = self.params.getlist("methods")
            coeff_folder = self.params.get("coeff_folder")
            coeff_template = self.params.get("coeff_template")
            use_regions = self.params.getlist("use_regions")

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
                out_coeffs[method]["all"] = all_conc
            self.data["saved_coefficients"] = out_coeffs
        return self.data["saved_coefficients"]

    def panel_coeff_prediction(self, force_recompute=False, boot=True):
        key = "panel_coeff_prediction"
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
                        rwd_nonlin = _get_rwd_nonlin_boot(coeffs, columns=columns)
                        prob_nonlin = _get_prob_nonlin_boot(coeffs, columns=columns)
                        r2 = np.zeros((rwd_lin.shape[0], 1, 1))
                        resid_var = np.sqrt(_get_nv_boot(coeffs, columns=columns))
                    else:
                        rwd_lin = _get_rwd_lin(cs)
                        prob_lin = _get_prob_lin(cs)
                        rwd_nonlin = _get_rwd_nonlin(cs)
                        prob_nonlin = _get_prob_nonlin(cs)
                        r2 = _get_template_columns(cs, "pseudoR2")
                        if r2.shape[1] > 0:
                            resid_var = 1 - r2
                        else:
                            resid_var = np.expand_dims(1, (0, 1))
                    rwd_out = mra.predict_asymp_dists(
                        rwd_lin, rwd_nonlin, resid_var, k=2, n=2
                    )
                    prob_out = mra.predict_asymp_dists(
                        prob_lin, prob_nonlin, resid_var, k=2, n=2
                    )
                    rwd = {"bind_err": rwd_out[0], "gen_err": rwd_out[1]}
                    prob = {"bind_err": prob_out[0], "gen_err": prob_out[1]}

                    out_prediction[method][region] = {"rwd": rwd, "prob": prob}
            self.data[key] = out_prediction
        return self.data[key]

    def panel_theory_comparison(self, **kwargs):
        key = "panel_theory_comparison"
        axs_inset = self.gss[key]
        
        preds, decs = self._get_gen_emp_pred(**kwargs)
        regions = self.params.getlist("use_regions")

        normalize_dimensions = self.params.getboolean("normalize_dimensions")
        n_virtual_dims = self.params.getint("n_virtual_dims")

        if normalize_dimensions:
            preds = mra.normalize_pred_dimensions(preds, n_virtual_dims)

        use_contrasts = self.params.getlist("use_contrasts")
        if use_contrasts is None:
            contrasts = preds[regions[0]].dtype.names
        else:
            contrasts = use_contrasts
        cnames = self.params.getlist("contrast_names")
        if cnames is None:
            cnames = contrasts
        comp_region = "all"
        for i, region in enumerate(regions):
            for j, contrast in enumerate(contrasts):
                pred_ij = preds[region][contrast]
                dec_ij = decs[region][contrast]
                comp_pred_ij = preds[comp_region][contrast]
                color = self.get_color_dict()[region]
                try:
                    print("{}-{}".format(region, comp_region), cnames[j])
                    mrv.plot_data_pred(
                        pred_ij[0, 0],
                        dec_ij[0, 0],
                        label=region,
                        color=color,
                        ax_comb=axs_inset[0, j],
                        comp_pred=comp_pred_ij[0, 0],
                        print_differences=True,
                    )
                except IndexError as e:
                    print(e)
                    s = "did not plot {} for {} (likely a nan)"
                    print(s.format(region, contrast))
        for i, j in u.make_array_ind_iterator(axs_inset.shape):
            ax = axs_inset[i, j]
            gpl.add_hlines(.125, ax)
            gpl.add_vlines(.5, ax)
            if j == 0:
                ax.set_ylabel("binding error rate")
            ax.set_xlabel("generalization\nerror rate")
        

    def panel_theory_dec_plot(self, **kwargs):
        key = "panel_theory_dec_plot"
        axs_dists, axs = self.gss[key]

        preds, decs = self._get_gen_emp_pred(**kwargs)
        regions = self.params.getlist("use_regions")

        normalize_dimensions = False
        n_virtual_dims = self.params.getint("n_virtual_dims")

        if normalize_dimensions:
            preds = mra.normalize_pred_dimensions(preds, n_virtual_dims)

        mrv.plot_distances(
            preds,
            axs=np.squeeze(axs_dists),
            other_key="non other",
            region_order=regions,
        )

        use_contrasts = self.params.getlist("use_contrasts")
        if use_contrasts is None:
            contrasts = preds[regions[0]].dtype.names
        else:
            contrasts = use_contrasts
        cnames = self.params.getlist("contrast_names")
        if cnames is None:
            cnames = contrasts
        comp_region = "all"
        for i, region in enumerate(regions):
            for j, contrast in enumerate(contrasts):
                pred_ij = preds[region][contrast]
                dec_ij = decs[region][contrast]
                comp_pred_ij = preds[comp_region][contrast]
                color = self.get_color_dict()[region]
                try:
                    mrv.plot_data_pred(
                        pred_ij[0, 0],
                        dec_ij[0, 0],
                        axs=axs[j],
                        label=region,
                        color=color,
                        comp_pred=comp_pred_ij[0, 0],
                    )
                except IndexError as e:
                    print(e)
                    s = "did not plot {} for {} (likely a nan)"
                    print(s.format(region, contrast))
                axs_dists[0, j].set_yticks([0, 1, 2])
                gpl.clean_plot(axs_dists[0, j], j)

        for j in range(axs.shape[0]):
            axs[j, 0].set_title(cnames[j])
            axs[j, 1].set_xlabel("subspace correlation (r)")
            # axs[j, 2].set_xlabel("generalization\nerror rate")
            # axs[j, 2].set_xlabel("generalization\nerror rate")
            gpl.add_hlines(0.125, axs[j, 0])
            gpl.add_hlines(0.5, axs[j, 1])
            gpl.add_hlines(0.125, axs[j, 2])
            gpl.add_vlines(0.5, axs[j, 2])

        # axs_dists[0, -1].legend(handles=(l_lin, l_con), frameon=False)
        axs[0, 1].set_ylabel("generalization\nerror rate")
        axs[0, 0].set_ylabel("binding error rate")
        # axs[0, 2].set_ylabel("binding error rate")

    def _get_subspace_correlations(
        self,
        force_reload=False,
        subspace_key="subspace_corr",
        linear_side="without_linear_side_incld",
    ):
        if self.data.get("subspace_correlations") is None or force_reload:
            methods = self.params.getlist("methods")
            coeff_folder = self.params.get("coeff_folder")
            subspace_template = self.params.get("subspace_template")
            use_regions = self.params.getlist("use_regions")

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
            self.data["subspace_correlations"] = out_corrs
        return self.data["subspace_correlations"]

    def panel_subspace_corr_example(self, force_recompute=False):
        key = "panel_subspace_corr_example"
        n_pwrs = 5
        n_trades = 101
        sum_d2s = np.linspace(5, 10, n_pwrs) ** 2
        trades = np.linspace(0, 1, n_trades)

        n_feats = self.params.getint("n_feats")
        n_vals = self.params.getint("n_vals")
        sigma = self.params.getfloat("sigma")
        if self.data.get(key) is None or force_recompute:
            gen_err = np.zeros((n_pwrs, n_trades))
            bind_err = np.zeros_like(gen_err)
            for i, sd2 in enumerate(sum_d2s):
                r1, err1 = mrt.vector_corr_ccgp(sd2, trades, sigma=sigma)
                r2, err2 = mrt.vector_corr_swap(
                    sd2, n_feats, n_vals, trades, sigma=sigma
                )
                gen_err[i] = err1
                bind_err[i] = err2
            self.data[key] = {
                "pwrs": sum_d2s,
                "r": r1,
                "bind_err": bind_err,
                "gen_err": gen_err,
            }
        return self.data[key]

    def panel_subspace_corr_range(self, force_recompute=False):
        key = "panel_subspace_corr_range"
        corrs = self._get_subspace_correlations(force_reload=force_recompute)
        pwr_params = self.params.getlist("pwr_params", typefunc=float)
        pwrs = np.linspace(*pwr_params[:2], int(pwr_params[2]))
        n_feats = self.params.getint("n_feats")
        n_vals = self.params.getint("n_vals")
        sigma = self.params.getfloat("sigma")
        if self.data.get(key) is None or force_recompute:
            out_arrs = {}
            for method, regions in corrs.items():
                out_arrs[method] = {}
                for region, arr in regions.items():
                    mr_bind = np.zeros(arr.shape + (len(pwrs),))
                    mr_gen = np.zeros_like(mr_bind)
                    for i, j in u.make_array_ind_iterator(arr[:, :2].shape):
                        out = mrt.pwr_range_corr(
                            arr[i, j], pwrs, n_feats, n_vals, sigma=sigma
                        )
                        mr_bind[i, j] = out[0]
                        mr_gen[i, j] = out[1]
                        mr_bind[i, 2] = arr[i, 2]
                        mr_gen[i, 2] = arr[i, 2]
                    out_arrs[method][region] = {
                        "corr_values": arr,
                        "bind_err": mr_bind,
                        "gen_err": mr_gen,
                    }
            self.data[key] = {"probed_pwrs": pwrs, "out": out_arrs}
        return self.data[key]
