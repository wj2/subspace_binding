
import numpy as np
import sklearn.decomposition as skd
import sklearn.manifold as skman
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import functools as ft
import scipy.io as sio
import scipy.stats as sts
import itertools as it

import general.utility as u
import general.plotting as gpl
import composite_tangling.code_analysis as ca
import multiple_representations.analysis as mra
import multiple_representations.auxiliary as mraux
import multiple_representations.theory as mrt


# def _get_stim_reps(mat, inter, pred_func, val_ext=(-1.5, 1.5), n_pts=100,
#                    link_function=None, zero_preds=None):
#     vals = np.expand_dims(np.linspace(*val_ext, n_pts), 1)
#     side1 = np.ones((n_pts, 1))
#     side2 = -np.ones((n_pts, 1))

#     args1 = np.concatenate((vals, side1), axis=1)
#     args2 = np.concatenate((vals, side2), axis=1)

#     pred1 = pred_func(args1)
#     pred2 = pred_func(args2)

#     pred1 = np.expand_dims(pred1, (0, 1))
#     pred2 = np.expand_dims(pred2, (0, 1))

#     if zero_preds is not None:
#         if zero_preds == 'right':
#             pred1 = np.concatenate((pred1, np.zeros_like(pred1)), axis=-1)
#             pred2 = np.concatenate((pred2, np.zeros_like(pred2)), axis=-1)
#         else:
#             pred1 = np.concatenate((np.zeros_like(pred1), pred1), axis=-1)
#             pred2 = np.concatenate((np.zeros_like(pred2), pred2), axis=-1)

#     mat_use = np.expand_dims(mat, 2)
#     np.expand_dims(pred1, -1)
#     np.expand_dims(pred2, -1)

#     stim1 = np.swapaxes(np.sum(mat_use*pred1, axis=-1) + inter, 1, 2)
#     stim2 = np.swapaxes(np.sum(mat_use*pred2, axis=-1) + inter, 1, 2)
#     if link_function is not None:
#         stim1 = link_function(stim1)
#         stim2 = link_function(stim2)
#     return stim1, stim2


def plot_split_halfs_full(fit_dict, pred_func, ax=None, use_regions=None,
                          pearson_brown=False, align_func=mra.compute_corr,
                          n_pts=100, ac_pt=0, wi_pt=1, wi_d_pt=1,
                          wi_color=None,
                          ac_color=None, zp1=None, zp2=None, **kwargs):
    if use_regions is None:
        use_regions = ('OFC', 'PCC', 'pgACC', 'VS', 'vmPFC')
    mat1 = np.concatenate(list(v[0][0] for (r, _, _), v in fit_dict.items()
                               if r[0] in use_regions),
                          axis=1)
    inter1 = np.concatenate(list(v[0][1] for (r, _, _), v in fit_dict.items()
                                 if r[0] in use_regions),
                            axis=1)
    mat2 = np.concatenate(list(v[1][0] for (r, _, _), v in fit_dict.items()
                               if r[0] in use_regions),
                          axis=1)
    inter2 = np.concatenate(list(v[1][1] for (r, _, _), v in fit_dict.items()
                                 if r[0] in use_regions),
                            axis=1)

    # side 1 and side 2 for models 1
    stim11_o1, stim12_o1 = mra._get_stim_reps(mat1, inter1, pred_func, n_pts=n_pts,
                                          zero_preds=zp1, **kwargs)
    stim11_o2, stim12_o2 = mra._get_stim_reps(mat1, inter1, pred_func, n_pts=n_pts,
                                          zero_preds=zp2, **kwargs)

    # side 1 and side 2 for models 2
    stim21_o1, stim22_o1 = mra._get_stim_reps(mat2, inter2, pred_func, n_pts=n_pts,
                                          zero_preds=zp1, **kwargs)
    stim21_o2, stim22_o2 = mra._get_stim_reps(mat2, inter2, pred_func, n_pts=n_pts,
                                          zero_preds=zp2, **kwargs)

    rs_wi_d = align_func(stim11_o2, stim12_o2)
    # rs_wi1 = align_func(stim11_o1, stim11_o2)
    # rs_wi2 = align_func(stim12_o1, stim12_o2)
    rs_wi1 = align_func(stim11_o1, stim12_o2)
    rs_wi2 = align_func(stim12_o1, stim11_o2)
    rs_ac1 = align_func(stim11_o1, stim21_o1)
    rs_ac2 = align_func(stim12_o2, stim22_o2)
    rs_wi = np.mean((rs_wi1, rs_wi2), axis=0)

    # rs_wi_d = np.mean((rs_wi_d1, rs_wi_d2), axis=0)
    rs_ac = np.sqrt(rs_ac1*rs_ac2)
    if pearson_brown:
        rs_wi_d = 2*rs_wi_d/(1 + rs_wi_d)
        rs_wi = 2*rs_wi/(1 + rs_wi)
        rs_ac = 2*rs_ac/(1 + rs_ac)

    gpl.violinplot(np.expand_dims(rs_wi, 0), [wi_pt], color=[wi_color],
                   ax=ax, showextrema=False, showmedians=True)
    # gpl.violinplot(np.expand_dims(rs_wi_d, 0), [wi_d_pt], color=[wi_color],
    #                ax=ax, showextrema=False, showmedians=True)
    gpl.violinplot(np.expand_dims(rs_ac, 0), [ac_pt], color=[ac_color],
                   ax=ax, showextrema=False, showmedians=True)


def plot_split_halfs_only(
        *rs,
        ax=None,
        pts=None,
        pt=0,
        colors=None,
        markerstyles=None,
        **kwargs,
):
    if ax is None:
        f, ax = plt.subplots(1, 1)
    if pts is None:
        pts = (pt,)*len(rs)
    if colors is None:
        colors = (None,)*len(rs)
    for i, r in enumerate(rs):
        gpl.violinplot(
            np.expand_dims(r, 0),
            [pts[i]],
            color=[colors[i]],
            ax=ax,
            showextrema=False,
            showmedians=True,
            markerstyles=markerstyles,
            **kwargs,
        )
    return ax


def plot_split_halfs(fit_dict, pred_func, ax=None, use_regions=None,
                     pearson_brown=False, align_func=mra.compute_corr,
                     n_pts=100, ac_pt=0, wi_pt=1, wi_color=None,
                     ac_color=None, zp1=None, zp2=None, markerstyles=None,
                     **kwargs):
    if use_regions is None:
        use_regions = ('OFC', 'PCC', 'pgACC', 'VS', 'vmPFC')
    mat1 = np.concatenate(list(v[0][0] for (r, _, _), v in fit_dict.items()
                               if r[0] in use_regions),
                          axis=1)
    inter1 = np.concatenate(list(v[0][1] for (r, _, _), v in fit_dict.items()
                                 if r[0] in use_regions),
                            axis=1)
    mat2 = np.concatenate(list(v[1][0] for (r, _, _), v in fit_dict.items()
                               if r[0] in use_regions),
                          axis=1)
    inter2 = np.concatenate(list(v[1][1] for (r, _, _), v in fit_dict.items()
                                 if r[0] in use_regions),
                            axis=1)
    k_list = list(
        list(zip((k[2],)*len(k[0]), range(len(k[0]))))
        for k in fit_dict.keys()
        if k[0][0] in use_regions
    )
    key_group = np.concatenate(k_list, axis=0)

    # side 1 and side 2 for models 1
    stim11, stim12 = mra._get_stim_reps(mat1, inter1, pred_func, n_pts=n_pts,
                                    **kwargs)

    # side 1 and side 2 for models 2
    stim21, stim22 = mra._get_stim_reps(mat2, inter2, pred_func, n_pts=n_pts,
                                    **kwargs)

    rs_wi = align_func(stim11, stim12)
    rs_ac1 = align_func(stim11, stim21)
    rs_ac2 = align_func(stim12, stim22)
    rs_ac = np.sqrt(rs_ac1*rs_ac2)
    if pearson_brown:
        rs_wi = 2*rs_wi/(1 + rs_wi)
        rs_ac = 2*rs_ac/(1 + rs_ac)
    gpl.violinplot(
        np.expand_dims(rs_wi, 0),
        [wi_pt],
        color=[wi_color],
        ax=ax,
        showextrema=False,
        showmedians=True,
        markerstyles=markerstyles
    )
    gpl.violinplot(
        np.expand_dims(rs_ac, 0),
        [ac_pt],
        color=[ac_color],
        ax=ax,
        showextrema=False,
        showmedians=True,
        markerstyles=markerstyles
    )

def plot_single_neurons(data, region=None, fwid=1, comp_warnings=None):
    plots = []
    for i, (date, n_neur) in data[['date', 'n_neurs']].iterrows():
        r, _, _ = date.split('-')
        if region is None or region == r:
            for neur_ind in range(n_neur):
                try:
                    if comp_warnings is not None:
                        k = tuple(date.split('-')[:2]) + (date, neur_ind)
                        include = not np.any(comp_warnings[k]['warning'])
                    else:
                        include = True
                    if include:
                        out = mra.make_psth_val_avg(data, date, neur_ind)
                        f_key = (date, neur_ind)
                        plots.append((f_key, out))
                except IndexError as e:
                    pass
    n_plots = len(plots)
    n_cols = int(np.ceil(np.sqrt(n_plots)))
    f, axs = plt.subplots(n_cols, n_cols, figsize=(fwid*n_cols, fwid*n_cols))
    ax_flat = axs.flatten()
    for i, data_p in enumerate(plots):
        ax = ax_flat[i]
        f_key, (xs, act, vs, sides) = data_p
        plot_value_tuning(xs, act, vs, sides, ax=ax)
        # mrv.plot_value_resp(xs, act, vs, sides, ax=ax)
        ax.set_title(f_key)
    
def plot_value_resp(xs, resp, vals, sides, n_val_bins=2, val_lim=None, ax=None,
                    eps=1e-10, side_colors=None, amt=.15, labels=None):
    if ax is None:
        f, ax = plt.subplots(1, 1)
    if val_lim is None:
        val_lim = (np.min(vals) - eps, np.max(vals) + eps)
    bins = np.linspace(*val_lim, n_val_bins + 1)
    groups = np.digitize(vals, bins)
    u_groups = np.unique(groups)
    u_sides = np.unique(sides)
    add_range = np.linspace(amt, 0, len(u_groups))
    if side_colors is None:
        side_colors = (None,)*len(u_sides)
    for j, side in enumerate(u_sides):
        side_mask = side == sides
        for i, ug in enumerate(u_groups):
            mask = np.logical_and(groups == ug, side_mask)
            if side_colors[j] is not None:
                color = gpl.add_color_value(side_colors[j], add_range[i])
            else:
                color = None
            if labels is not None:
                label = labels[side][i]
            else:
                label = ''
            gpl.plot_trace_werr(xs, resp[mask], ax=ax, color=color,
                                label=label)

def plot_value_tuning(xs, resp, vals, sides, n_val_bins=5, val_lim=None,
                      ax=None, eps=1e-10, x_start=100, x_end=1000,
                      plot_model=None,
                      side_colors=None, **kwargs):
    if ax is None:
        f, ax = plt.subplots(1, 1)
    if val_lim is None:
        val_lim = (np.min(vals) - eps, np.max(vals) + eps)
    bins = np.linspace(*val_lim, n_val_bins + 1)
    groups = np.digitize(vals, bins)
    u_groups = np.unique(groups)
    u_sides = np.unique(sides)
    if side_colors is None:
        side_colors = (None,)*len(u_sides)
    x_mask = np.logical_and(xs >= x_start, xs < x_end)

    for j, side in enumerate(u_sides):
        resps = []
        vs = []
        side_mask = side == sides
        for i, ug in enumerate(u_groups):
            mask = np.logical_and(ug == groups, side_mask)
            vs.append(np.mean(vals[mask]))
            resp_m = np.squeeze(resp[mask])
            if len(resp_m.shape) > 1:
                resp_m = np.mean(resp_m[:, x_mask], axis=1)
            resps.append(resp_m)
        gpl.plot_trace_werr(vs, resps, jagged=True, ax=ax,
                            color=side_colors[j], **kwargs)
            
    
def visualize_model_weights(comp_dict, use_regions=None,
                            v1_keys=('noise',),
                            v2_keys=('null', 'interaction_spline'),
                            v3_keys=('interaction', 'null_spline'),
                            highlight_inds=None,
                            print_props=True,
                            **kwargs):
    if use_regions is None:
        use_regions = ('OFC', 'PCC', 'pgACC', 'VS', 'vmPFC')
    weights = []
    highlights = []
    monkeys = []
    for ind, v in comp_dict.items():
        (r, monk, _, _) = ind
        if r in use_regions and not np.any(v['warning']):
            k_weights = v['weight']
            ws1 = np.sum(k_weights[v1_i] for v1_i in v1_keys)
            ws2 = np.sum(k_weights[v2_i] for v2_i in v2_keys)
            ws3 = np.sum(k_weights[v3_i] for v3_i in v3_keys)
            weights.append((ws1, ws2, ws3))
            monkeys.append(monk)
            if ind in highlight_inds:
                highlights.append((ws1, ws2, ws3))

    monkeys = np.array(monkeys)
    u_monks = np.unique(monkeys)
    w_arr = np.array(weights)
    if print_props:
        m_types = np.argmax(w_arr, axis=1)
        print('{}: noise best fit: {:.2f}'.format(use_regions,
                                              np.mean(m_types == 0)))
        print('{}: linear best fit: {:.2f}'.format(use_regions,
                                               np.mean(m_types == 1)))
        print('{}: interaction best fit: {:.2f}'.format(use_regions,
                                                    np.mean(m_types == 2)))
        for um in u_monks:
            mask = monkeys == um
            m_types = np.argmax(w_arr[mask], axis=1)
            print(
                '{}, {}: noise best fit: {:.2f}  {}/{}'.format(
                    use_regions,
                    um,
                    np.mean(m_types == 0),
                    np.sum(m_types == 0),
                    len(m_types),
                )
            )
            print(
                '{}, {}: linear best fit: {:.2f}  {}/{}'.format(
                    use_regions, um, np.mean(m_types == 1),
                    np.sum(m_types == 1),
                    len(m_types),
                )
            )
            print(
                '{}, {}: interaction best fit: {:.2f}  {}/{}'.format(
                    use_regions, um, np.mean(m_types == 2), np.sum(m_types == 2),
                    len(m_types))
            )
            
    ax = gpl.visualize_simplex_2d(w_arr,  **kwargs)
    if len(highlights) > 0:
        h_arr = np.array(highlights)

        gpl.visualize_simplex_2d(h_arr, plot_bounds=False,
                                 plot_outline=True, **kwargs)
    return w_arr
    

def plot_dist_dict(
        dist_dict,
        color_dict=None,
        axs=None,
        fwid=3,
        pks=("d_l", "d_n"),
        plot_cond=('subj_ev_left', 'subj_ev_right'),
        xs_dict=None,
):
    n_plots = len(dist_dict)
    if color_dict is None:
        color_dict = {}
    if axs is None:
        f, axs = plt.subplots(n_plots, 1, figsize=(fwid, fwid*n_plots))
    for i, (r, cond_dict) in enumerate(dist_dict.items()):
        for pk in pks:
            pk_fits = cond_dict[plot_cond][pk]
            if xs_dict is not None:
                xs = xs_dict[r][plot_cond][1]
            else:
                xs = np.arange(pk_fits.shape[-1])
            gpl.plot_trace_werr(
                xs, pk_fits, conf95=True, color=color_dict.get(pk), ax=axs[i]
            )
        axs[i].set_title(r)
    
        
def _dec_res_func(res):
    dec, xs = res[:2]
    dec = np.mean(dec, axis=1)
    out = [dec, xs]
    if len(res) > 2:
        gen = res[-1]
        gen = np.mean(gen, axis=1)
        out.append(gen)
    return out


def _pred_res_func(res):
    out = (res["d_l"], res["d_n"])
    return out

            
def _organize_dec_dict(
        dec_dict,
        region_list=None,
        comb_func=np.mean,
        cond_groups=mra.dec_cond_groups,
        res_func=_dec_res_func,
        default_len=2,
):
    if region_list is None:    
        region_list = dec_dict.keys()
    out_dict = {}
    for j, r in enumerate(region_list):
        out_dict[r] = {}
        cond_dict = dec_dict[r]
        for i, (cond, ks) in enumerate(cond_groups.items()):
            outs = None
            for ind, k in enumerate(ks):
                res = cond_dict[k]
                if res is not None:
                    out = res_func(res)
                    if outs is None:
                        outs = list([] for _ in out)
                    list(o.append(out[i]) for i, o in enumerate(outs))
            if outs is None:
                outs = (None,)*default_len
            else:
                outs = list(comb_func(o, axis=0) for o in outs)
            out_dict[r][cond] = outs
    return out_dict

        
def plot_pred_dict(
        dec_dict,
        l_color=None,
        n_color=None,
        axs=None,
        fwid=1,
        set_title=False,
        y_label=None,
        cond_groups=mra.dec_cond_groups,
        offset=.2,
        t_ind=-1,
        res_func=_pred_res_func,
        **kwargs,
):
    n_plots = len(cond_groups)
    if axs is None:
        f, axs = plt.subplots(
            n_plots,
            1,
            figsize=(fwid, fwid*n_plots),
            sharey=True,
        )
    org_dict = _organize_dec_dict(
        dec_dict, cond_groups=cond_groups, res_func=res_func, **kwargs
    )
    regions = list(org_dict.keys())
    print(org_dict.keys())
    for j, (r, conds) in enumerate(org_dict.items()):
        for i, (cond, (dl, dn)) in enumerate(conds.items()):
            if dl is not None:
                gpl.violinplot(
                    [dl[:, t_ind]], [j - offset/2], ax=axs[i], color=l_color
                )
            if dn is not None:
                gpl.violinplot(
                    [dn[:, t_ind]], [j + offset/2], ax=axs[i], color=n_color
                )
    for i in range(len(conds)):
        axs[i].set_xticks(range(len(regions)))
        axs[i].set_xticklabels(regions)
        gpl.clean_plot(axs[i], 0)
        gpl.add_hlines(0, axs[i])
        if i < len(conds) - 1:
            gpl.clean_plot_bottom(axs[i])


def plot_monkey_pred_dict(
        monkey_dict,
        l_color=None,
        n_color=None,
        axs=None,
        fwid=1,
        offset=.2,
        t_ind=-1,
        excl_all=True,
        single_letter=True,
        comb_func=np.concatenate,
        **kwargs,
):
    r_dict = {}
    for i, (m, comb_dict) in enumerate(monkey_dict.items()):
        org_dict = _organize_dec_dict(
            comb_dict["predictions"], res_func=_pred_res_func, comb_func=comb_func,
        )
        for k, v in org_dict.items():
            kd = r_dict.get(k, {})
            kd[m] = v
            r_dict[k] = kd
    if excl_all:
        r_dict.pop("all")
    if axs is None:
        f, axs = plt.subplots(
            2,
            len(r_dict),
            figsize=(fwid*len(r_dict), fwid*2),
            sharey=True,
        )
    for j, (region, m_dict) in enumerate(r_dict.items()):
        for k, (monkey, conds) in enumerate(m_dict.items()):
            for i, (cond, (dl, dn)) in enumerate(conds.items()):
                if dl is not None:
                    gpl.violinplot(
                        [dl[:, t_ind]], [k - offset/2], ax=axs[i, j], color=l_color
                    )
                if dn is not None:
                    gpl.violinplot(
                        [dn[:, t_ind]], [k + offset/2], ax=axs[i, j], color=n_color
                    )
                if i == 0:
                    axs[i, j].set_title(region)
                if i == len(conds) - 1:
                    axs[i, j].set_xticks(range(len(m_dict)))
                    m_names = list(m_dict.keys())
                    if single_letter:
                        m_names = list(n[0] for n in m_names)
                    axs[i, j].set_xticklabels(m_names)
    for i, j in u.make_array_ind_iterator(axs.shape):
        gpl.clean_plot(axs[i, j], j)
        if i == 0:
            gpl.clean_plot_bottom(axs[i, j])
        gpl.add_hlines(0, axs[i, j])
    return axs

            
def plot_monkey_dec_dict(monkey_dict, axs=None, fwid=1, **kwargs):
    if axs is None:
        f, axs = plt.subplots(
            2,
            len(monkey_dict),
            figsize=(fwid*len(monkey_dict), fwid*2),
            sharex=True,
            sharey=True,
        )
    for i, (m, comb_dict) in enumerate(monkey_dict.items()):
        ax = axs[:, i:i+1]
        plot_dec_dict(comb_dict["decoding"], axs=ax, **kwargs)
        ax[0, 0].set_title(m)
    return axs

            
def plot_dec_dict(
        dec_dict,
        color_dict=None,
        axs=None,
        fwid=1,
        set_title=False,
        y_label=None,
        cond_groups=mra.dec_cond_groups,
        plot_gen=True,
        **kwargs,
):
    n_plots = len(cond_groups)
    n_regions = len(dec_dict)
    if color_dict is None:
        color_dict = {}
    if axs is None:
        f, axs = plt.subplots(
            n_plots,
            n_regions,
            figsize=(fwid*n_regions, fwid*n_plots),
            sharex=True,
            sharey=True,
            squeeze=False,
        )
    org_dict = _organize_dec_dict(dec_dict, cond_groups=cond_groups, **kwargs)
    for j, (r, conds) in enumerate(org_dict.items()):
        for i, (cond, (dec, xs, gen)) in enumerate(conds.items()):
            gpl.plot_trace_werr(
                xs,
                dec,
                ax=axs[i, j],
                color=color_dict.get(r),
                conf95=True,                
            )
            if not np.all(np.isnan(gen)) and plot_gen:
                gpl.plot_trace_werr(
                    xs,
                    gen,
                    ax=axs[i, j],
                    ls="dashed",
                    color=color_dict.get(r),
                    conf95=True,
                    plot_outline=True
                )

            if set_title:
                axs[i, j].set_title(r)
            if j == 0 and y_label is None:
                axs[i, j].set_ylabel(cond)
            elif y_label is not None:
                axs[i, j].set_ylabel(y_label)
            gpl.add_hlines(0.5, axs[i, j])
    return axs


def plot_current_past_regions_dict(
        dec_run_dict,
        color_dict=None,
        axs=None,
        fwid=2,
        plot_regions=None,
        t_ind=-1,
        offset=.2,
):
    n_plots = 2
    n_times = 2
    dec_dict = dec_run_dict["decoding"]
    timing_dict = dec_run_dict["timing"]
    if color_dict is None:
        color_dict = {}
    if axs is None:
        f, axs = plt.subplots(
            n_plots,
            n_times,
            figsize=(fwid*n_times, fwid*n_plots),
            sharex=True,
            sharey=True,
        )
    if plot_regions is None:
        plot_regions = dec_dict.keys()
        
    o1_to_o2_key = ('subj_ev offer 1', 'subj_ev offer 2')
    o2_to_o1_key = ('subj_ev offer 2', 'subj_ev offer 1')
    
    o1_time_o2o1_key = (('subj_ev offer 1', 'Offer 2 on'),
                        ('subj_ev offer 1', 'Offer 1 on'))
    for i, region in enumerate(plot_regions):
        dec, xs, gen = dec_dict[region][o1_to_o2_key]
        gpl.violinplot(
            [np.mean(dec, axis=1)[..., t_ind]],
            [i - offset/2],
            color=color_dict.get(region),
            ax=axs[0, 0],
        )
        gpl.violinplot(
            [np.mean(gen, axis=1)[..., t_ind]],
            [i + offset/2],
            color=color_dict.get(region),
            ax=axs[0, 1],
        )
        dec, xs, gen = dec_dict[region][o2_to_o1_key]
        gpl.violinplot(
            [np.mean(dec, axis=1)[..., t_ind]],
            [i - offset/2],
            color=color_dict.get(region),
            ax=axs[0, 1],
        )
        gpl.violinplot(
            [np.mean(gen, axis=1)[..., t_ind]],
            [i + offset/2],
            color=color_dict.get(region),
            ax=axs[0, 0],
        )

        dec, xs, gen = timing_dict[region][o1_time_o2o1_key]
        gpl.violinplot(
            [np.mean(dec, axis=1)[..., t_ind]],
            [i - offset/2],
            color=color_dict.get(region),
            ax=axs[1, 1],
        )
        gpl.violinplot(
            [np.mean(gen, axis=1)[..., t_ind]],
            [i + offset/2],
            color=color_dict.get(region),
            ax=axs[1, 0],
        )
    for i, j in u.make_array_ind_iterator(axs.shape):
        gpl.add_hlines(.5, axs[i, j])
        gpl.clean_plot(axs[i, j], j)


def plot_current_past_dict(
        dec_run_dict,
        color_dict=None,
        axs=None,
        fwid=2,
        plot_regions=None,
):
    n_plots = 2
    n_regions = 2
    dec_dict = dec_run_dict["decoding"]
    timing_dict = dec_run_dict["timing"]
    if color_dict is None:
        color_dict = {}
    if axs is None:
        f, axs = plt.subplots(
            n_regions,
            n_plots,
            figsize=(fwid*n_regions, fwid*n_plots),
            sharex=True,
            sharey=True,
        )
    if plot_regions is None:
        plot_regions = dec_dict.keys()
        
    o1_to_o2_key = ('subj_ev offer 1', 'subj_ev offer 2')
    o2_to_o1_key = ('subj_ev offer 2', 'subj_ev offer 1')
    
    o1_time_o2o1_key = (('subj_ev offer 1', 'Offer 2 on'),
                        ('subj_ev offer 1', 'Offer 1 on'))
    for i, region in enumerate(plot_regions):
        dec, xs, gen = dec_dict[region][o1_to_o2_key]
        gpl.plot_trace_werr(
            xs,
            np.mean(dec, axis=1),
            color=color_dict.get(region),
            ax=axs[0, 0],
            conf95=True,
        )
        gpl.plot_trace_werr(
            xs,
            np.mean(gen, axis=1),
            color=color_dict.get(region),
            ax=axs[0, 1],
            ls="dashed",
            conf95=True,
        )
        dec, xs, gen = dec_dict[region][o2_to_o1_key]
        gpl.plot_trace_werr(
            xs,
            np.mean(dec, axis=1),
            color=color_dict.get(region),
            ax=axs[0, 1],
            conf95=True,
        )
        gpl.plot_trace_werr(
            xs,
            np.mean(gen, axis=1),
            color=color_dict.get(region),
            ax=axs[0, 0],
            ls="dashed",
            conf95=True,
        )

        dec, xs, gen = timing_dict[region][o1_time_o2o1_key]
        gpl.plot_trace_werr(
            xs,
            np.mean(dec, axis=1),
            color=color_dict.get(region),
            ax=axs[1, 1],
            conf95=True,
        )
        gpl.plot_trace_werr(
            xs,
            np.mean(gen, axis=1),
            color=color_dict.get(region),
            ax=axs[1, 0],
            ls="dashed",
            conf95=True,
        )
        for i, j in u.make_array_ind_iterator(axs.shape):
            gpl.add_hlines(.5, axs[i, j])
            gpl.clean_plot(axs[i, j], j)


region_list = ('OFC', 'PCC', 'pgACC', 'VS', 'vmPFC')
def print_all_region_stats(data, region_list=region_list, **kwargs):
    print_data_stats(data, **kwargs)
    for r in region_list:
        print("--------- {} ---------".format(r))
        print_data_stats(data, region=r, **kwargs)

def print_data_stats(data, neur_thr=5, region=None):
    if region is not None:
        mask = np.array(list(
            region in np.unique(np.concatenate(r))
            for r in data["neur_regions"]
        ))
        data = data.session_mask(mask)
    n_sessions = len(data.data)
    n_neur_list = data["n_neurs"].to_numpy()
    n_neurons = np.sum(n_neur_list)
    print("total sessions: {}".format(n_sessions))
    print("median number of neurons: {} ({} - {})".format(
        np.median(n_neur_list), np.min(n_neur_list), np.max(n_neur_list)
    ))
    print("sessions with fewer than {} neurons: {}/{} (for {}/{} total neurons)".format(
        neur_thr,
        np.sum(n_neur_list < neur_thr),
        n_sessions,
        np.sum(n_neur_list[n_neur_list < neur_thr]),
        n_neurons,
    ))
    

def plot_stan_corr(fit_dict, pred_func, ax=None, align_func=mra.compute_corr,
                   **kwargs):
    mat, inter = mraux.make_fit_matrix(fit_dict)

    stim1, stim2 = mra._get_stim_reps(mat, inter, pred_func, n_pts=2, **kwargs)

    rs = np.expand_dims(align_func(stim1, stim2), 0)
    gpl.violinplot(rs, [0], ax=ax)

def plot_sklm_fits(fit_dict, pred_func, ax=None, use_regions=None,
                   n_pts=100, **kwargs):
    if use_regions is None:
        use_regions = ('OFC', 'PCC', 'pgACC', 'VS', 'vmPFC')

    mat = np.concatenate(list(v[0][0] for (r, _, _), v in fit_dict.items()
                               if r[0] in use_regions),
                          axis=1)
    inter = np.concatenate(list(v[0][1] for (r, _, _), v in fit_dict.items()
                                if r[0] in use_regions),
                           axis=1)
    stim1, stim2 = mra._get_stim_reps(mat, inter, pred_func, n_pts=n_pts,
                                  **kwargs)
    return gpl.plot_highdim_trace(stim1, stim2, ax=ax)
    
def plot_stan_fits(fit_dict, pred_func, plot_points=True, ms=.1, ax=None,
                   **kwargs):
    mat, inter = mraux.make_fit_matrix(fit_dict)

    stim1, stim2 = mra._get_stim_reps(mat, inter, pred_func, **kwargs)
    ax = gpl.plot_highdim_trace(stim1, stim2, plot_points=plot_points,
                                ms=ms, ax=ax)
    return ax

def plot_gp_fits(models, feats, acts, n_dims=10, axs=None, fwid=3, ppop_ind=0,
                 sig_te=None):
    if axs is None:
        f, axs = plt.subplots(n_dims, 1, figsize=(fwid, n_dims*fwid))
    f_i = feats[ppop_ind]
    for di in range(n_dims):
        ax = axs[di]
        if sig_te is not None and sig_te[ppop_ind, 0, di] > 0:
            ax.set_ylabel('*')
        
        m_ijk = models[ppop_ind, 0, di]
        a_ik = acts[ppop_ind, :, di]
        u_fs = np.unique(f_i[:, 1])
        act_mus = list(a_ik[f_i[:, 1] == uf] for uf in u_fs)

        mask1 = f_i[:, 0] == 0
        act1_mus = list(a_ik[np.logical_and(mask1, f_i[:, 1] == uf)]
                        for uf in u_fs)
        mask2 = f_i[:, 0] == 1
        act2_mus = list(a_ik[np.logical_and(mask2, f_i[:, 1] == uf)] 
                        for uf in u_fs)

        gp1_mus = list(m_ijk.predict(np.array([[0, uf]])) for uf in u_fs)
        gp2_mus = list(m_ijk.predict(np.array([[1, uf]])) for uf in u_fs)

        gpl.plot_trace_werr(u_fs, act_mus, jagged=True, ax=ax)
        l1 = gpl.plot_trace_werr(u_fs, act1_mus, jagged=True, ax=ax)
        l2 = gpl.plot_trace_werr(u_fs, act2_mus, jagged=True, ax=ax)

        ax.plot(u_fs, gp1_mus, color=l1[0].get_color(), ls='dashed')
        ax.plot(u_fs, gp1_mus, color='k', ls='dashed')
        ax.plot(u_fs, gp2_mus, color=l2[0].get_color(), ls='dashed')
    return axs

lin_color = np.array((254, 189, 42))/255
conj_color = np.array((161, 27, 155))/255
both_color = np.array((45, 113, 142))/255


def plot_submanifolds(*groups, ax=None, use_means=True,
                      colors=None, use_targeted=False,
                      rescale_arrays=True,
                      space_color=None,
                      linestyles=None,
                      only_first=10000):
    if ax is None:
        f, ax = plt.subplots(1, 1, subplot_kw={'projection': '3d'})
    if colors is None:
        colors = ((None, None),)*len(groups)
    if linestyles is None:
        linestyles = (None,)*len(groups)
    if rescale_arrays:
        new_groups = []
        for g in groups:
            g_arr = np.mean(np.stack(g, axis=0), axis=2)
            g_mu, trs = mra.scale_matrix(g_arr, ret_trs=True)
            new_groups.append(list(trs(g_i.T).T for g_i in g))
        groups = new_groups
    full_list = []
    list(full_list.extend(g) for g in groups[:only_first])
    if use_means:
        full_list = list(np.mean(fi, axis=1, keepdims=True) for fi in full_list)
    if use_targeted:
        dims = []
        for i, g in enumerate(groups):
            v1 = g[0] - g[1]
            v2 = g[2] - g[3]
            s1 = g[0] - g[2]
            s2 = g[1] - g[3]
            v1, v2, s1, s2 = list(np.mean(entry, axis=1, keepdims=True)
                                  for entry in (v1, v2, s1, s2))
            v12 = np.mean((v1, v2), axis=0)
            s12 = np.mean((s1, s2), axis=0)

            dims.extend([v12, s12])
        p = skd.PCA(3)
        dims = u.make_unit_vector(np.concatenate(dims, axis=1).T)
        # print(np.sum(dims[0]*dims[2]),
        #       np.sum(dims[0]*dims[4]))
        # print(np.sum(dims[1]*dims[3]),
        #       np.sum(dims[1]*dims[5]))
        p.fit_transform(dims)
    else:
        p = skd.PCA(3)
        full_arr = np.concatenate(full_list, axis=1).T
        p.fit_transform(full_arr)
    for i, g in enumerate(groups):
        high_left, low_left, high_right, low_right = (g_i.T for g_i in g)
        if colors[i] is not None:
            low_col = gpl.add_color_value(colors[i], -.2)
            high_col = gpl.add_color_value(colors[i], .2)
            use_colors = (high_col, low_col, high_col, low_col)
        else:
            use_colors = (colors[i],)*4
        # gpl.plot_highdim_points(high_left, low_left, high_right, low_right,
        #                         p=p, ax=ax, colors=use_colors)

        left_val = np.stack((np.mean(high_left, axis=0),
                             np.mean(low_left, axis=0)),
                            axis=0)
        right_val = np.stack((np.mean(high_right, axis=0),
                             np.mean(low_right, axis=0)),
                             axis=0)
        high_side = np.stack((np.mean(low_right, axis=0),
                             np.mean(low_left, axis=0)),
                             axis=0)
        low_side = np.stack((np.mean(high_right, axis=0),
                             np.mean(high_left, axis=0)),
                            axis=0)
        l_color, r_color = colors[i]
        gpl.plot_highdim_trace(high_side, low_side, ax=ax, color=space_color,
                               p=p, linestyle=linestyles[i])
        gpl.plot_highdim_trace(left_val, ax=ax, color=l_color,
                               p=p, linestyle=linestyles[i])
        gpl.plot_highdim_trace(right_val, ax=ax, color=r_color,
                               p=p, linestyle=linestyles[i])


def plot_single_rdm_dict(rdms, specific_ds, axs=None, fwid=3, p_thr=.05,
                         color=None):
    if axs is None:
        f, axs = plt.subplots(
            1, len(specific_ds), figsize=(fwid*len(specific_ds), fwid)
        )
    for i, (k, ds) in enumerate(specific_ds.items()):
        axs[i].hist(ds, density=True, color=color)
        gpl.add_vlines(0, axs[i])
        gpl.clean_plot(axs[i], 1)
        axs[i].set_title(k)
        if np.mean(ds > 0) > 1 - p_thr:
            mu = np.mean(ds)
            pt = axs[i].get_ylim()[1]
            axs[i].plot(mu, pt, '*', ms=5, color=color)


def visualize_full_resps(pts, ax=None, lh_color="blue", rh_color="red",
                         matched_color="gray", collapse_groups=None,
                         collapse_colors=None,
                         lh_inds=[2, 5], rh_inds=[1, 6],
                         matched_inds=[0, 3, 4, 7],
                         **kwargs):
    if ax is None:
        f, ax = plt.subplots(1, 1, subplot_kw={'projection': '3d'})
    if collapse_groups is None:
        collapse_groups = (
            ((1, 2), (5, 6)),
            ((1, 5), (2, 6)),
            ((1, 6), (2, 5)),
        )
    if collapse_colors is None:
        collapse_colors = (None,)*len(collapse_groups)
    if pts.shape[1] > 3:
        p = skd.PCA(3)
        pts = p.fit_transform(pts)

    for i, cg in enumerate(collapse_groups):
        color = collapse_colors[i]
        for indiv in cg:
            l = ax.plot(*pts[list(indiv)].T, color=color, **kwargs)
            color = l[0].get_color()
    
    ax.plot(*pts[lh_inds].T, 'o', color=lh_color, **kwargs)
    ax.plot(*pts[rh_inds].T, 'o', color=rh_color, **kwargs)
    ax.plot(*pts[matched_inds].T, 'o', color=matched_color, **kwargs)

            
def visualize_full_rdm(rdm, ax=None, lh_color="blue", rh_color="red",
                       matched_color="gray", collapse_groups=None,
                       collapse_colors=None,
                       lh_inds=[2, 5], rh_inds=[1, 6],
                       matched_inds=[0, 3, 4, 7],
                       **kwargs):
    m = skman.MDS(dissimilarity="precomputed", n_components=3)
    pts = m.fit_transform(rdm)
    return visualize_full_resps(*args, **kwargs)


def plot_group_rdm_dict(out_dict, axs=None, fwid=3):
    if axs is None:
        n_rows = len(list(out_dict.values())[0][2])
        n_cols = len(out_dict)
        f, axs = plt.subplots(
            n_rows, n_cols, figsize=(fwid*n_cols, fwid*n_rows), sharex="col",
        )
    for i, (region, dec_dict) in enumerate(out_dict.items()):
        plot_single_rdm_dict(dec_dict[0], dec_dict[2], axs=axs[:, i])
        if region is None:
            region = 'all'
        axs[-1, i].set_xlabel(region)


def plot_group_decoding_dict(out_dict, theor_dict=None, axs=None, fwid=3):
    if axs is None:
        n_rows = len(list(out_dict.values())[0])
        n_cols = len(out_dict)
        f, axs = plt.subplots(
            n_rows, n_cols, figsize=(fwid*n_cols, fwid*n_rows), sharey="row",
        )
    if theor_dict is None:
        theor_dict = {}
    for i, (region, dec_dict) in enumerate(out_dict.items()):
        td = theor_dict.get(region)
        print(region)
        plot_single_decoding_dict(dec_dict, axs=axs[:, i], theor_dict=td)
        if region is None:
            region = 'all'
        axs[-1, i].set_xlabel(region)
    return axs


def plot_single_decoding_dict(out_dict, theor_dict=None, axs=None, fwid=3):
    if axs is None:
        f, axs = plt.subplots(
            len(out_dict), 1, figsize=(fwid, fwid*len(out_dict)), sharey="row",
        )
    if theor_dict is None:
        theor_dict = {}
    for i, (k, outs) in enumerate(out_dict.items()):
        td = theor_dict.get(k)
        # print(k)
        all_samps = []
        for j, out in enumerate(outs):
            dec_samps = np.mean(out[0][..., -1], axis=1)
            all_samps.append(dec_samps)

            gpl.violinplot([dec_samps], [j], ax=axs[i])
            if len(out) > 2:
                gen_samps = np.mean(out[-1][..., -1], axis=1)
                gpl.violinplot([gen_samps], [j], ax=axs[i], color='b')
            if td is not None:
                pred = 1 - np.mean(td[j][1], axis=1, keepdims=True)
                # pred_bind = 1 - np.mean(td[j][0], axis=1, keepdims=True)
                # print(u.conf_interval(pred_bind, withmean=True, perc=90))
                if np.nanvar(pred, axis=0) > 1e-5:
                    gpl.plot_trace_werr([j], pred,
                                        ax=axs[i],
                                        fill=False,
                                        conf95=True,
                                        color='b')
        mu_samps = np.mean(all_samps, axis=0)
        print("{}: {} - {}".format(
            k, *u.conf_interval(mu_samps, withmean=True, perc=90)[:, 0]
        ))
        gpl.add_hlines(.5, axs[i])
        axs[i].set_title(k)
        gpl.clean_plot(axs[i], 0)
        axs[i].set_xticks(range(len(outs)))
        axs[i].set_ylabel('decoding performance')
    return axs


def plot_linear_continuous(thetas=None, sigmas=None, n_samps=1000, ax=None,
                           cmap="Purples", buff=.3, fwid=1.5):
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(fwid, fwid))
    rng = np.random.default_rng()
    if thetas is None:
        thetas = np.linspace(0, np.pi/2, 100)
    if sigmas is None:
        sigmas = np.linspace(0.05, .3, 5)
    cmap = plt.get_cmap(cmap)
    colors = cmap(np.linspace(buff, 1 - buff, len(sigmas)))

    errs = np.zeros((len(thetas), len(sigmas)))
    err_theor = np.zeros_like(errs)
    rs = np.zeros_like(thetas)
    for i, theta in enumerate(thetas):
        a = np.array([[1, 0]]).T
        b = np.array([[np.cos(theta), np.sin(theta)]]).T

        A = np.concatenate((a, b), axis=1)

        stim = rng.normal(size=(n_samps, 2))
        rep = stim @ A
        A_lst = np.linalg.lstsq(rep, stim)[0]
        rs[i] = a.T @ b

        for j, sig in enumerate(sigmas):
            # err_theor[i, j] = (1 + (np.cos(theta)/np.sin(theta))**2
            #                    + 1/np.sin(theta)**2)*sig**2
            err_theor[i, j] = (2*sig**2)/(np.sin(theta)**2)
            rep = stim @ A + rng.normal(0, sig, size=(n_samps, 2))
            dec = rep @ A_lst
            errs[i, j] = np.mean(np.sum((dec - stim)**2, axis=1))

    for j, sig in enumerate(sigmas):
        ax.plot(rs, errs[:, j], color=colors[j],
                label=r'$\sigma$ = {:.2f}'.format(sig))
        gpl.plot_trace_werr(rs, err_theor[:, j], ax=ax, color=colors[j],
                            plot_outline=True, linestyle='dashed')
    # ax.set_ylim([0, 1])
    ax.legend(frameon=False)
    ax.set_yscale('log')
    yb = ax.get_ylim()[0]
    _ = ax.set_ylim([yb, 1])
    ax.set_ylabel('decoder error')
    ax.set_xlabel('subspace correlation (r)')
    return ax


def plot_bias(data,
              decode_var='subj_ev',
              order_choice_field='choice offer 1 (==1) or 2 (==0)',
              side_choice_field='Choice left (==1) or Right (==0)',
              color_dict=None,
              ax=None,
              **kwargs):
    if ax is None:
        f, ax = plt.subplots(1, 1)
    # o1_target = '{} offer 1'.format(decode_var)
    # o2_target = '{} offer 2'.format(decode_var)
    # l_target = '{}_left'.format(decode_var)
    # r_target = '{}_right'.format(decode_var)
    order_bias = list(np.mean(x) for x in data[order_choice_field] == 1)
    side_bias = list(np.mean(x) for x in data[side_choice_field] == 1)
    region = list(x[0][0] for x in data['neur_regions'])
    if color_dict is None:
        color_dict = {}
    for i, r in enumerate(region):
        ax.plot(order_bias[i], side_bias[i], 'o', color=color_dict.get(r), **kwargs)


def plot_bhv_dec_hist(
        dec,
        gen,
        regions,
        color_dict=None,
        ax=None,
        fwid=2,
        t_ind=-1,
        minor=0,
        u_rs=None,
):
    regions = np.array(regions)
    if u_rs is None:
        u_rs = np.unique(regions)
    
    n_regions = len(u_rs)
    dec = dec[..., t_ind]
    gen = gen[..., t_ind]
    if ax is None:
        f, ax = plt.subplots(1, 1)
    for i, ur in enumerate(u_rs):
        mask = ur == regions
        if sum(mask) > 0:
            diffs = dec[mask] - gen[mask]
            avg_diff = np.mean(diffs, axis=0)
            gpl.violinplot([avg_diff], [i + minor], ax=ax)
            # print(ur, np.mean(avg_diff > 0))
    gpl.add_hlines(0, ax)
    ax.set_xticks(range(n_regions))
    ax.set_xticklabels(u_rs)
        
def plot_bhv_dec(dec, gen, regions, color_dict=None, ax=None, animals=None,
                 targ_animal=None, add_lines=True, t_ind=-1, dec_pops=None,
                 gen_pops=None, **kwargs):
    if ax is None:
        f, ax = plt.subplots(1, 1)
    if color_dict is None:
        color_dict = {}

    dec = dec[..., t_ind:]
    gen = gen[..., t_ind:]
    dec_pts = np.nanmean(dec, axis=1)
    gen_pts = np.nanmean(gen, axis=1)
    diff_pts = np.nanmean(dec - gen, axis=1)
    pt_save = {}
    cv_save = {}
    for i, r in enumerate(regions):
        color = color_dict.get(r)
        if animals is not None and targ_animal is not None:
            include = animals[i] == targ_animal
        else:
            include = True

        if include:
            # x_err = gpl.std(dec[i])[..., -1:]
            # y_err = gpl.sem(gen[i])[..., -1:]
            dec_i, gen_i, diffs_i = pt_save.get(r, ([], [], []))
            dec_i.append(dec_pts[i])
            gen_i.append(gen_pts[i])
            diffs_i.append(diff_pts[i])
            pt_save[r] = (dec_i, gen_i, diffs_i)
            dec_cv_i, gen_cv_i = cv_save.get(r, ([], []))
            dec_cv_i.append(dec[i])
            gen_cv_i.append(gen[i])
            cv_save[r] = (dec_cv_i, gen_cv_i)
            gpl.plot_trace_werr(dec_pts[i], gen_pts[i],
                                fill=False,  # err=y_err, err_x=x_err,
                                ax=ax,
                                color=color, points=True,
                                **kwargs)
    for r, (dec, gen, _) in pt_save.items():
        dec_cv, gen_cv = cv_save[r]
        print(r, sts.ttest_1samp(pt_save[r][2], 0, alternative="greater"))
        for i, dcv_i in enumerate(dec_cv):
            pass
        # print(r, list(sts.ttest_ind(dec_cv[i], gen_cv[i], equal_var=False)
        #               for i in range(len(dec_cv))))
        print(r, len(dec))
        gpl.plot_trace_werr(np.expand_dims(dec, 1),
                            np.expand_dims(gen, 1),
                            ax=ax,
                            fill=False,
                            color=color_dict.get(r),
                            points=True,
                            ms=5,
                            **kwargs)
    for r1, r2 in it.combinations(pt_save.keys(), 2):
        print(r1, r2, sts.ttest_ind(pt_save[r1][2], pt_save[r2][2]))

    if add_lines:
        gpl.add_hlines(.5, ax)
        gpl.add_vlines(.5, ax)
        ax.set_aspect('equal')
    return ax


def plot_dist_mat(rdm, labels=None, ax=None, highlights=None,
                  highlight_colors=None, fig=None, **kwargs):
    if highlight_colors is None:
        highlight_colors = {}
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    if labels is None:
        labels = ('',)*rdm.shape[0]
    xs = np.arange(rdm.shape[0])
    m = gpl.pcolormesh(xs, xs, rdm, ax=ax, **kwargs)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=90, family="monospace")
    ax.set_yticks(xs)
    ax.set_yticklabels(labels, family="monospace")
    ax.set_aspect("equal")
    if highlights is not None:
        for k, pts in highlights.items():
            color = highlight_colors.get(k)
            for pt in pts:
                edge = .08
                xy = (pt[0] - .5 + edge, pt[1] - .5 + edge)
                rect = plt.Rectangle(
                    xy,
                    1 - 2*edge,
                    1 - 2*edge,
                    fill=False,
                    edgecolor=color,
                    lw=2,
                )
                ax.add_artist(rect)
    if fig is not None:
        fig.colorbar(m, ax=ax, label='distance', ticks=[0, .1, .2])


def plot_dists_region(out_rdm_dict, use_pairs, regions, c_colors, axs=None,
                      labels=None):
    if axs is None:
        f, axs = plt.subplots(1, 2)
    if labels is None:
        labels = (None,)*c_colors
    ax_r, ax_a = axs
    minor_ticks = np.linspace(-.2, .2, len(c_colors))
    for i, pair in enumerate(use_pairs):
        ds = []
        for r in regions[:-1]:
            ds_r = out_rdm_dict[r][2][pair]
            ds.append(ds_r)
            d_interv = u.conf_interval(ds_r, withmean=True, perc=90)[:, 0]
            print(
                "{}: {} interval: {}".format(r, labels[i], d_interv)
            )

        xs = np.arange(len(regions)) + minor_ticks[i]
        gpl.violinplot(ds, xs, ax=ax_r,
                       color=(c_colors[i],)*len(ds))

    gpl.clean_plot(ax_r, 0)
    ax_r.set_ylabel('distance')

    for (i1, i2) in it.combinations(range(len(use_pairs)), 2):
        p1, p2 = use_pairs[i1], use_pairs[i2]
        if labels is not None:
            l1, l2 = labels[i1], labels[i2]
        else:
            l1, l2 = p1, p2
        for r in regions:
            d1 = out_rdm_dict[r][2][p1]
            d2 = out_rdm_dict[r][2][p2]
            interv = u.conf_interval(d1 - d2, withmean=True)[:, 0]
            print(
                "{}: {} - {} interval: {}".format(r, l1, l2, interv)
            )

    gpl.clean_plot_bottom(ax_r, keeplabels=True)
    gpl.add_hlines(0, ax_r)

    ax_r.set_xticks(range(len(regions[:-1])))
    ax_r.set_xticklabels(regions[:-1])

    handles = []
    for i, pair in enumerate(use_pairs):
        ds = [out_rdm_dict["all"][2][pair]]
        colors = [c_colors[i]]
        tick = minor_ticks[i]
        p = gpl.violinplot(ds, [tick], ax=ax_a,
                           color=colors, labels=[labels[i]])
        gpl.clean_plot(ax_a, 0)
        if len(labels[i]) > 0:
            used_color = p["cmedians"].get_color()
            patch = patches.Patch(color=used_color)
            handles.append(patch)
        d_interv = u.conf_interval(ds, withmean=True, perc=90)[:, 0]
        print(
            "{}: {} interval: {}".format("all", labels[i], d_interv)
        )
    ax_a.legend(handles, labels, frameon=False)

    gpl.clean_plot_bottom(ax_a, keeplabels=True)
    gpl.add_hlines(0, ax_a)

    ax_a.set_xticks([0])
    ax_a.set_xticklabels(["all"])
    yl0 = ax_a.get_ylim()[0]
    yl1 = ax_r.get_ylim()[1]
    ax_r.set_ylim((-.05, .1))
    ax_a.set_ylim((-.1, .3))


def plot_choice_sensitivity(
        data,
        dead_perc=30, c1_targ=2,
        c2_targ=3,
        decode_var='subj_ev',
        use_split_dec=None,
        order_choice_field='choice offer 1 (==1) or 2 (==0)',
        side_choice_field='Choice left (==1) or Right (==0)',
        use_order=True,
        colors=None,
        ax=None,
        max_abs_diff=4,
        min_abs_diff=0,
        **kwargs
):
    if ax is None:
        f, ax = plt.subplots(1, 1)
    if colors is None:
        colors = (None,)*3
    if use_order:
        l1_templ = '{} offer 1'
        l2_templ = '{} offer 2'
        choice_field = order_choice_field
    else:
        l1_templ = '{}_left'
        l2_templ = '{}_right'
        choice_field = side_choice_field
    o1_target = l1_templ.format(decode_var)
    o2_target = l2_templ.format(decode_var)
    out = mra._compute_masks(data, o1_target, o2_target, dead_perc=dead_perc,
                             use_split_dec=use_split_dec, c1_targ=c1_targ,
                             c2_targ=c2_targ)
    mask_o1_h, mask_o1_l, mask_o2_h, mask_o2_l = out

    mask_high = mask_o1_h.rs_and(mask_o2_h)
    mask_low = mask_o1_l.rs_and(mask_o2_l)
    mask_highlow = mask_o1_h.rs_and(mask_o2_l)
    mask_lowhigh = mask_o1_l.rs_and(mask_o2_h)
    mask_lhhl = mask_highlow.rs_or(mask_lowhigh)
    use_masks = (mask_lhhl, mask_low, mask_high)
    labels = ("mixed", "low only", "high only")

    corr = []
    for i, mask in enumerate(use_masks):
        if mask is None:
            use_choice = data[choice_field]
            use_diff = data[o1_target] - data[o2_target]
        else:
            data_mask = data.mask(mask)
            o1_val = data_mask[o1_target]
            o2_val = data_mask[o2_target]

            use_diff = o1_val - o2_val
            use_choice = data_mask[choice_field]
        use_choice = np.concatenate(use_choice)
        use_diff = np.concatenate(use_diff)
        diff_mask = np.logical_and(min_abs_diff <= np.abs(use_diff),
                                   np.abs(use_diff) <= max_abs_diff)
        use_choice = use_choice[diff_mask]
        use_diff = use_diff[diff_mask]

        corr.append(np.mean((use_choice > .5) == (use_diff > 0)))

        xs, ys = gpl.digitize_vars(use_diff, use_choice)
        gpl.plot_trace_werr(xs, ys, jagged=True, ax=ax, label=labels[i],
                            color=colors[i])
    ax.set_xlabel(r'$\Delta$' + ' expected value\n(offer 1 - 2)')
    ax.set_ylabel('p(choose offer 1)')
    return corr


def plot_distances(
        data,
        div_dict=mra.full_div_dict,
        use_other=True,
        other_key='side only',
        region_order=('OFC', 'PCC', 'pgACC', 'VS', 'vmPFC', 'all'),
        minor_move=.1,
        lin_color=lin_color,
        conj_color=conj_color,
        both_color=both_color,
        lin_label='linear', conj_label='mixed',
        nonlin_label='nonlinear',
        both_label='both', axs=None, fwid=3,
        add_val=.2,
        mult_nl=True
):
    if use_other:
        div_dict = div_dict[other_key]
    else:
        div_dict = div_dict['non other']
    if axs is None:
        n_panels = len(div_dict)
        f, axs = plt.subplots(1, n_panels, figsize=(fwid*n_panels, fwid),
                              sharex=True, sharey=True)
    else:
        f = None
    max_val = 0
    for i, (div_name, dks) in enumerate(div_dict.items()):
        for j, region in enumerate(region_order):
            dl, dn, sig, sem = [], [], [], []
            for dk in dks:
                res = data[region][dk][0, 0]
                if len(res.dtype) > 0:
                    dl.append(np.mean(res['d_l'][0, 0], axis=1, keepdims=True))
                    dn.append(np.mean(res['d_n'][0, 0], axis=1, keepdims=True))
                    sig.append(np.mean(res['sigma'][0, 0], axis=1, keepdims=True))
                    sem.append(np.mean(res['sem'][0, 0], axis=1, keepdims=True))

            dl = np.concatenate(dl, axis=0)
            dn = np.concatenate(dn, axis=0)
            if mult_nl:
                dn = dn*np.sqrt(2)
            sig = np.concatenate(sig, axis=0)
            sem = np.concatenate(sem, axis=0)
            dn_ci_high, dn_ci_low = u.conf_interval(dn, perc=90, withmean=True)[:, 0]
            print('{}, {}, nonlinear: {} - {}'.format(
                div_name, region, dn_ci_low, dn_ci_high))
            dl_ci_high, dl_ci_low = u.conf_interval(dl, perc=90, withmean=True)[:, 0]
            print('{}, {}, linear: {} - {}'.format(
                div_name, region, dl_ci_low, dl_ci_high))

            if j == len(region_order) - 1:
                use_lin_label = lin_label
                use_nonlin_label = nonlin_label
            else:
                use_lin_label = ''
                use_nonlin_label = ''
            gpl.violinplot(dl.T, [j - minor_move/2], ax=axs[i],
                           color=[lin_color], showextrema=False,
                           showmedians=True, labels=[lin_label])
            gpl.violinplot(dn.T, [j + minor_move/2], ax=axs[i],
                           color=[conj_color], showextrema=False,
                           showmedians=True, labels=[nonlin_label])
            max_val = np.max([max_val, np.max(dn), np.max(dl)])
        axs[i].set_xticks(range(len(region_order)))
        axs[i].set_xticklabels(region_order, rotation=45)
        gpl.clean_plot(axs[i], i)
    axs[0].set_yticks([0, 1, 2])
    axs[0].set_ylim([0, max_val + add_val])
    axs[1].set_ylim([0, max_val + add_val])
    
    axs[0].set_ylabel('estimated distance')
    return f, axs

# def plot_correlations(path, epoch_order=('E1_E', 'E1_M', 'E2_E', 'E2_M'),
#                       epoch_compare=(('E1_{}', 'E1_{}'), ('E2_{}', 'E2_{}'),
#                                      ('E1_{}', 'E2_{}'))
#                       epoch_names=('within offer 1', 'within offer 2',
#                                    'across offer 1 and 2'),
#                       axs=None, fwid=3, epochs_out=None,
#                       region_order=('OFC', 'PCC', 'pgACC', 'VS', 'vmPFC'),
#                       minor_move=.18, bar_width=.2,
#                       use_time='E',
#                       **kwargs):
#     if axs is None:
#         f, axs = plt.subplots(1, 1,
#                               figsize=(fwid, fwid*.8),
#                               sharex=True, sharey=True)
    
#     if epochs_out is None:
#         epochs_out = {}
#         for epoch in epoch_order:
#             left_val = mra.apply_coeff_func(path, mra.get_left_val_samples,
#                                             **kwargs)
#             right_val = mra.apply_coeff_func(path, mra.get_right_val_samples,
#                                              **kwargs)
#             epochs_out[epoch] = (left_val, right_val)
#     for i, epoch in enumerate(epoch_order):
#         lin, conj, both = epochs_out[epoch]
#         for j, region in enumerate(region_order):
#             lin_reg = np.array(list(np.nanmean(x) for x in lin[region]))
#             conj_reg = np.array(list(np.nanmean(x) for x in conj[region]))
#             both_reg = np.array(list(np.nanmean(x) for x in both[region]))
#             b = minor_move*len(lin_reg)
#             xs = np.linspace(-b/2, b/2, len(lin_reg)) + j
            
#             l_lin = axs[i].bar(xs, lin_reg, width=bar_width, color=lin_color,
#                                 label=lin_label)
#             l_con = axs[i].bar(xs, conj_reg, bottom=lin_reg, width=bar_width,
#                                color=conj_color, label=conj_label)
#             l_bot = axs[i].bar(xs, both_reg,
#                                 bottom=lin_reg + conj_reg, width=bar_width,
#                                 color=both_color, label=both_label)
#         axs[i].set_xticks(np.arange(len(region_order)))
#         axs[i].set_xticklabels(region_order, rotation=60)
#         axs[i].set_title(epoch_names[i])
#         gpl.clean_plot(axs[i], i)
#     axs[-1].legend(handles=(l_lin, l_con, l_bot), frameon=False)
#     axs[0].set_ylabel('proportion of\nneurons')
#     return epochs_out, (f, axs)
    
def visualize_regr_means(p1, p2, ax=None, fwid=3, skip=1,
                         grey_color=(.7, .7, .7), alpha=.5,
                         vis=None, pop_ind=0):
    if ax is None:
        f = plt.figure(figsize=(fwid, fwid))
        ax = f.add_subplot(1, 1, 1, projection='3d')

    pts = np.concatenate((p1, p2), axis=1)
    mask = np.all(~np.isnan(pts), axis=1)
    p = skd.PCA()
    p.fit(pts[mask].T)
    p1_low = p.transform(p1[mask].T)
    p2_low = p.transform(p2[mask].T)

    for i in range(0, len(p1_low), skip):
        ax.plot([p1_low[i, 0], p2_low[i, 0]],
                [p1_low[i, 1], p2_low[i, 1]],
                [p1_low[i, 2], p2_low[i, 2]],
                color=grey_color, alpha=alpha)
        
    ax.plot(p1_low[:, 0], p1_low[:, 1], p1_low[:, 2])
    ax.plot(p2_low[:, 0], p2_low[:, 1], p2_low[:, 2])
    if vis is not None:
        if vis.shape[1] > p.n_features_:
            vis = vis[:, mask]
        v_low = p.transform(vis)
        ax.plot(v_low[:, 0], v_low[:, 1], v_low[:, 2], 'o')
                           
                
def plot_selectivity_type(path, epoch_order=('E1_E', 'E1_M', 'E2_E', 'E2_M'),
                          epoch_names=('offer 1 on', 'offer 1 delay',
                                       'offer 2 on', 'offer 2 delay'),
                          axs=None, fwid=3, epochs_out=None,
                          region_order=('OFC', 'PCC', 'pgACC', 'VS', 'vmPFC'),
                          minor_move=.18, bar_width=.2,
                          lin_color=lin_color,
                          conj_color=conj_color,
                          both_color=both_color,
                          lin_label='linear', conj_label='mixed',
                          both_label='both',
                          **kwargs):
    if axs is None:
        f, axs = plt.subplots(1, len(epoch_order),
                              figsize=(fwid*len(epoch_order), fwid*.8),
                              sharex=True, sharey=True)
    if epochs_out is None:
        epochs_out = {}
        for epoch in epoch_order:
            lin = mra.apply_coeff_func(path, mra.get_lin_select, **kwargs)
            conj = mra.apply_coeff_func(path, mra.get_conj_select, **kwargs)
            both = mra.apply_coeff_func(path, mra.get_mult_select, **kwargs)        
            epochs_out[epoch] = (lin, conj, both)
    for i, epoch in enumerate(epoch_order):
        lin, conj, both = epochs_out[epoch]
        for j, region in enumerate(region_order):
            lin_reg = np.array(list(np.nanmean(x) for x in lin[region]))
            conj_reg = np.array(list(np.nanmean(x) for x in conj[region]))
            both_reg = np.array(list(np.nanmean(x) for x in both[region]))
            b = minor_move*len(lin_reg)
            xs = np.linspace(-b/2, b/2, len(lin_reg)) + j
            
            l_lin = axs[i].bar(xs, lin_reg, width=bar_width, color=lin_color,
                                label=lin_label)
            l_con = axs[i].bar(xs, conj_reg, bottom=lin_reg, width=bar_width,
                               color=conj_color, label=conj_label)
            l_bot = axs[i].bar(xs, both_reg,
                                bottom=lin_reg + conj_reg, width=bar_width,
                                color=both_color, label=both_label)
        axs[i].set_xticks(np.arange(len(region_order)))
        axs[i].set_xticklabels(region_order, rotation=60)
        axs[i].set_title(epoch_names[i])
        gpl.clean_plot(axs[i], i)
    axs[-1].legend(handles=(l_lin, l_con, l_bot), frameon=False)
    axs[0].set_ylabel('proportion of\nneurons')
    return epochs_out, (f, axs)
    
def plot_dists(x, pred_vals, 
               ax=None, fwid=3,
               minor_move=.18, bar_width=.2,
               lin_color=np.array((254, 189, 42))/255,
               conj_color=np.array((161, 27, 155))/255,
               both_color=np.array((45, 113, 142))/255,
               lin_label='linear', conj_label='nonlinear',
               both_label='both',
               **kwargs):
    if ax is None:
        f, ax = plt.subplots(1, 1,
                              figsize=(fwid, fwid))
    sigma = pred_vals['sigma'][0, 0]
    dl = pred_vals['d_l'][0, 0]
    dn = pred_vals['d_n'][0, 0]
    dl_mu = np.mean(dl)
    dn_mu = np.mean(dn)
    dl_std = np.std(dl)
    dn_std = np.std(dn)
    total_dist_mu = np.mean(dl + dn)
    xs = np.array([x])
    l_lin = ax.bar(xs, dl_mu/total_dist_mu, width=bar_width, color=lin_color,
                   label=lin_label)
    l_con = ax.bar(xs, dn_mu/total_dist_mu, bottom=dl_mu/total_dist_mu,
                   width=bar_width,
                   color=conj_color, label=conj_label)
    # l_bot = ax.bar(xs, both_reg,
    #                    bottom=lin_reg + conj_reg, width=bar_width,
    #                    color=both_color, label=both_label)
    return l_lin, l_con

def plot_dec_gen_pred(pred_vals, dec_vals, x_val, ax=None, minor_tick=.2,
                      color=None, symbol_list=('o', 's', '*'), **kwargs):
    
    if ax is None:
        f, ax = plt.subplots(1, 1)
    
    p_ccgp = np.mean(pred_vals['pred_ccgp'][0, 0], axis=1, keepdims=True)
    dec_perf = np.mean(dec_vals['dec'][0, 0][..., -1], axis=1, keepdims=True)
    dec_gen = np.mean(dec_vals['gen'][0, 0][..., -1], axis=1, keepdims=True)

    gpl.violinplot(dec_perf.T, [x_val - minor_tick],
                   color=[color], ax=ax, showextrema=False, showmedians=False,
                   markerstyles=[symbol_list[0]])
    
    gpl.violinplot(dec_gen.T, [x_val],
                   color=[color], ax=ax, showextrema=False, showmedians=False,
                   markerstyles=[symbol_list[1]])
    gpl.violinplot(p_ccgp.T, [x_val + minor_tick],
                   color=[color], ax=ax, showextrema=False, showmedians=False,
                   markerstyles=[symbol_list[2]])


def plot_data_pred(
        pred_vals,
        dec_vals,
        n_feats=2,
        n_vals=2,
        axs=None,
        color=None,
        line_alpha=.2,
        label='',
        comp_pred=None,
        ax_bin=None,
        ax_gen=None,
        ax_comb=None,
        print_differences=False,
):
    if axs is None and ax_bin is None and ax_gen is None and ax_comb is None:
        f, axs = plt.subplots(1, 3)        
    if axs is not None and len(axs) == 3:
        (ax_bin, ax_gen, ax_comb) = axs

    dl = np.mean(pred_vals['d_l'][0, 0], axis=1)
    dn = np.mean(pred_vals['d_n'][0, 0], axis=1)
    sigma = np.mean(pred_vals['sigma'][0, 0], axis=1)
    sem = np.mean(pred_vals['sem'][0, 0], axis=1)
    p_ccgp = np.mean(pred_vals['pred_ccgp'][0, 0], axis=1, keepdims=True)
    p_bind = np.mean(pred_vals['pred_bin'][0, 0], axis=1, keepdims=True)

    dec_gen = np.mean(dec_vals['gen'][0, 0][..., -1], axis=1, keepdims=True)

    dn = np.sqrt(2)*dn

    if comp_pred is not None and print_differences:
        comp_ccgp = np.mean(comp_pred['pred_ccgp'][0, 0], axis=1, keepdims=True)
        comp_bind = np.mean(comp_pred['pred_bin'][0, 0], axis=1, keepdims=True)
        high, low = u.conf_interval(p_ccgp - comp_ccgp, withmean=True)[:, 0]
        p = np.mean(p_ccgp - comp_ccgp > 0)
        print("ccgp: {low:.2f} - {high:.2f}, p = {p}".format(low=low, high=high, p=p))
        high, low = u.conf_interval(p_bind - comp_bind, withmean=True)[:, 0]
        p = np.mean(p_bind - comp_bind > 0)
        print("bind: {low:.2f} - {high:.2f}, p = {p}".format(low=low, high=high, p=p))

    pwr = np.mean(dl**2 + dn**2) 
    ts = np.linspace(.01, 1, 100)

    sigma_m = np.mean(sigma)
    sem = np.mean(sem)
    r1, gen_err = mrt.vector_corr_ccgp(pwr, ts, sigma=sigma_m, sem=sem)
    r1, bin_err = mrt.vector_corr_swap(pwr, n_feats, n_vals,  ts,
                                       sigma=sigma_m**2,
                                       sem=sem)

    r_emp = np.expand_dims(dl**2/(dn**2 + dl**2 + sem**2), 1)
    if ax_gen is not None:
        l = ax_gen.plot(r1, gen_err, color=color, alpha=line_alpha)
        color = l[0].get_color()
        # ax_gen.plot(r_emp, 1 - p_ccgp, 'o', color=color)
        gpl.plot_trace_werr(r_emp, 1 - p_ccgp, color=color, conf95=True, fill=False,
                            points=True, ax=ax_gen, label=label)
        # ax_gen.plot(r_emp, 1 - dec_gen, 'o', color=color)
        gpl.plot_trace_werr(r_emp, 1 - dec_gen, color='k', ms=3, conf95=True, fill=False,
                            points=True, ax=ax_gen)
        gpl.plot_trace_werr(r_emp, 1 - dec_gen, color=color, conf95=True, fill=False,
                            points=True, ax=ax_gen)
        ax_gen.legend(frameon=False)


    if ax_bin is not None:
        ax_bin.plot(r1, bin_err, color=color, alpha=line_alpha)
        gpl.plot_trace_werr(r_emp, 1 - p_bind, color=color, conf95=True, fill=False,
                            points=True, ax=ax_bin)
        # ax_bin.plot(r_emp, 1 - p_bind, 'o', color=color)

    if ax_comb is not None:
        ax_comb.plot(gen_err, bin_err, color=color, alpha=line_alpha)
        gpl.plot_trace_werr(1 - p_ccgp, 1 - p_bind, color=color, conf95=True,
                            fill=False, points=True, ax=ax_comb)
        # ax_comb.set_xscale('log')
        # ax_comb.set_yscale('log')
    
def plot_3d_fit(mus, ax=None, masks='time', colors=('r', 'g', 'b'),
                cmap_name='PuOr'):
    if ax is None:
        f = plt.figure()
        ax = f.add_subplot(1, 1, 1, projection='3d')

    cmap = plt.get_cmap(cmap_name)
    p = skd.PCA(3)
    mus_rep = p.fit_transform(mus)

    if masks == 'time':
        pos_o1 = np.array([2, 4, 7, 8]) - 1
        neg_o1 = np.array([1, 3, 5, 6]) - 1
        
        pos_o2 = np.array([3, 4, 6, 8]) - 1
        neg_o2 = np.array([1, 2, 5, 7]) - 1
        
        pos_side = np.array([5, 7, 8, 6]) - 1
        neg_side = np.array([1, 2, 4, 3]) - 1
        
        pns = ((pos_o1, neg_o1), (pos_o2, neg_o2), (pos_side, neg_side))

        high = np.array([3, 6]) - 1
        equal = np.array([1, 4, 5, 8]) - 1
        low = np.array([2, 7]) - 1
        label = 'epoch {}'
        flip = 'side'
    if masks == 'space':
        pos_left = np.array([2, 4, 6, 8]) - 1
        neg_left = np.array([1, 3, 5, 7]) - 1
        
        pos_right = np.array([3, 4, 7, 8]) - 1
        neg_right = np.array([1, 2, 5, 6]) - 1
        
        pos_epoch = np.array([5, 6, 8, 7]) - 1
        neg_epoch = np.array([1, 2, 4, 3]) - 1

        pns = ((pos_left, neg_left), (pos_right, neg_right), (pos_epoch, neg_epoch))

        high = np.array([3, 7]) - 1
        equal = np.array([1, 4, 5, 8]) - 1
        low = np.array([2, 6]) - 1
        label = 'side {}'
        flip = 'epoch'
    out = mra.compute_within_across_corr(mus, np.array(pns))
    # out = None
    for i, (pos, neg) in enumerate(pns):
        rep_mask = np.stack((mus_rep[pos], mus_rep[neg]), axis=1)
        for j, line in enumerate(rep_mask):
            if j == 0:
                if i < 2:
                    label_str = label.format(i + 1)
                else:
                    label_str = flip
            else:
                label_str = ''
            l = ax.plot(*line.T, color=colors[i],
                        label=label_str)
    ax.plot(*mus_rep[high].T, 'o', color=cmap(.99), ms=20,
            label=label.format(1) + ' higher')
    ax.plot(*mus_rep[equal].T, 'o', color=cmap(.5), ms=20, label='neutral')
    ax.plot(*mus_rep[low].T, 'o', color=cmap(0), ms=20,
            label=label.format(2) + ' higher')
    ax.legend(frameon=False)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    return ax, out

def plot_lin_nonlin_schem(line_pts=2, sep_dist=.5, val_dist=1, nonlin_dist=.2,
                          ax=None, fwid=5, lin1_col=(.1, .5, .1),
                          lin2_col=(.1, .8, .1),
                          nonlin_col=(.8, .1, .1), comb1_col=(.5, .7, .1),
                          comb2_col=(.7, .5, .1),
                          ms=8, lin_alpha=.3, seed=None):
    if ax is None:
        f = plt.figure(figsize=(fwid, fwid))
        ax = f.add_subplot(1, 1, 1, projection='3d')
    lin_pts_1 = np.array([[-sep_dist/2, -val_dist/2],
                          [-sep_dist/2, val_dist/2]])
    lin_pts_2 = np.array([[sep_dist/2, -val_dist/2],
                          [sep_dist/2, val_dist/2]])

    # figure out actual rotation matrix do 45 by 45
    rot = np.array([[0, -1],
                    [1, 0],
                    [1, -1]])
    rot = u.make_unit_vector(rot.T).T
    lin_pts_1 = np.dot(rot, lin_pts_1.T).T
    lin_pts_2 = np.dot(rot, lin_pts_2.T).T
    rng = np.random.default_rng(seed)
    nonlin_perts = u.make_unit_vector(rng.normal(0, nonlin_dist, (4, 3)))
    nonlin_perts = nonlin_perts*nonlin_dist

    lin_pts_all = np.concatenate((lin_pts_1, lin_pts_2), axis=0)
    lin_pts_c1 = np.stack((lin_pts_1[0], lin_pts_2[0]), axis=0)
    lin_pts_c2 = np.stack((lin_pts_1[1], lin_pts_2[1]), axis=0)
    
    ax.plot(*lin_pts_1.T, color=lin1_col, alpha=lin_alpha)
    ax.plot(*lin_pts_2.T, color=lin1_col, alpha=lin_alpha, label='value')
    ax.plot(*lin_pts_c1.T, color=lin2_col, alpha=lin_alpha, label='side')
    ax.plot(*lin_pts_c2.T, color=lin2_col, alpha=lin_alpha)
    
    # ax.plot(*lin_pts_2.T, color=lin_col, alpha=lin_alpha)
    # ax.plot(*lin_pts_1.T, 'o', color=lin_col, ms=ms, alpha=lin_alpha)
    # ax.plot(*lin_pts_2.T, 'o', color=lin_col, ms=ms, alpha=lin_alpha)

    for i, lp_i in enumerate(lin_pts_all):
        pert_plot = np.stack((lp_i, lp_i + nonlin_perts[i]), axis=0)
        ax.plot(*pert_plot.T, color=nonlin_col, alpha=lin_alpha)

    comb_pts_all = lin_pts_all + nonlin_perts
    ax.plot(*comb_pts_all[:2].T, color=comb1_col, label='right value')
    ax.plot(*comb_pts_all[2:].T, color=comb2_col, label='left value')

    ax.plot(*(comb_pts_all[2:] - comb_pts_all[3:4]
              + comb_pts_all[1:2]).T, linestyle='dashed', color=comb2_col)

    ax.plot(*comb_pts_all[:2].T, 'o', color=comb1_col, ms=ms)
    ax.plot(*comb_pts_all[2:].T, 'o', color=comb2_col, ms=ms)

    ax.legend(frameon=False)
    ax.set_xlabel('dim 1')
    ax.set_ylabel('dim 2')
    ax.set_zlabel('dim 3')
    ax.view_init(10, 190)
    return f, ax

def plot_all_gen(gen_dict, axs=None, fwid=4, upsampled=False, prediction=None):
    n_plots = len(gen_dict) 
    if axs is None:
        f, axs = plt.subplots(n_plots, 1, figsize=(fwid, fwid*n_plots),
                              sharex=True)
    for i, (key, out) in enumerate(gen_dict.items()):
        dec, xs = out[:2]
        if len(out) > 2:
            gen = out[-1]
        else:
            gen = dec
        try:
            split_k0 = key[0].split('_') 
            if len(split_k0) > 2:
                var = '_'.join(split_k0[:2])
                dec_cond = split_k0[2]
            else:
                var, dec_cond = key[0].split('_')
            gen_cond = key[1].split('_')[-1]
        except:
            print(key[0])
            var, dec_cond = key[0].split(' ', 1)
            var, gen_cond = key[1].split(' ', 1)
        if prediction is not None:
            pred = prediction.get(key)
            if pred is None:
                pred = prediction.get(key[::-1])
            if pred is not None:
                pred_ccgp = np.mean(pred['pred_ccgp'], axis=1)
            else:
                pred_ccgp = None
        else:
            pred_ccgp = None
        plot_decoding_gen(xs, dec, gen, var, dec_cond=dec_cond,
                          gen_cond=gen_cond, ax=axs[i], upsampled=upsampled,
                          pred_ccgp=pred_ccgp)
    return axs

@gpl.ax_adder
def plot_regression_performance(xs, regr, ax=None, filt_0=False):
    for i, neur in enumerate(regr):
        if filt_0:
            new_neur = np.zeros_like(neur)
            mask = np.mean(neur, axis=1) > 0
            new_neur[mask] = neur[mask]
            neur = new_neur
        gpl.plot_trace_werr(xs, neur.T, ax=ax)

def plot_loss_gen(xs, *loss_dicts, axs=None, fwid=3, t_ind=-1,
                  vert_scale=.2):
    n_pts = int(len(loss_dicts[0]))
    if axs is None:
        f, ax = plt.subplots(1, 1, figsize=(fwid, fwid*n_pts*vert_scale))
    yval_ind = 0
    yvals = np.linspace(0, 1, n_pts)
    yval_dict = {}
    prefix_list = []
    for loss_dict in loss_dicts:
        for key, ld_k in loss_dict.items():
            ks = key.split('_')
            if len(ks) == 1:
                ks = key.split(' ', maxsplit=1)
            prefix = ks[1]
            if prefix not in yval_dict.keys():
                prefix_list.append(prefix)
                yval_dict[prefix] = yvals[yval_ind]
                yval_ind = yval_ind + 1
            yval = yval_dict[prefix]
            ld_m = np.mean(ld_k, axis=1)
            gpl.plot_horiz_conf_interval(yval, ld_m[:, t_ind], ax=ax, conf95=True)
    gpl.add_vlines(0, ax=ax)
    ax.set_yticks(yvals)
    ax.set_yticklabels(prefix_list)
        
@gpl.ax_adder
def plot_decoding_gen(xs, dec, dec_gen, factor='', ax=None,
                      gen_template='generalization to {}',
                      dec_template='trained/tested on {}',
                      x_label='time from offer onset',
                      y_label='{} decoding performance',
                      pred_label='predicted',
                      dec_cond='',
                      gen_cond='',
                      pred_ccgp=None, pred_binding=None,
                      upsampled=False):
    dec_label = dec_template.format(dec_cond)
    gen_label = gen_template.format(gen_cond)
    if upsampled:
        plot_dec = dec
        plot_dec_gen = dec_gen
        error_func = gpl.std
    else:
        plot_dec = np.mean(dec, axis=1)
        plot_dec_gen = np.mean(dec_gen, axis=1)
        error_func = gpl.conf95_interval
    gpl.plot_trace_werr(xs, plot_dec, error_func=error_func, ax=ax,
                        label=dec_label)
    gpl.plot_trace_werr(xs, plot_dec_gen, error_func=error_func, ax=ax, 
                        label=gen_label)
    if pred_ccgp is not None:
        pred_ccgp = np.stack((pred_ccgp,)*len(xs), axis=1)
        gpl.plot_trace_werr(xs, pred_ccgp, conf95=True, ax=ax, 
                            label=pred_label)
    gpl.add_hlines(0.5, ax)
    gpl.add_hlines(0.7, ax)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label.format(factor))

def plot_code(code, plot_3d=True, ax=None, fsize=(3, 3), save_fig=None,
              ticks=None):
    if plot_3d:
        proj = '3d'
        n_comp = 3
    else:
        proj = None
        n_comp = 2
    if ax is None:
        f = plt.figure(figsize=fsize)
        ax = f.add_subplot(1, 1, 1, projection='3d')
    reps = code.get_all_representations()
    p = skd.PCA(n_comp)
    reps_trs = p.fit_transform(reps)
    reps_plot = list(reps_trs[:, i] for i in range(n_comp))
    ax.plot(*reps_plot, 'o')
    lim = np.max(np.abs(reps_trs))
    ax.set_xlim([-lim, lim])
    ax.set_ylim([-lim, lim])
    ax.set_zlim([-lim, lim])
    if ticks is None:
        ticks = np.linspace(-np.round(lim, 0), np.round(lim, 0), 3)
    else:
        ticks = np.linspace(-ticks, ticks, 3)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_zticks(ticks)
    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    ax.set_zlabel('PC 3')
    if save_fig is not None:
        f.savefig(save_fig, transparent=True, bbox_inches='tight')
    return f, ax

def plot_ccgp_error_tradeoff(pwrs, err_theory, err_emp, swap_theor,
                             gen_theory, gen_emp, ax=None, log_scale=False,
                             plot_min_pt=False, format_txt='SNR = {:.0f}'):
    if ax is None:
        f, ax = plt.subplots(1, 1)
    for i in range(err_theory.shape[1]):
        l = ax.plot(swap_theor[:, i], gen_theory[:, i],
                    label=format_txt.format(pwrs[i]))
        col = l[0].get_color()
        if plot_min_pt:
            err_prob = swap_theor[:, i] + gen_theory[:, i]
            ep_ind = np.argmin(err_prob)
            ax.plot(swap_theor[ep_ind, i], gen_theory[ep_ind, i], 'o',
                    color=col, markersize=5)
    ax.legend(frameon=False)
    ax.set_xlabel('misbinding error rate')
    ax.set_ylabel('generalization error rate')
    gpl.clean_plot(ax, 0)
    if log_scale:
        ax.set_xscale('log')
        ax.set_yscale('log')

def plot_ccgp_error_rates(pwrs, err_theory, err_emp, swap_theor, gen_theory,
                          gen_emp, axs=None):
    if axs is None:
        f, axs = plt.subplots(1, 2, sharey=True)

    l = gpl.plot_trace_werr(pwrs, 1 - err_emp.T, ax=axs[0], conf95=True,
                            label='all errors')
    col = l[0].get_color()
    axs[0].plot(pwrs, err_theory, color='k', linestyle='dashed')
    axs[0].plot(pwrs, err_theory, color=col, linestyle='dashed')
    axs[0].plot(pwrs, swap_theor, label='misbinding errors')

    l = gpl.plot_trace_werr(pwrs, 1 - gen_emp.T, ax=axs[1], conf95=True)
    col = l[0].get_color()
    axs[1].plot(pwrs, gen_theory, label='gen', linestyle='dashed',
                color='k')
    axs[1].plot(pwrs, gen_theory, label='gen', linestyle='dashed',
                color=col)
    axs[0].set_yscale('log')
    axs[1].set_yscale('log')
    axs[0].legend(frameon=False)
    axs[0].set_xlabel('SNR')
    axs[1].set_xlabel('SNR')
    axs[0].set_ylabel('error rate')
    axs[1].set_ylabel('generalization error rate')
    return axs
