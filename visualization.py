
import numpy as np
import sklearn.decomposition as skd
import matplotlib.pyplot as plt
import functools as ft

import general.utility as u
import general.plotting as gpl
import composite_tangling.code_analysis as ca
import multiple_representations.analysis as mra

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
    # out = mra.compute_within_across_corr(mus, masks)
    out = None
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
    axs[0].legend(frameon=False)
    axs[0].set_xlabel('SNR')
    axs[1].set_xlabel('SNR')
    axs[0].set_ylabel('error rate')
    axs[1].set_ylabel('generalization error rate')
    return axs
