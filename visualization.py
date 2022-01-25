
import numpy as np
import sklearn.decomposition as skd
import matplotlib.pyplot as plt
import functools as ft

import general.plotting as gpl
import composite_tangling.code_analysis as ca

def plot_all_gen(gen_dict, axs=None, fwid=4, upsampled=False):
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
            var, dec_cond = key[0].split('_')
            var, gen_cond = key[1].split('_')
        except:
            var, dec_cond = key[0].split(' ', 1)
            var, gen_cond = key[1].split(' ', 1)
        print(xs.shape, dec.shape, gen.shape)
        plot_decoding_gen(xs, dec, gen, var, dec_cond=dec_cond,
                          gen_cond=gen_cond, ax=axs[i], upsampled=upsampled)
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
