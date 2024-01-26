
import argparse
import pickle
from datetime import datetime
import os

import multiple_representations.figures as mrf


def create_parser():
    parser = argparse.ArgumentParser(
        description='perform correlation analyses on Fine data'
    )
    out_template = "correlation_f-{fig_str}_{jobid}.pkl"
    parser.add_argument('-o', '--output_template', default=out_template, type=str,
                        help='file to save the output in')
    parser.add_argument(
        '--output_folder',
        default="../results/subspace_binding/correlations/",
        type=str,
        help='file to save the output in'
    )
    parser.add_argument(
        "--data_folder",
        default="../data/subspace_binding"
    )
    parser.add_argument("--include_safe", action="store_true", default=False)
    parser.add_argument("--correct_only", action="store_true", default=False)
    parser.add_argument("--data_field", default="subj_ev", type=str)
    parser.add_argument("--jobid", default="0000", type=str)
    parser.add_argument("--fig_ind", default=None, type=int)
    default_fit_folder = "../results/subspace_binding/lm_fits/"
    parser.add_argument("--lm_fits", default=default_fit_folder)
    return parser

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()

    args.date = datetime.now()
    
    fig_data = {}
    figs = (
        "nominal", "timecourse", "time", "monkey", "model_mix", "subset", "nonlinear"
    )
    if args.fig_ind is not None:
        plot_figs = (figs[args.fig_ind],)
    else:
        plot_figs = figs

    if "nominal" in plot_figs:
        f = mrf.SelectivityFigure(data=fig_data.get('selectivity-nom'))
        f.panel_eg_neurons()
        f.panel_model_comp(use_folder=args.lm_fits)
        f.panel_subspace_corr(recompute=False)
        fig_data['selectivity-nom'] = f.get_data()
        f.save(
            'selectivity-nom_fig_{jobid}.svg'.format(jobid=args.jobid),
            use_bf=args.output_folder,
        )
    
    if "timecourse" in plot_figs:
        f = mrf.SelectivityFigure(data=fig_data.get('selectivity-tc'))
        new_f = f.panel_subspace_corr_tc(recompute=False)
        fig_data['selectivity-tc'] = f.get_data()
        fn = 'selectivity-tc_fig_{jobid}.svg'.format(jobid=args.jobid)
        path = os.path.join(args.output_folder, fn)
        new_f.savefig(path, bbox_inches="tight", transparent=True)
    
    if "time" in plot_figs:
        f = mrf.SelectivityFigure(data=fig_data.get('selectivity-time'))
        f.panel_subspace_corr_time(recompute=True)
        fig_data['selectivity-time'] = f.get_data()
        f.save(
            'selectivity-time_fig_{jobid}.svg'.format(jobid=args.jobid),
            use_bf=args.output_folder,
        )

    if "monkey" in plot_figs:
        f = mrf.SelectivityFigure(data=fig_data.get('selectivity-monkey'))
        f.panel_subspace_corr_monkey(recompute=False)
        fig_data['selectivity-monkey'] = f.get_data()
        f.save(
            'selectivity-monkey_fig_{jobid}.svg'.format(jobid=args.jobid),
            use_bf=args.output_folder,
        )
    
    if "model_mix" in plot_figs:
        f = mrf.SelectivityFigure(data=fig_data.get('selectivity-mix'))
        f.panel_subspace_corr_model_mix(recompute=False, use_folder=args.lm_fits)
        fig_data['selectivity-mix'] = f.get_data()
        f.save(
            'selectivity-mix_fig_{jobid}.svg'.format(jobid=args.jobid),
            use_bf=args.output_folder,
        )
        
    if "subset" in plot_figs:
        f = mrf.SelectivityFigure(data=fig_data.get('selectivity-subset'))
        f.panel_subspace_corr_subset(recompute=False)
        fig_data['selectivity-subset'] = f.get_data()
        f.save(
            'selectivity-subset_fig_{jobid}.svg'.format(jobid=args.jobid),
            use_bf=args.output_folder,
        )
        
    if "nonlinear" in plot_figs:
        f = mrf.SelectivityFigure(fig_key='selectivity_figure_nonlin', 
                                  data=fig_data.get('selectivity_nl'))
        f.panel_eg_neurons()
        f.panel_subspace_corr()
        fig_data['selectivity_nl'] = f.get_data()
        f.save(
            'selectivity-nl_fig_{jobid}.svg'.format(jobid=args.jobid),
            use_bf=args.output_folder,
        )

    save_dict = {
        "args": vars(args),
        "fig_data": fig_data
    }

    if args.fig_ind is None:
        fig_str = "all"
    else:
        fig_str = "-".join(plot_figs)
    file = args.output_template.format(
        fig_str=fig_str,
        jobid=args.jobid,
    )
    path = os.path.join(args.output_folder, file)
    pickle.dump(save_dict, open(path, "wb"))
    
