
import argparse
import pickle
from datetime import datetime
import os

import multiple_representations.figures as mrf


def create_parser():
    parser = argparse.ArgumentParser(
        description='perform correlation analyses on Fine data'
    )
    out_template = "comb-rep_p-{panel_str}_{jobid}.pkl"
    parser.add_argument('-o', '--output_template', default=out_template, type=str,
                        help='file to save the output in')
    parser.add_argument(
        '--output_folder',
        default="../results/subspace_binding/combined_rep/",
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
    parser.add_argument("--panel_ind", default=None, type=int)
    return parser

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()

    args.date = datetime.now()
    
    fig_data = {}
    panels = (
        "vis", "dists", "subspace_corr", "choice_bhv", "dec_bhv",
    )
    if args.panel_ind is not None:
        plot_panels = (panels[args.panel_ind],)
        panel_str = "-".join(plot_panels)
    else:
        panel_str = "all"
        plot_panels = panels

    fig_key = "combined_rep"
    cr_fig = mrf.CombinedRepFigure(data=fig_data.get(fig_key))
    if "vis" in plot_panels:
        cr_fig.panel_vis()
    
    if "dists" in plot_panels:
        cr_fig.panel_dists(recompute=False)
    
    if "subspace_corr" in plot_panels:
        cr_fig.panel_subspace_corr(recompute=False)
        
    if "choice_bhv" in plot_panels:
        cr_fig.panel_choice_bhv()
    
    if "dec_bhv" in plot_panels:
        cr_fig.panel_dec_bhv(recompute=False)
        f_additional = cr_fig.additional_dec_bhv_sweep(recompute=False)
        fn = "cr_add_fig_{panel_str}_{jobid}.svg".format(
            panel_str=panel_str, jobid=args.jobid
        )
        path = os.path.join(args.output_folder, fn)
        f_additional.savefig(path, bbox_inches="tight", transparent=True)

    fig_data[fig_key] = cr_fig.get_data()        

    save_dict = {
        "args": vars(args),
        "fig_data": fig_data
    }

    cr_fig.save(
        "cr_fig_{panel_str}_{jobid}.svg".format(panel_str=panel_str, jobid=args.jobid),
        use_bf=args.output_folder,
    )
    file = args.output_template.format(
        panel_str=panel_str,
        jobid=args.jobid,
    )
    path = os.path.join(args.output_folder, file)
    pickle.dump(save_dict, open(path, "wb"))
    
