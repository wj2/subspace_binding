
import argparse
from datetime import datetime

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
    parser.add_argument("--data_field", default="subj_ev", type=str)
    parser.add_argument("--jobid", default="0000", type=str)
    
    return parser

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()

    args.date = datetime.now()

    if args.include_safe:
        param_key = "decoding_safe_figure"
    else:
        param_key = "decoding_figure"

    fig_data = {}
    fig_key = 'decoding'
    dec_fig = mrf.DecodingFigure(data=fig_data.get(fig_key), fig_key=param_key)
    dec_fig.panel_ev_generalization(force_reload=True, force_recompute=True)
    dec_fig.panel_ev_direct_predictions(force_refit=True)

    dec_fig.save_pickles(
        args.output_folder, suff="_{jobid}".format(jobid=args.jobid),
    )
    dec_name = "ev_generalization_{jobid}.mat".format(jobid=args.jobid)
    pred_name = "ev_direct_predictions_{jobid}.mat".format(jobid=args.jobid)

    fig_key = 'theory'
    th_fig = mrf.TheoryFigure(data=fig_data.get(fig_key))
    ps = th_fig.panel_theory_dec_plot(
        dec_folder=args.output_folder, dec_file=dec_name, pred_file=pred_name,
    )
    ps = th_fig.panel_theory_comparison(
        dec_folder=args.output_folder, dec_file=dec_name, pred_file=pred_name,
    )

    fig_data['theory'] = th_fig.get_data()

    fn = "theory_fig_{jobid}.svg".format(jobid=args.jobid)
    th_fig.save(fn, use_bf=args.output_folder)

