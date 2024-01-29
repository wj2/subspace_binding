
import argparse
from datetime import datetime
import pickle
import os

import multiple_representations.figures as mrf


def create_parser():
    parser = argparse.ArgumentParser(
        description='perform correlation analyses on Fine data'
    )
    out_template = "tempchange_{jobid}.pkl"
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
    parser.add_argument(
        "--percentile", default=80, type=float
    )
    parser.add_argument("--jobid", default="0000", type=str)
    
    return parser

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()

    args.date = datetime.now()

    fig_data = {}
    fig_key = "temp_change"
    tcd = mrf.TemporalChangeDecoding(data=fig_data.get(fig_key))
    tcd.panel_dec_tc(train_trl_perc=args.percentile)
    tcd.panel_dec_region(train_trl_perc=args.percentile)

    fig_data[fig_key] = tcd.get_data()
    ks = ("panel_dec_tc", "panel_dec_region")
    save_dict = {k: fig_data[fig_key][k] for k in ks}

    fn = args.output_template.format(jobid=args.jobid)
    fp = os.path.join(args.output_folder, fn)
    pickle.dump(save_dict, open(fp, "wb"))
    tcd.save(
        'temp_change_{jobid}_fig.svg'.format(jobid=args.jobid),
        use_bf=args.output_folder,
    )
    

