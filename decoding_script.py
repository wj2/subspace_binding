
import argparse
import pickle
from datetime import datetime
import os

import multiple_representations.figures as mrf
import multiple_representations.analysis as mra
import multiple_representations.auxiliary as mraux

all_regions = ("OFC", "PCC", "pgACC", "vmPFC", "VS", "all")


def create_parser():
    parser = argparse.ArgumentParser(
        description='perform decoding analyses on Fine data'
    )
    out_template = "decoding_m-{monkey}_r-{regions}_{jobid}.pkl"
    parser.add_argument('-o', '--output_template', default=out_template, type=str,
                        help='file to save the output in')
    parser.add_argument(
        '--output_folder',
        default="../results/subspace_binding/decoding/",
        type=str,
        help='file to save the output in'
    )
    parser.add_argument(
        "--data_folder",
        default="../data/subspace_binding"
    )
    parser.add_argument("--subsample_neurons", default=None, type=int)
    parser.add_argument("--pca_pre", default=.99, type=float)
    parser.add_argument("--resamples", default=200, type=int)
    parser.add_argument("--exclude_middle_percentiles", default=15, type=float)
    parser.add_argument("--min_trials", default=160, type=int)
    parser.add_argument("--dec_less", default=True, type=bool)
    parser.add_argument("--tbeg", default=100, type=float)
    parser.add_argument("--tend", default=1000, type=float)
    parser.add_argument("--winsize", default=300, type=float)
    parser.add_argument("--winstep", default=300, type=float)
    parser.add_argument("--include_safe", action="store_true", default=False)
    parser.add_argument("--correct_only", action="store_true", default=False)
    parser.add_argument("--regions", default=all_regions, nargs="+", type=str)
    parser.add_argument("--data_field", default="subj_ev", type=str)
    parser.add_argument("--use_split_dec", default=None)
    parser.add_argument("--use_time", action="store_true", default=False)
    parser.add_argument("--time_accumulate", action="store_true", default=False)
    parser.add_argument("--jobid", default="0000", type=str)
    parser.add_argument("--region_ind", default=None, type=int)
    parser.add_argument("--monkey_ind", default=None, type=int)
    return parser

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()

    args.date = datetime.now()
    dec_fig = mrf.DecodingFigure()
    exper_data = dec_fig.get_experimental_data(data_folder=args.data_folder)
    if args.region_ind is not None:
        regions = (all_regions[args.region_ind],)
    else:
        regions = args.regions
    if args.monkey_ind is not None:
        targ_m = mraux.monkey_list[args.monkey_ind]
        s_mask = exper_data["animal"] == targ_m
        exper_data = exper_data.session_mask(s_mask)
    else:
        targ_m = "all"
    
    data_field = args.data_field
    dead_perc = args.exclude_middle_percentiles
    min_trials = args.min_trials
    dec_less = args.dec_less
    def mask_func(x): return x < 1
    mask_var = "prob"
    if args.include_safe:
        def mask_func(x): return x <= 1

    decoding_results = {}
    timing_results = {}
    timing_gen_results = {}
    for region in regions:        
        if region == "all":
            use_regions = None
        else:
            use_regions = (region,)
        func_args = (
            exper_data,
            args.tbeg,
            args.tend,
            data_field,
        )
        func_kwargs = dict(
            winsize=args.winsize,
            pre_pca=args.pca_pre,
            pop_resamples=args.resamples,
            tstep=args.winstep,
            time_accumulate=args.time_accumulate,
            regions=use_regions,
            correct_only=args.correct_only,
            subsample_neurons=args.subsample_neurons,
            dead_perc=dead_perc,
            mask_func=mask_func,
            mask_var=mask_var,
            min_trials=min_trials,
            dec_less=dec_less,
            use_split_dec=args.use_split_dec,
            use_time=args.use_time,            
        )
        out = mra.compute_timewindow_dec(*func_args, **func_kwargs)
        timing_gen_results[region] = out

        out = mra.compute_all_generalizations(
            *func_args, **func_kwargs,
        )
        decoding_results[region] = out
        
        out = mra.compute_time_dec(*func_args, **func_kwargs)
        timing_results[region] = out


    pred_results = dec_fig._direct_predictions(
        "dec",
        decs=decoding_results,
        use_regions=args.regions,
        time_acc=args.time_accumulate,
    )

    pred_time_results = dec_fig._direct_predictions(
        "dec",
        decs=timing_results,
        use_regions=args.regions,
        time_acc=args.time_accumulate,        
    )

    decoding_results = mraux.remove_pops(decoding_results)
    timing_results = mraux.remove_pops(timing_results)
    timing_gen_results = mraux.remove_pops(timing_gen_results)
    
    save_dict = {
        "args": vars(args),
        "decoding": decoding_results,
        "predictions": pred_results,
        "timing": timing_results,
        "predictions_timing": pred_time_results,
        "timing_gen": timing_gen_results,
    }
    r_str = "-".join(regions)
    file = args.output_template.format(
        regions=r_str,
        monkey=targ_m,
        jobid=args.jobid,
    )
    path = os.path.join(args.output_folder, file)
    pickle.dump(save_dict, open(path, "wb"))
    
