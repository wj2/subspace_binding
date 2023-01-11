
import argparse
import scipy.stats as sts
import numpy as np
import pickle
import functools as ft
from datetime import datetime

import general.utility as u
import multiple_representations.analysis as mra
import multiple_representations.auxiliary as mraux

def create_parser():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('input_file', type=str)
    parser.add_argument('-o', '--output_folder',
                        default='../results/subspace_fits/',
                        type=str,
                        help='folder to save the output in')
    parser.add_argument('--stan_iter', default=500, type=int)
    return parser

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    
    input_dict = pickle.load(open(args.input_file, 'rb'))

    out = mra.fit_stan_models(
        input_dict,
        iter=args.stan_iter,
    )
    mraux.save_model_fits(out, args.output_folder)
    
    
