# coding: utf-8

from __future__ import annotations

import os
import tempfile
import shutil
from fnmatch import fnmatch
from multiprocessing import Pool as ProcessPool
from typing import Any
from argparse import ArgumentParser
from tqdm import tqdm

import awkward as ak
import numpy as np
import uproot

# load the nn evaluation
from branchnames import nn_columns
from first_nn import evaluate_events, evaluate_events_param
from feature_calc import spin2id, load_input_pipe
from features import feats

from lumin.nn.ensemble.ensemble import Ensemble
from lumin.nn.callbacks.data_callbacks import ParametrisedPrediction

#
# configurations
#

masses = [
    250, 260, 270, 280, 300, 320, 350, 400, 450, 500, 550, 600, 650,
    700, 750, 800, 850, 900, 1000, 1250, 1500, 1750, 2000, 2500, 3000,
]
spins = [0, 1]

klub_index_cols = ["EventNumber", "RunNumber", "lumi"]
klub_cut_cols = ['isLeptrigger', "pairType", 'nleps', 'nbjetscand'] 


def struct_to_float_array(arr):
    return arr.astype([(name, np.float32) for name in arr.dtype.names], copy=False).view(np.float32).reshape((-1, len(arr.dtype)))


def make_parser():
    parser = ArgumentParser(description=("Run DNN evaluation with a given model"
                                         "on a provided list of samples"))
    parser.add_argument("-f", "--file_names", type=str, nargs="+", required=False,
                        help="file name(s). Use this argument if running with Condor.")
    parser.add_argument("-m", "--model_name", type=str, required=True, default='nonparam_baseline',
                        help="Model name as provided in Readme (TODO)")
    parser.add_argument("-p", "--parametrised", action="store_true",
                        help="If set, evaluate parametrised model.")
    return parser


def evaluate_file_condor(input_file_path: str, model_name: str, parametrised: bool):
    ensembles = [('/eos/user/j/jowulff/res_HH'
                 f'/cms_runII_dnn_resonant/{model_name}'
                 f'/weights/selected_set_0_{model_name}'),
                 ('/eos/user/j/jowulff/res_HH'
                 f'/cms_runII_dnn_resonant/{model_name}'
                 f'/weights/selected_set_1_{model_name}')]
    input_pipes = [('/eos/user/j/jowulff/res_HH'
                   f'/cms_runII_dnn_resonant/{model_name}'
                   f'/weights/selected_set_0_{model_name}_input_pipe.pkl'),
                    ('/eos/user/j/jowulff/res_HH'
                   f'/cms_runII_dnn_resonant/{model_name}'
                   f'/weights/selected_set_1_{model_name}_input_pipe.pkl')]
    # prepare expressions
    input_pipe_0 = load_input_pipe(input_pipes[0])
    input_pipe_1 = load_input_pipe(input_pipes[1])
    cont_feats, cat_feats = feats[model_name][0], feats[model_name][1]
    expressions = list(set(klub_index_cols) | set(cont_feats+cat_feats))
    # load the klub array
    with uproot.open(input_file_path) as f:
        even_data = f["data_0"].arrays(expressions=expressions)
        odd_data = f["data_1"].arrays(expressions=expressions)
    
    all_feats = cont_feats+cat_feats
    cont_feats_0 = input_pipe_0.transform(struct_to_float_array(even_data[cont_feats].to_numpy()))
    cont_feats_1 = input_pipe_1.transform(struct_to_float_array(odd_data[cont_feats].to_numpy()))
    cat_feats_0 = struct_to_float_array(even_data[cat_feats].to_numpy())
    cat_feats_1 = struct_to_float_array(odd_data[cat_feats].to_numpy())

    ensemble_0 = Ensemble.from_save(ensembles[0])
    ensemble_1 = Ensemble.from_save(ensembles[1])
    # run the evaluation
    if parametrised == True:
        raise NotImplementedError("param prediction not supported yet.")
    else:
        preds_0 = ak.flatten(ensemble_1.predict(np.hstack([cont_feats_0, cat_feats_0])))
        preds_1 = ak.flatten(ensemble_0.predict(np.hstack([cont_feats_1, cat_feats_1])))
        preds = ak.concatenate((preds_0, preds_1))
        index_arr_0 = even_data[klub_index_cols] 
        index_arr_1 = odd_data[klub_index_cols] 
    index_arr = ak.concatenate((index_arr_0, index_arr_1))
    out_dict = {f'dnn_{model_name}': preds}
    for field in index_arr.fields:
        out_dict[field] = index_arr[field]
    with uproot.update(input_file_path) as f:
        f["evaluation"] = out_dict 


def evaluate_files(input_files: list, model_name:str, parametrised: bool):
    for file in input_files:
        evaluate_file_condor(input_file_path=file,
                             model_name=model_name,
                             parametrised=parametrised)
    print("done")


# entry hook
if __name__ == "__main__":

    parser = make_parser()
    args = parser.parse_args()
    #if args.use_gpu:
        #import torch
        #try:
            #torch.cuda.set_device(0)
            #print("Using GPU:")
            #print(torch.cuda.get_device_name())
        #except:
            #print("couldn't access GPU")
    evaluate_files(input_files=args.file_names,
                            model_name=args.model_name,
                            parametrised=args.parametrised)