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
from feature_calc import calc_feats, spin2id
from branchnames import nn_columns
from features import feats 

#
# configurations
#

masses = [
    250, 260, 270, 280, 300, 320, 350, 400, 450, 500, 550, 600, 650,
    700, 750, 800, 850, 900, 1000, 1250, 1500, 1750, 2000, 2500, 3000,
]

id2spin = {0:'radion', 2:'graviton'}
spin2id = {id2spin[i]:i for i in id2spin}
klub_index_cols = ["EventNumber", "RunNumber", "lumi"]
klub_cut_cols = ['isLeptrigger', "pairType", 'nleps', 'nbjetscand'] 

baseline_selection = (
    "isLeptrigger & "
    "((pairType == 0) | (pairType == 1) | (pairType == 2)) & "
    "(nleps == 0) & "
    "(nbjetscand > 1)"
)


def make_parser():
    parser = ArgumentParser(description=("Run DNN evaluation with a given model"
                                         "on a provided list of samples"))
    parser.add_argument("-f", "--file", type=str, nargs="+", required=True,
                        help="file name. Use this argument if running with Condor.")
    parser.add_argument("-i", "--sample_id", type=int, required=True,
                        help="sample_id (part of json)")
    parser.add_argument("-s", "--sum_w", type=float, required=True,
                        help="sum_w of the sample")
    parser.add_argument("-y", "--year", type=str, required=True,
                        help="2016, 2016APV, 2017 or 2018")
    parser.add_argument("--add_htautau", action="store_true",
                        help="If set, evaluate htautau model.")
    return parser
    

def evaluate_file_condor(input_file_path: str, sample_id: int, sum_w: float, year: str, add_htautau: bool) -> None:
    """
    Output branches will match the length of the KLUB branches. Arrays are padded with -1 where 
    events don't pass the baseline selection
    """
    # prepare expressions
    expressions = list(set(klub_index_cols) | set(nn_columns))
    # get the bool mask
    with uproot.open(input_file_path) as f:
        klub_index = f["HTauTauTree"].arrays(expressions=list(set(klub_cut_cols) | set(klub_index_cols)))

    with uproot.open(input_file_path) as f:
        input_array = f["HTauTauTree"].arrays(expressions=expressions, cut=baseline_selection)
        if ("Bulk" in input_file_path) and ("_VBF_" in input_file_path):
            raise ValueError("Not using VBF signal skims.")
        elif "_ggF_" in input_file_path:
            sample_name = input_file_path.split('/')[-2]
            # sample_name will be something like SKIM_ggF_BulkGraviton_m2000
            # read mass and spin
            mass = int(sample_name.split("_")[-1][1:])
            spin = (sample_name.split("_")[-2].replace("Bulk", "")).lower()
            output_array = calc_feats(input_array, sample_id=sample_id, sum_w=sum_w, mass=mass, spin=spin2id[spin], year=year, add_htautau=add_htautau)
            # output_array = evaluate_events_mass_param(input_array, [mass], model_name=model_name)
        else:
            output_array = calc_feats(input_array, sample_id=sample_id, sum_w=sum_w, mass=125., spin=-1, year=year, add_htautau=add_htautau)

    # store the output as root
    # since we're using condor and the xrootd file transfer, 
    # we only need to write to the cwd
    # and then condor transfers to eos for us
    file_name = f"./{input_file_path.split('/')[-1]}"
    with uproot.recreate(file_name) as output_file:
        output_file["features"] = dict(zip(output_array.fields, ak.unzip(output_array)))
        output_file["HTauTauTree"] = dict(zip(klub_index.fields, ak.unzip(klub_index)))


def evaluate_files(input_files: list, sample_id: int, sum_w:float, year: str, add_htautau: bool) -> None:
    for file in input_files:
        evaluate_file_condor(input_file_path=file,
                            sample_id=sample_id,
                            sum_w=sum_w,
                            year=year,
                            add_htautau=add_htautau)


# entry hook
if __name__ == "__main__":

    parser = make_parser()
    args = parser.parse_args()
    evaluate_files(input_files=args.file,
                    sample_id=args.sample_id,
                    sum_w=args.sum_w,
                    year=args.year,
                    add_htautau=args.add_htautau)
    