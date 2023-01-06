import re
import sys
import os
import argparse
import pickle
from collections import defaultdict
from tqdm import tqdm

import pandas as pd
import seaborn as sns
import numpy as np
import ipdb

from utils.wandb_utils import get_wandb_summary, get_model_compression


COLUMNS = ["model", "seed", "dataset", "metric", "value", "compression", "data_type", "compression_dataset"] # for pandas dataframe
ndcg_pattern = re.compile("(?<=eval_beir\.py).*INFO.*NDCG@10:.*") # for parsing logs for ndcg@10
metric_pattern = re.compile("(?<=eval_beir\.py).*INFO.*:.*:.*") # for parsing logs for any metric
seed_pattern = re.compile("seed_[0-9]+")
model_pattern = re.compile("[A-z]*(?=/seed_)")

gender_datasets = ["nq-train-new-complement", "nq-train-new-gender", "nq-train-new-male", "nq-train-new-female",
                     "msmarco-female", "msmarco-male", "msmarco-neutral", "msmarco-gender"]
"""
Pattern
[10/11/2022 20:31:08] {eval_beir.py:61} INFO - dbpedia-entity : NDCG@10: 21.3
"""

def get_model_and_seed(text):
    # pattern will always be prefix/MODEL/SEED_N/run.log for normal datasets, but there might be more things between seed and run.log
    seed = seed_pattern.search(text).group(0)
    model = model_pattern.search(text).group(0)
    #prefix, model, seed, _ = text.rsplit("/", 3)
    return model, seed

def add_average_to_dataset(df, metric="NDCG@10", model="contriever", compression_dataset="biasinbios"):
    # TODO maybe filter on datasets that are appropriate for the model? -msmarco for things fine tuned on ms marco + make sure valid for language
    # filtering dataset
    exclude = ['trec-covid', 'trec-covid-v2']
    new_rows = []
    mod_df = df[ ~df["dataset"].isin(exclude) ]
    mod_df = mod_df [ mod_df["model"] == model ]
    mod_df = mod_df [ mod_df["compression_dataset"] == compression_dataset ]    
    
    all_seeds = mod_df["seed"].unique()
    for seed in all_seeds:
        seed_slice = mod_df[ mod_df ["seed"] == seed ]
        # iterate across data_types
        data_types = seed_slice["data_type"].unique()
        for dt in data_types:
            data_slice = seed_slice[ seed_slice["data_type"] == dt ]
            # get compression
            compression = data_slice["compression"].unique()
            if len(compression) > 1:
                sys.exit(f"data error, should not be more than one compression amount per slice and got {compression}")
            if metric == "all":
                all_metrics = data_slice["metric"].unique()
            else:
                all_metrics = [metric]
            for m in all_metrics:
                metric_slice = data_slice[ data_slice["metric"] == m ]
                avg = metric_slice["value"].mean()
                new_row = [model, seed, "all", m, avg, compression[0], dt, compression_dataset]
                new_rows.append(new_row)
    df2 = pd.DataFrame(data=new_rows, columns=df.columns)

    return pd.concat([df, df2], ignore_index=True, axis=0)

def generate_all_seed_logs(log_pattern, desired_seeds):
    log_paths = []
    seed_list = desired_seeds if desired_seeds else range(25)
    for i in seed_list:
        if i==13:
            continue
        log_paths.append(log_pattern.format(i))
    return log_paths

def setup_argparse():
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', nargs="+", help='limit to just these seeds')
    p.add_argument('--output', default='beir/results/contriever/logs_df_{}.pkl', help="place to save dataframe")
    p.add_argument('--metric', default='NDCG@10', choices=['NDCG@10', 'all'])
    p.add_argument('--gender', action='store_true', help="process gender logs instead of standard beir")
    p.add_argument('--inlp', action='store_true', help="process inlp logs instead of standard beir")
    p.add_argument('--compression_dataset', default='biasinbios', 
                  choices=['biasinbios', 'biasinbios_profession','wizard', 'wizard_binary', 'wikipedia', 'wikipedia_binary'])
    p.add_argument('--save_compression', action='store_true')
    return p.parse_args()


if __name__ == "__main__":
    args = setup_argparse()
    
    regex_pattern = ndcg_pattern if args.metric == "NDCG@10" else metric_pattern
    compression_dataset = args.compression_dataset
    output = args.output.format(compression_dataset)
    desired_seeds = set(map(int, args.seeds)) if bool(args.seeds) else False

    base_pattern = "beir/results/contriever/seed_{}/"  # I changed the base patterns just to make parsing results files easier for special experiments
    if args.inlp:
        base_pattern = os.path.join(base_pattern, "inlp")
    if not args.gender:
        log_pattern = os.path.join(base_pattern, "run.log")
        log_files = generate_all_seed_logs(log_pattern, desired_seeds)
    else:
        log_files = []
        for ds in gender_datasets:
            log_pattern = os.path.join(base_pattern, f"{ds}/run.log")
            log_files.extend(generate_all_seed_logs(log_pattern, desired_seeds))
    
    project_name = f"seraphinatarrant/{compression_dataset} MDL probing"
    runs_df = get_wandb_summary(project_name=project_name)
    t2m2s2c = get_model_compression(runs_df)
    if args.save_compression:
        with open(f"compression_logs/{compression_dataset}_type2model2seed2compression.pkl", "wb") as fout: # save for later since it's easier to have this mapping
            pickle.dump(t2m2s2c, fout)

    all_results = []
    for log_file in tqdm(log_files):
        with open(log_file, "r") as fin:
            log = fin.readlines()

        model, seed = get_model_and_seed(log_file)
        for data_type in t2m2s2c.keys():
            m2s2c = t2m2s2c[data_type]
            compression = m2s2c[model].get(seed)
            for line in log:
                result = regex_pattern.search(line)
                if result:
                    _, result = result.group(0).split('-', 1)
                    ds, metric, val = map(str.strip, result.split(':'))
                    #print(ds, metric, val)
                    all_results.append([model, seed, ds, metric, float(val), compression, data_type, compression_dataset])

    df = pd.DataFrame(data=all_results,columns=COLUMNS)
    new_df = add_average_to_dataset(df, args.metric, compression_dataset=compression_dataset)
    new_df.to_pickle(output)