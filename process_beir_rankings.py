import os
import argparse
from tqdm import tqdm

import torch
import pandas as pd
import ipdb

def setup_argparse():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=str, default="beir/results/contriever/test/test_results.pt")
    p.add_argument('--dataset', type=str, default="scifact")
    return p.parse_args()


if __name__ == "__main__":
    args = setup_argparse()
    
    # load in the results save file, and the qrels file
    # format [query_id]["corpus_id"]["score"] for top 1k by default. Len will be # queries
    print(f"Loading in {args.input} and qrels for {args.dataset}")
    results = torch.load(args.input)
    qrels = pd.read_csv(f"beir/{args.dataset}/qrels/test.tsv", sep='\t', dtype={"query-id": str, "corpus-id": str})
    # for each query, grab the rank of the correct file, output: query-id, corpus-id, rank. 
    # for perturbations tbd also include perturbation_type (to_female, etc)
    rank_data = []
    for qid, res in tqdm(results.items()):
        #ipdb.set_trace()
        gold_corpus_id = qrels[ qrels["query-id"] == qid ]["corpus-id"].values[0]
        rank = {key: rank for rank, key in enumerate(sorted(res, key=res.get, reverse=True), 1)}.get(gold_corpus_id, -1)
        # TODO what do I do if not in list?
        
        rank_data.append({
            "query-id": qid, 
            "corpus-id": gold_corpus_id, 
            "rank": rank
        })
        # TODO add support for perturbations
    rank_df = pd.DataFrame(data=rank_data, columns=["query-id","corpus-id","rank"])
    print("Sample processed results")
    print(rank_df.sample(10))
    
    output = os.path.join(os.path.split(args.input)[0], f"{args.dataset}_rank.pkl")
    print(f"Saving to {output}")
    rank_df.to_pickle(output)
