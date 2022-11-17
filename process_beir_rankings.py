import os
import random
import argparse
from tqdm import tqdm

import torch
import pandas as pd
import ipdb

def setup_argparse():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=str, default="beir/results/contriever/seed_17/nq-train-new-gender/results.pt")
    p.add_argument('--dataset', type=str, default="nq-train-new-gender")
    p.add_argument('--ranking', action='store_true', help="calc and save ranks of gold docs")
    p.add_argument('--samples', action='store_true', help="print random samples of retrieved docs")
    return p.parse_args()


if __name__ == "__main__":
    args = setup_argparse()
    
    # load in the results save file, and the qrels file
    # format [query_id]["corpus_id"]["score"] for top 1k by default. Len will be # queries
    print(f"Loading in {args.input} and qrels for {args.dataset}")
    results = torch.load(args.input)
    qrels = pd.read_csv(f"beir/{args.dataset}/qrels/test.tsv", sep='\t', dtype={"query-id": str, "corpus-id": str})
    
    if args.ranking:
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
    
    if args.samples:
        n_samples = 10
        top_k = 10
        print("reading queries")
        queries = pd.read_json(f"beir/{args.dataset}/queries.jsonl", lines=True)
        # randomly sample
        qids = random.sample(list(results.keys()), n_samples)
        corpus = pd.read_json(f"beir/{args.dataset}/corpus.jsonl", lines=True)
        for qid in qids:
            query_text = queries[ queries["_id"] == qid ]["text"].values[0]
             
            answer_texts = []
            res = results[qid]
            results_docs = sorted(res, key=res.get, reverse=True)[:top_k]
            
            gold_corpus_id = qrels[ qrels["query-id"] == qid ]["corpus-id"].values[0]
            rank = {key: rank for rank, key in enumerate(sorted(res, key=res.get, reverse=True), 1)}.get(gold_corpus_id, -1)
            
            # sub_corpus = []
            # print("reading corpus")
            # corpus_reader = pd.read_json(f"beir/{args.dataset}/corpus.jsonl", lines=True, chunksize=500000)
            # for chunk in tqdm(corpus_reader):
            #    keep = chunk[chunk["_id"].isin(results_docs)]
            #    sub_corpus.append(keep)
            # corpus = pd.concat(sub_corpus, ignore_index=True)
            
            gold_doc = corpus[ corpus["_id"] == gold_corpus_id ]["text"].values[0]        
            print("-"*89)
            print(f"Query: {query_text}")
            print(f"gold_rank: {rank}")
            print(f"gold_doc:\n{gold_corpus_id}\n{gold_doc}")
            print("*"*3)
            print("Answers")
            for doc_id in results_docs:
                doc, score = corpus[ corpus["_id"] == doc_id ]["text"].values[0], res[doc_id]
                answer_texts.append((doc, score))
                print(doc_id)
                print(f"{score} {doc}")

