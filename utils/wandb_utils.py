import wandb
import pandas as pd

from collections import defaultdict


def get_wandb_summary(project_name="seraphinatarrant/biasinbios MDL probing"):
    # Get compression from wandb api
    api = wandb.Api()

    # Project is specified by <entity/project-name>
    runs = api.runs(project_name)

    summary_list, config_list, name_list = [], [], []
    for run in runs: 
        # .summary contains the output keys/values for metrics like accuracy.
        #  We call ._json_dict to omit large files 
        summary_list.append(run.summary._json_dict)

        # .config contains the hyperparameters.
        #  We remove special values that start with _.
        config_list.append(
            {k: v for k,v in run.config.items()
            if not k.startswith('_')})

        # .name is the human-readable name of the run.
        name_list.append(run.name)

    runs_df = pd.DataFrame({
        "summary": summary_list,
        "config": config_list,
        "name": name_list
        })
    return runs_df

def get_model_compression(wandb_df) -> defaultdict(dict):
    all_c = [i.get("compression", 0) for i in wandb_df["summary"]]
    all_seeds = ([i["seed"] for i in wandb_df["config"]])
    all_models = [i.get("model_subtype", "contriever") for i in wandb_df["config"]]
    all_data_types = [i.get("type", "raw") for i in wandb_df["config"]]

    unique_types = set(all_data_types)
    type2model2seed2compression = {i: defaultdict(dict) for i in unique_types}

    for data_type, model,seed,compression in zip(all_data_types, all_models, all_seeds, all_c):
        type2model2seed2compression[data_type][model][f"seed_{seed}"] = compression
    
    return type2model2seed2compression