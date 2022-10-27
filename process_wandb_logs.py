import argparse
import pickle
from utils.wandb_utils import get_wandb_summary, get_model_compression


def setup_argparse():
    p = argparse.ArgumentParser()
    p.add_argument('--output', default='type2model2seed2compression.pkl', help="place to save output dict")
    p.add_argument('-pn','--project_name', default='seraphinatarrant/Bias in Bios MDL probing', help="wandb project name")
    return p.parse_args()


if __name__ == "__main__":
    
    args = setup_argparse()
    runs_df = get_wandb_summary(project_name=args.project_name)
    t2m2s2c = get_model_compression(runs_df)
    with open(args.output, "wb") as fout: 
        pickle.dump(t2m2s2c, fout)
