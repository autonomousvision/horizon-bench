import pandas as pd
from horizon_bench.Config import WORKSPACE_PATH, PAPERS_PARQUET
import numpy as np


def preprocess_iclr_dataset(best_papers = True, start_idx=0, end_idx=100):
    # manual checks result in the exclusion of the following papers, by id:
    excluded_papers = [39370,]

    df = pd.read_parquet(f"{WORKSPACE_PATH}/data/iclr.parquet")
    df = df[~df['decision'].isin(['Withdrawn', 'Desk Reject'])]

    df = df[~df.index.isin(excluded_papers)]
    df = df[df.year == 2026]

    df['scores'] = df['scores'].apply(lambda x: x.tolist())
    df = df.sort_values(by='scores', key=lambda x: x.apply(lambda s: np.mean(s) if isinstance(s, list) and len(s) > 0 else -1), ascending=False)

    if best_papers:
        df = df.iloc[start_idx:end_idx] # take top 40 papers
        filepath = f"{WORKSPACE_PATH}/data/iclr_preprocessed_{start_idx}_{end_idx}.parquet"
        df.to_parquet(filepath)
    else:
        df = df.iloc[-end_idx:-start_idx] # take bottom 40 papers
        filepath = f"{WORKSPACE_PATH}/data/iclr_worst_preprocessed_{start_idx}_{end_idx}.parquet"
        df.to_parquet(filepath)

    print(df[['id', 'title', 'scores', 'decision']].head())
    return filepath

def load_papers():
    df = pd.read_parquet(PAPERS_PARQUET)
    return df

if __name__ == "__main__":
    preprocess_iclr_dataset(best_papers=True, start_idx=0, end_idx=100)
    # preprocess_iclr_dataset(best_papers=False)