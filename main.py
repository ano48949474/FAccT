import pickle
import os
from tqdm import tqdm

from joblib import Parallel, delayed

from bias_computers import measure_computer, metric_computer
from helpers import generate_graphs_grid

n_params_values = 2
n_seeds = 2

if __name__ == "__main__":

    for seed in range(n_seeds):
        os.makedirs(f"results/final_{seed}", exist_ok=True)
        for scenario in ["political", "social", "collab"]:

            os.makedirs(f"results/final_{seed}/{scenario}", exist_ok=True)
            os.makedirs(f"results/final_{seed}/{scenario}/graphs", exist_ok=True)
            os.makedirs(f"results/final_{seed}/{scenario}/measures", exist_ok=True)
            os.makedirs(f"results/final_{seed}/{scenario}/metrics", exist_ok=True)

            print(f"Scenario : {scenario}")
            G_list = generate_graphs_grid(scenario, n_params_values, seed)

            with open(
                f"results/final_{seed}/{scenario}/graphs/G_list.pkl", "wb"
            ) as file:
                pickle.dump(G_list, file)

            G_list = [(i[0], i[1], scenario, seed) for i in G_list]
            results_f1 = Parallel(n_jobs=-1)(
                delayed(measure_computer)(args)
                for args in tqdm(G_list, desc="Processing measures")
            )

            results_f2 = Parallel(n_jobs=-1)(
                delayed(metric_computer)(args)
                for args in tqdm(G_list, desc="Processing metrics")
            )
