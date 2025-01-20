import numpy as np
from tqdm import tqdm
from itertools import product
from graph_generator import generator


def get_grid(param_values):
    """
    Get Homophily and size ratio grid
    """
    grid = list(
        product(
            param_values["size_ratio"],
            param_values["beta"],
        )
    )
    return grid


def generate_graphs_grid(scenario, n_params_values, seed):
    """
    Generate a list of graphs based on different scenarios and parameter grids.
    the fixed parameters (n, m, scale) have been set to reproduce the 3 datsets at hand

    This function creates a grid of parameter values for the specified scenario
    and generates graphs using the `generator` function. The generated graphs are
    paired with their corresponding parameter values.

    Parameters:
    ----------
    scenario : str
        The type of scenario to generate graphs for. Supported scenarios are:
        - "political": Creates graphs with a political blog context.
        - "social": Creates graphs with a social context.
        - "collab": Creates graphs for a collaboration network.
    n_params_values : int
        The number of distinct values for each parameter in the grid.
    seed : int
        The random seed for reproducibility when generating graphs.

    Returns:
    -------
    list
        A list of tuples, where each tuple contains:
        - A tuple of parameter values (size_ratio, beta).
        - A generated graph object.
    """
    if scenario == "social":
        param_values = {
            "size_ratio": [round(i, 3) for i in np.linspace(0.5, 0.9, n_params_values)],
            "beta": [
                round(i, 3) for i in np.linspace(0, 3.8, n_params_values)
            ],
        }
    else:
        param_values = {
            "size_ratio": [round(i, 3) for i in np.linspace(0.5, 0.9, n_params_values)],
            "beta": [
                round(i, 3) for i in np.linspace(0, 8, n_params_values)
            ],
        }
    if scenario == "political":
        G_list = [
            (
                (size_ratio, beta),
                generator(
                    1222,
                    m=14,
                    size_ratio=size_ratio,
                    beta=beta,
                    scale=0.08,
                    scenario="political",
                    seed=seed,
                ),
            )
            for size_ratio, beta in tqdm(get_grid(param_values))
        ]
    elif scenario == "social":

        G_list = [
            (
                (size_ratio, beta),
                generator(
                    1034,
                    m=13,
                    size_ratio=size_ratio,
                    beta=beta,
                    scenario="social",
                    seed=seed,
                ),
            )
            for size_ratio, beta in tqdm(get_grid(param_values))
        ]

    elif scenario == "collab":
        G_list = [
            (
                (size_ratio, beta),
                generator(
                    860,
                    m=3,
                    size_ratio=size_ratio,
                    beta=beta,
                    scenario="collab",
                    scale=1,
                    seed=seed,
                ),
            )
            for size_ratio, beta in tqdm(get_grid(param_values))
        ]

    return G_list
