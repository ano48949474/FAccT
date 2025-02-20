import os
import numpy as np
import pandas as pd
import networkx as nx

from node2vec import Node2Vec

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_recall_curve,
    f1_score,
    accuracy_score,
)

from structural_measures import StructuralMeasures
from fairness_metrics import (
    representation_bias,
    demographic_parity,
    equal_opportunity,
)


def measure_computer(args):
    measures_list = []
    params, G, scenario, seed = args
    os.makedirs(f"results/final_{seed}/{scenario}/{params}", exist_ok=True)

    measures_computer = StructuralMeasures(G)

    measures_list.append(
        ["closeness", measures_computer.omega(measures_computer.closeness_)]
    )
    measures_list.append(
        ["betweeness", measures_computer.omega(measures_computer.betweeness_)]
    )
    measures_list.append(
        ["prestige", measures_computer.omega(measures_computer.prestige_)]
    )
    measures_list.append(["degree", measures_computer.omega(measures_computer.degree_)])
    measures_list.append(
        ["constraint", measures_computer.omega(measures_computer.constraint_)]
    )
    measures_list.append(
        ["density", measures_computer.omega(measures_computer.density_)]
    )
    measures_list.append(
        ["heterogeneity", measures_computer.omega(measures_computer.heterogeneity_)]
    )
    measures_list.append(["isolation", measures_computer.isolation()])
    measures_list.append(["diameter", measures_computer.diameter()])
    measures_list.append(["control", measures_computer.control()])
    measures_list.append(["avg_mixed_distance", measures_computer.avg_mixed_distance()])
    measures_list.append(["assortativity", measures_computer.assortativity()])
    measures_list.append(["power_law_exponent", measures_computer.power_law_exponent()])
    measures_list.append(
        ["information_unfairness", measures_computer.information_unfairness()]
    )
    pd.DataFrame(measures_list).to_csv(
        f"results/final_{seed}/{scenario}/{params}/measures.csv"
    )


def metric_computer(args):
    metrics_list = []

    params, G, scenario, seed = args
    os.makedirs(f"results/final_{seed}/{scenario}/{params}", exist_ok=True)

    edges = list(G.edges())
    non_edges = list(nx.non_edges(G))
    np.random.shuffle(non_edges)

    sensitive_attributes = dict(
        [(node, G.nodes[node]["sensitive"]) for node in G.nodes()]
    )

    train_and_val_edges, test_edges = train_test_split(
        edges, test_size=0.2, random_state=13
    )
    train_and_val_non_edges, test_non_edges = train_test_split(
        non_edges[: len(edges)], test_size=0.2, random_state=13
    )

    train_edges, val_edges = train_test_split(
        train_and_val_edges, test_size=0.1, random_state=13
    )
    train_non_edges, val_non_edges = train_test_split(
        train_and_val_non_edges, test_size=0.1, random_state=13
    )

    y_train = [1] * len(train_edges) + [0] * len(train_non_edges)
    y_val = [1] * len(val_edges) + [0] * len(val_non_edges)
    y_test = [1] * len(test_edges) + [0] * len(test_non_edges)

    G_train = G.copy()
    G_train.remove_edges_from(test_edges + val_edges)

    def n2v():
        node2vec = Node2Vec(
            G_train,
            dimensions=4,
            walk_length=32,
            num_walks=16,
            p=2,
            q=2,
            workers=10,
            quiet=True,
        )
        model = node2vec.fit()

        embeddings = dict([(node, model.wv[str(node)]) for node in G.nodes()])
        return embeddings

    def svd():
        d = 4
        U, S, _ = np.linalg.svd(nx.to_numpy_array(G_train), full_matrices=False)
        U_k = U[:, :d]
        S_k = np.diag(S[:d])
        embeddings = np.dot(U_k, S_k)
        embeddings = dict([(list(G.nodes)[i], emb) for i, emb in enumerate(embeddings)])
        return embeddings

    for method_name, method in [("n2v", n2v), ("svd", svd)]:

        embeddings = method()

        def generate_edge_features(node1, node2, embeddings=embeddings):
            return embeddings[node1] * embeddings[node2]

        metrics_list.append(
            [
                f"representation_bias_{method_name}",
                representation_bias(
                    sensitive_attributes, generate_edge_features, list(G.edges)
                ),
            ]
        )

        X_train = [
            generate_edge_features(node1, node2)
            for node1, node2 in train_edges + train_non_edges
        ]
        X_val = [
            generate_edge_features(node1, node2)
            for node1, node2 in val_edges + val_non_edges
        ]
        X_test = [
            generate_edge_features(node1, node2)
            for node1, node2 in test_edges + test_non_edges
        ]

        logreg_model = LogisticRegression()
        logreg_model.fit(X_train, y_train)

        val_scores = logreg_model.predict_proba(X_val)[:, 1]
        precision, recall, thresholds = precision_recall_curve(y_val, val_scores)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        optimal_idx = f1_scores.argmax()
        optimal_threshold = thresholds[optimal_idx]

        test_scores = logreg_model.predict_proba(X_test)[:, 1]
        test_preds = [int(score >= optimal_threshold) for score in test_scores]

        test_predictions = [
            (edge[0], edge[1], pred)
            for edge, pred in list(zip(test_edges + test_non_edges, test_preds))
        ]

        metrics_list.append([f"f1_score_{method_name}", f1_score(y_test, test_preds)])
        metrics_list.append(
            [f"accuracy_score_{method_name}", accuracy_score(y_test, test_preds)]
        )
        metrics_list.append(
            [
                f"demographic_parity_{method_name}",
                demographic_parity(test_predictions, sensitive_attributes),
            ]
        )
        metrics_list.append(
            [
                f"equal_opportunity_{method_name}",
                equal_opportunity(test_predictions, y_test, sensitive_attributes),
            ]
        )
        pd.DataFrame(metrics_list).to_csv(
            f"results/final_{seed}/{scenario}/{params}/metrics.csv"
        )
