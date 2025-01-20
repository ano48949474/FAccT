import numpy as np
import networkx as nx
import warnings
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore", category=FutureWarning, module="networkx")


class StructuralMetrics:
    """
    Implementation of structural bias measures
    """

    def __init__(self, G):
        self.G = G
        self.nodes = list(G.nodes())
        self.neighbors_dict = {n: list(G.neighbors(n)) for n in G.nodes()}
        self.sensitive_nodes = [i for i in self.nodes if G.nodes[i]["sensitive"] == 1]
        self.non_sensitive_nodes = [
            i for i in self.nodes if G.nodes[i]["sensitive"] == 0
        ]

        self.clustering_coeffs = nx.clustering(G)
        self.betweeness_centrality = nx.betweenness_centrality(self.G)
        self.closeness_centrality = nx.closeness_centrality(self.G)
        self.prestige_centrality = nx.eigenvector_centrality(G, max_iter=int(10e6))
        self.shortest_paths = dict(nx.shortest_path(G))
        self.resistance_distances = nx.resistance_distance(G)

    def omega(self, f):
        sensitive_values = [f(v) for v in self.sensitive_nodes]
        non_sensitive_values = [f(v) for v in self.non_sensitive_nodes]

        mean_sensitive = np.mean(sensitive_values)
        mean_non_sensitive = np.mean(non_sensitive_values)

        return (mean_non_sensitive - mean_sensitive) / np.mean(
            mean_non_sensitive + mean_sensitive
        )

    def omega_effective_resistance(self, f):
        sensitive_values = [f(v) for v in self.sensitive_nodes]
        non_sensitive_values = [f(v) for v in self.non_sensitive_nodes]

        mean_sensitive = np.mean(sensitive_values)
        mean_non_sensitive = np.mean(non_sensitive_values)

        return np.abs(mean_sensitive - mean_non_sensitive)

    #########################################################

    def degree_(self, node):
        return len(self.neighbors_dict[node])

    def degree(self):
        return self.omega(self.degree_)

    ###########################

    def constraint_(self, node):
        return np.sum([len(self.neighbors_dict[v]) for v in self.neighbors_dict[node]])

    def constraint(self):
        return self.omega(self.constraint_)

    ###########################

    def clustering_coefficient_(self, node):
        return self.clustering_coeffs[node]

    def clustering_coefficient(self):
        return self.omega(self.clustering_coefficient_)

    ###########################

    def density_(self, node):
        sub_G = self.G.subgraph(self.neighbors_dict[node] + [node])
        return nx.density(sub_G)

    def density(self):
        return self.omega(self.density_)

    ###########################

    def prestige_(self, node):
        return self.prestige_centrality[node]

    def prestige(self):
        return self.omega(self.prestige_)

    ###########################

    def get_power_law_coeff_from_deg_list(self, deg_list):

        deg_values = list(range(np.min(deg_list), np.max(deg_list) + 1))
        counts = [deg_list.count(i) for i in deg_values]

        deg_points = [i for i in list(zip(deg_values, counts)) if i[1] > 0]
        log_deg_x = [[np.log(i[0])] for i in deg_points]
        log_deg_y = [np.log(i[1]) for i in deg_points]

        lin_reg = LinearRegression()
        lin_reg.fit(log_deg_x, log_deg_y)
        return -lin_reg.coef_[0]

    def power_law_exponent(self):
        degrees = dict(nx.degree(self.G))
        sensitive_degrees_list = [
            degrees[node] for node in self.nodes if self.G.nodes[node]["sensitive"] == 1
        ]
        non_sensitive_degrees_list = [
            degrees[node] for node in self.nodes if self.G.nodes[node]["sensitive"] == 0
        ]

        return self.get_power_law_coeff_from_deg_list(
            sensitive_degrees_list
        ) / self.get_power_law_coeff_from_deg_list(non_sensitive_degrees_list)

    ###########################

    def betweeness_(self, node):
        return self.betweeness_centrality[node]

    def betweeness(self):
        return self.omega(self.betweeness_)

    ###########################

    def closeness_(self, node):
        return self.closeness_centrality[node]

    def closeness(self):
        return self.omega(self.closeness_)

    ###########################

    def heterogeneity_(self, node):
        mean_neigh_sensitive = np.mean(
            [self.G.nodes[v]["sensitive"] for v in self.neighbors_dict[node]]
        )
        return 1 - 2 * np.abs(mean_neigh_sensitive - 0.5)

    def heterogeneity(self):
        return self.omega(self.heterogeneity_)

    ###########################
    def assortativity(self):
        return nx.attribute_assortativity_coefficient(self.G, "sensitive")

    ###########################

    def avg_mixed_distance(self):

        distances_between_groups = []
        for u in self.sensitive_nodes:
            for v in self.non_sensitive_nodes:
                try:
                    distances_between_groups.append(len(self.shortest_paths[u][v]))
                except KeyError:
                    pass

        return np.mean(distances_between_groups)

    ###########################
    # Effective Resistance measures

    def isolation_(self, v):
        res = np.mean([self.resistance_distances[v][u] for u in self.G.nodes])
        return res

    def isolation(self):
        return self.omega_effective_resistance(self.isolation_)

    def diameter_(self, v):
        res = np.max([self.resistance_distances[v][u] for u in self.G.nodes])
        return res

    def diameter(self):
        return self.omega_effective_resistance(self.diameter_)

    def control_(self, v):
        res = np.sum(
            [self.resistance_distances[v][u] for u in list(nx.neighbors(self.G, v))]
        )
        return res

    def control(self):
        return self.omega_effective_resistance(self.control_)

    ###########################
    # information unfairness

    def max_distance(self, a, b, c):
        distance_1 = abs(a - b)
        distance_2 = abs(a - c)
        distance_3 = abs(b - c)
        return max(distance_1, distance_2, distance_3)

    def information_unfairness(self):

        adjacency_matrix = nx.adjacency_matrix(self.G).toarray()
        accessibility_matrix = (
            (1 / 2) * adjacency_matrix
            + (1 / 3) * (adjacency_matrix.dot(adjacency_matrix))
            + (1 / 4) * (adjacency_matrix.dot(adjacency_matrix).dot(adjacency_matrix))
        )
        np.fill_diagonal(accessibility_matrix, 0)

        #############################################################
        nodes_self = list(self.G.nodes())
        category_1_nodes = [i for i in nodes_self if self.G.nodes[i]["sensitive"] == 1]
        category_2_nodes = [i for i in nodes_self if self.G.nodes[i]["sensitive"] == 0]

        intra_category_mask = np.zeros_like(accessibility_matrix)
        for node_i in category_1_nodes:
            for node_j in category_1_nodes:
                intra_category_mask[node_i, node_j] = 1
        intra_category_accessibility_1 = np.multiply(
            accessibility_matrix, intra_category_mask
        )
        average_intra_category_accessibility_1 = np.mean(intra_category_accessibility_1)

        intra_category_mask = np.zeros_like(accessibility_matrix)
        for node_i in category_2_nodes:
            for node_j in category_2_nodes:
                intra_category_mask[node_i, node_j] = 1
        intra_category_accessibility_2 = np.multiply(
            accessibility_matrix, intra_category_mask
        )
        average_intra_category_accessibility_2 = np.mean(intra_category_accessibility_2)

        intra_category_mask = np.zeros_like(accessibility_matrix)
        for node_i in category_1_nodes:
            for node_j in category_2_nodes:
                intra_category_mask[node_i, node_j] = 1
        intra_category_accessibility_1_2 = np.multiply(
            accessibility_matrix, intra_category_mask
        )
        average_intra_category_accessibility_1_2 = np.mean(
            intra_category_accessibility_1_2
        )

        value = self.max_distance(
            average_intra_category_accessibility_1,
            average_intra_category_accessibility_2,
            average_intra_category_accessibility_1_2,
        )
        return value
