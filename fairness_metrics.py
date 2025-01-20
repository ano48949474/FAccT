import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def demographic_parity(predictions, sensitive_attrs):
    """
    demographic_parity fairness metric for edge prediction
    """
    group_0 = [
        pred
        for node1, node2, pred in predictions
        if sensitive_attrs[node1] == sensitive_attrs[node2]
    ]
    group_0_rate = np.mean(group_0)

    group_1 = [
        pred
        for node1, node2, pred in predictions
        if sensitive_attrs[node1] != sensitive_attrs[node2]
    ]
    group_1_rate = np.mean(group_1)

    parity_difference = group_0_rate - group_1_rate

    return parity_difference


def equal_opportunity(predictions, true_labels, sensitive_attrs):
    """
    equal_opportunity fairness metric for edge prediction
    """
    group_0_true_positives = [
        pred
        for (node1, node2, pred), true in zip(predictions, true_labels)
        if (sensitive_attrs[node1] == sensitive_attrs[node2]) and true == 1
    ]
    group_0_positive_rate = (
        np.mean(group_0_true_positives) if len(group_0_true_positives) > 0 else 0
    )

    group_1_true_positives = [
        pred
        for (node1, node2, pred), true in zip(predictions, true_labels)
        if (sensitive_attrs[node1] != sensitive_attrs[node2]) and true == 1
    ]
    group_1_positive_rate = (
        np.mean(group_1_true_positives) if len(group_1_true_positives) > 0 else 0
    )

    opportunity_difference = group_0_positive_rate - group_1_positive_rate

    return opportunity_difference


def representation_bias(sensitive_attributes, feature_aggregation_function, edges_list):
    """
    representation_bias fairness metric for edge prediction
    """
    edges_embeddings = np.array(
        [feature_aggregation_function(node1, node2) for node1, node2 in edges_list]
    )
    edges_sensitive_attributes = [
        sensitive_attributes[node1] != sensitive_attributes[node2]
        for node1, node2 in edges_list
    ]
    emb_train, emb_test, sensitive_train, sensistive_test = train_test_split(
        edges_embeddings, edges_sensitive_attributes, test_size=0.3, random_state=13
    )

    mean_sens_attribute = np.mean(sensitive_train)
    random_pred = np.random.binomial(1, mean_sens_attribute, size=len(sensistive_test))
    auc_random = roc_auc_score(sensistive_test, random_pred)
    model = LogisticRegression()
    model.fit(emb_train, sensitive_train)
    auc_model = roc_auc_score(sensistive_test, model.predict_proba(emb_test)[:, 1])
    res = auc_model - auc_random
    return res
