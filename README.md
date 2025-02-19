# Topological Fairness in Graphs: Exploring Structural Bias and Its Impact on Link Prediction

This anonymized repository provides Python code to reproduce experiments from the paper _"Topological Fairness in Graphs: Exploring Structural Bias and Its Impact on Link Prediction"_, submited to The 2025 ACM Conference of Fairness, Accountability, and Transparency (FAccT' 25).

## Abstract

A common assumption in fair link prediction is that graphs' structural properties explain disparities in predictive performance. While previous work has largely focused on the tendency of nodes with similar attributes to connect - captured by measures such as assortativity - as a proxy for structural bias, we propose a framework that introduces additional structural measures to better capture the relationship between graph topology and group fairness in link prediction. In order to provide a more comprehensive understanding of this phenomenon, our framework addresses both measurement and data generation issues. First, we introduce a series of structural bias measures, inspired among others by concepts from social capital theory, as well as graph generation methods designed to allow fine-grained control of graphs' key topological aspects. This makes it possible to simulate diverse and realistic graph structures with different levels of bias. Next, we conduct an in-depth analysis of the correlation between these bias measurements and the fairness of link prediction approaches. When quantitatively analyzing the relationships between structural bias measures and fairness metrics, our findings first show that predictors' unfairness can very often be deduced from the structure of the graph, but also challenge the systematic reliance on assortativity as the sole canonical measure of structural bias. We demonstrate that while assortativity remains important, complementing it with additional measures is essential for capturing complementary dimensions of the topological dynamics influencing the fairness of predictive models.

## Environment
```
joblib==1.4.2
networkx==3.3
node2vec==0.5.0
numpy==1.26.4
pandas==2.2.2
scikit-learn==1.5.0
tqdm==4.66.4
```

## Running the code

```
python main.py
```
