# AdaVis: Adaptive and Explainable Visualization Recommendation for Tabular Data

## Introduction

This is the code repository for our Paper entitled "AdaVis: Adaptive and Explainable Visualization Recommendation for Tabular Datas". We propose an adaptive and explainable approach for recommending one or multiple visualizations for tabular datasets. AdaVis uses a knowledge graph with box embeddings and attention mechanisms to model one-to-many relations between datasets and visualizations, providing both recommendations and insights into the reasoning behind them.

## Installation

### Code

- data_loader.py: It defines custom PyTorch datasets and iterators for efficiently handling data loading, batching, and sampling of positive and negative examples during model training and testing.
- inference.py: This script uses trained knowledge graph embedding model to predict the charts for the input dataset. 
- model.py: It defines the knowledge graph embedding model and its components to learn the relationship between chart types and data characteristics.
- run.py: It sets up configurations, manages data loaders, initializes the model and training.
- weighted_sampler.py: It implements a sampler to sample the training data from the corpus.

### Key package

| Name           | Version      |
| -------------- | ------------ |
| python         | 3.7.11       |
| numpy          | 1.21.2       |
| pytorch        | 1.6.0        |
| torchvision    | 0.7.0        |
| matplotlib     | 3.5.0        |
| tensorboard    | 2.6.0        |
| tensorboardx   | 1.8          |
