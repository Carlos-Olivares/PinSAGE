"""
Script that reads from raw MovieLens-1M data and dumps into a pickle
file the following:

* A heterogeneous graph with categorical features.
* A list with all the movie titles.  The movie titles correspond to
  the movie nodes in the heterogeneous graph.

This script exemplifies how to prepare tabular data with textual
features.  Since DGL graphs do not store variable-length features, we
instead put variable-length features into a more suitable container
(e.g. torchtext to handle list of texts)
"""

import os
import re
import argparse
import pickle
import pandas as pd
import numpy as np
import scipy.sparse as ssp
import dgl
import torch
import torchtext
from builder import PandasGraphBuilder
from data_utils import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=str)
    parser.add_argument('out_directory', type=str)
    args = parser.parse_args()
    directory = args.directory
    out_directory = args.out_directory
    os.makedirs(out_directory, exist_ok=True)

    ## Build heterogeneous graph

    # Load data
    users = pd.read_csv(os.path.join(directory, 'carts.csv')) #.astype('category')
    
    movies = pd.read_csv(os.path.join(directory, 'items.csv')).sort_values('ItemID', ascending = True).reset_index(drop=True)#.astype('category')
    movies = movies.drop(['subcat_comercial'], axis = 1)

    ratings = pd.read_csv(os.path.join(directory, 'purchases.csv'))#.astype('category')

    # Split train - test set
    ratings_train = ratings.query('test_set == 0').drop('test_set', axis = 1)
    ratings_test = ratings.query('test_set == 1').drop('test_set', axis = 1)

    # # Filter the users and items that never appear in the ratings_train table.
    distinct_users_in_ratings = ratings_train['venta_id_crp'].unique()
    distinct_movies_in_ratings = ratings_train['ItemID'].unique()
    users = users[users['venta_id_crp'].isin(distinct_users_in_ratings)]
    movies = movies[movies['ItemID'].isin(distinct_movies_in_ratings)]

    users = users.astype('category')
    movies = movies.astype('category')
    ratings = ratings.astype('category')

    # # Build graph
    graph_builder = PandasGraphBuilder()
    graph_builder.add_entities(users, 'venta_id_crp', 'user')
    graph_builder.add_entities(movies, 'ItemID', 'movie')
    graph_builder.add_binary_relations(ratings_train, 'venta_id_crp', 'ItemID', 'watched')
    graph_builder.add_binary_relations(ratings_train, 'ItemID', 'venta_id_crp', 'watched-by')

    g = graph_builder.build()

    # # Assign features.
    # # Note that variable-sized features such as texts or images are handled elsewhere.
    # g.nodes['user'].data['id'] = torch.LongTensor(users['venta_id_crp'].cat.codes.values)

    cat_cols = ['salado', 'dulce', 'frio', 'caliente', 'harina', 'carne', 'familiar', 'individual', 'bebestible', 'hojas', 'lacteo', 'alcohol']
    for col in cat_cols:
      g.nodes['movie'].data[col] = torch.LongTensor(movies[col].cat.codes.values)

    # g.edges['watched'].data['rating'] = torch.LongTensor(ratings['rating'].values)
    # g.edges['watched'].data['timestamp'] = torch.LongTensor(ratings['timestamp'].values)
    # g.edges['watched-by'].data['rating'] = torch.LongTensor(ratings['rating'].values)
    # g.edges['watched-by'].data['timestamp'] = torch.LongTensor(ratings['timestamp'].values)

    # # Train-validation-test split
    # # This is a little bit tricky as we want to select the last interaction for test, and the
    # # second-to-last interaction for validation.
    # train_indices, val_indices, test_indices = train_test_split_by_time(ratings, 'timestamp', 'user_id')

    # # Build the graph with training interactions only.
    train_g = g #build_train_graph(g, train_indices, 'user', 'movie', 'watched', 'watched-by')
    print(g)
    assert train_g.out_degrees(etype='watched').min() > 0

    dgl.save_graphs(os.path.join(out_directory, 'train_g.bin'), train_g)

    dataset = {
        'test-set': ratings_test,
        'user-type': 'user',
        'item-type': 'movie',
        'user-to-item-type': 'watched',
        'item-to-user-type': 'watched-by'}

    with open(os.path.join(out_directory, 'data.pkl'), 'wb') as f:
        pickle.dump(dataset, f)

    ratings_test.to_csv(os.path.join(out_directory, 'testing_set.csv'))
