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
import datetime

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
    users = pd.read_csv('data_fork//users.csv').astype('category')
    movies = pd.read_csv('data_fork//items.csv').astype('category').drop('subcat_comercial', axis = 1)
    ratings = pd.read_csv('data_fork//purchases.csv')
    ratings['timestamp'] = [np.random.choice(pd.date_range(datetime.datetime(2013,1,1),datetime.datetime(2020,1,3))) for i in range(len(ratings))]

    # Filter the users and items that never appear in the rating table.
    distinct_users_in_ratings = ratings['UserID'].unique()
    distinct_movies_in_ratings = ratings['ItemID'].unique()
    users = users[users['UserID'].isin(distinct_users_in_ratings)]
    movies = movies[movies['ItemID'].isin(distinct_movies_in_ratings)]

    print(movies)
    print(users)

    # Group the movie features into genres (a vector), year (a category), title (a string)
    # genre_columns = movies.columns.drop(['movie_id', 'title', 'year'])
    # movies[genre_columns] = movies[genre_columns].fillna(False).astype('bool')
    # movies_categorical = movies.drop('title', axis=1)

    # Build graph
    graph_builder = PandasGraphBuilder()
    graph_builder.add_entities(users, 'UserID', 'user')
    graph_builder.add_entities(movies, 'ItemID', 'movie')
    graph_builder.add_binary_relations(ratings, 'UserID', 'ItemID', 'watched')
    graph_builder.add_binary_relations(ratings, 'ItemID', 'UserID', 'watched-by')

    g = graph_builder.build()

    # Assign features.
    # Note that variable-sized features such as texts or images are handled elsewhere.
    g.nodes['user'].data['usuario_id_crp'] = torch.LongTensor(users['usuario_id_crp'].cat.codes.values)

    g.nodes['movie'].data['salado'] = torch.LongTensor(movies['salado'].cat.codes.values)
    g.nodes['movie'].data['dulce'] = torch.LongTensor(movies['dulce'].cat.codes.values)
    g.nodes['movie'].data['frio'] = torch.LongTensor(movies['frio'].cat.codes.values)
    g.nodes['movie'].data['caliente'] = torch.LongTensor(movies['caliente'].cat.codes.values)
    g.nodes['movie'].data['harina'] = torch.LongTensor(movies['harina'].cat.codes.values)
    g.nodes['movie'].data['carne'] = torch.LongTensor(movies['carne'].cat.codes.values)
    g.nodes['movie'].data['familiar'] = torch.LongTensor(movies['familiar'].cat.codes.values)
    g.nodes['movie'].data['individual'] = torch.LongTensor(movies['individual'].cat.codes.values)
    g.nodes['movie'].data['bebestible'] = torch.LongTensor(movies['bebestible'].cat.codes.values)
    g.nodes['movie'].data['hojas'] = torch.LongTensor(movies['hojas'].cat.codes.values)
    g.nodes['movie'].data['lacteo'] = torch.LongTensor(movies['lacteo'].cat.codes.values)
    g.nodes['movie'].data['alcohol'] = torch.LongTensor(movies['alcohol'].cat.codes.values)

    g.edges['watched'].data['Q_compra'] = torch.LongTensor(ratings['Q_compra'].values)
    g.edges['watched'].data['timestamp'] = torch.LongTensor(ratings['timestamp'].values)
    g.edges['watched-by'].data['Q_compra'] = torch.LongTensor(ratings['Q_compra'].values)
    g.edges['watched-by'].data['timestamp'] = torch.LongTensor(ratings['timestamp'].values)

    # Train-validation-test split
    # This is a little bit tricky as we want to select the last interaction for test, and the
    # second-to-last interaction for validation.
    train_indices, val_indices, test_indices = train_test_split_by_time(ratings, 'timestamp', 'UserID')
    print(g)
    # Build the graph with training interactions only.
    train_g = build_train_graph(g, train_indices, 'user', 'movie', 'watched', 'watched-by')
    assert train_g.out_degrees(etype='watched').min() > 0

    # Build the user-item sparse matrix for validation and test set.
    val_matrix, test_matrix = build_val_test_matrix(g, val_indices, test_indices, 'user', 'movie', 'watched')

    ## Build title set

    # movie_textual_dataset = {'title': movies['title'].values}

    # The model should build their own vocabulary and process the texts.  Here is one example
    # of using torchtext to pad and numericalize a batch of strings.
    #     field = torchtext.data.Field(include_lengths=True, lower=True, batch_first=True)
    #     examples = [torchtext.data.Example.fromlist([t], [('title', title_field)]) for t in texts]
    #     titleset = torchtext.data.Dataset(examples, [('title', title_field)])
    #     field.build_vocab(titleset.title, vectors='fasttext.simple.300d')
    #     token_ids, lengths = field.process([examples[0].title, examples[1].title])

    ## Dump the graph and the datasets

    dgl.save_graphs(os.path.join(out_directory, 'train_g.bin'), train_g)

    dataset = {
        'val-matrix': val_matrix,
        'test-matrix': test_matrix,
        'item-texts': None,
        'item-images': None,
        'user-type': 'user',
        'item-type': 'movie',
        'user-to-item-type': 'watched',
        'item-to-user-type': 'watched-by',
        'timestamp-edge-column': 'timestamp'}

    with open(os.path.join(out_directory, 'data.pkl'), 'wb') as f:
        pickle.dump(dataset, f)
