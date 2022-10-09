#!/usr/bin/env python3
'''
Copyright 2019-2021 Duncan Deveaux

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
'''

import sys
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
sys.path.append('..')
import tools.exit_model as model
from tools.locations import get_input_interaction, get_input_for_location
from mi import get_roundabouts_geometry, get_roundabouts_inputs, get_similarity_groups, reformat_index, create_dir


def load_matrix(path):
    '''Load acc matrix data'''
    data = []
    inputs = os.listdir(path)
    for i in inputs:
        data.append(pd.read_json('{}/{}'.format(path, i)))

    if len(data) == 0:
        sys.exit('No {} input file located.'.format(path))

    r_names = list(data[0])
    r_names.remove('SHUFFLED')

    res_matrix = pd.DataFrame(columns=r_names, index=r_names)
    for i in range(len(r_names)):
        for j in range(i + 1, len(r_names)):
            key1, key2 = r_names[i], r_names[j]

            values12, values21 = [], []
            values11, values22 = [], []
            for matrix in data:
                values12.append(matrix.at[key1, key2])
                values21.append(matrix.at[key2, key1])
                values11.append(matrix.at[key1, key1])
                values22.append(matrix.at[key2, key2])

            values12 = np.array(values12)
            values21 = np.array(values21)
            values11 = np.array(values11)
            values22 = np.array(values22)

            res_matrix.at[key1, key2] = np.mean(values12)
            res_matrix.at[key2, key1] = np.mean(values21)
            res_matrix.at[key1, key1] = np.mean(values11)
            res_matrix.at[key2, key2] = np.mean(values22)

    return res_matrix

def get_clusters(cluster_matrix, nb_clusters, method='ward'):

    fl = []
    if method != 'DBSCAN':
        linkage_matrix = linkage(squareform(cluster_matrix), method=method)
        fl = fcluster(linkage_matrix, nb_clusters, criterion='maxclust')
    else:
        dbscan = DBSCAN(metric='precomputed', min_samples=1, eps=5.0)
        dbscan.fit(cluster_matrix)
        fl = dbscan.labels_

    clusters = {}
    for (i, clust_id) in enumerate(fl):
        if clust_id not in clusters:
            clusters[clust_id] = []
        clusters[clust_id].append(cluster_matrix.index[i])

    return clusters

def get_similarity_clusters(clusters_list, key):

    similar_keys = []
    distant_keys = []

    for clust_id in clusters_list:
        if key in clusters_list[clust_id]:
            for member in clusters_list[clust_id]:
                if member != key:
                    similar_keys.append(member)
        else:
            distant_keys.extend(clusters_list[clust_id])

    return (pd.DataFrame(index=similar_keys), pd.DataFrame(index=distant_keys))


def generate_training_set(
        roundabouts_data,
        targetkey,
        other_rds,
        target_proportion):

    # Proportion from the trained roundabout.
    amount_target = int(TRAINING_SIZE * target_proportion)
    amount_all_other = int(TRAINING_SIZE - amount_target)

    if len(other_rds) == 0:
        raise ValueError('No other roundabout to use / {}.'.format(targetkey))

    amount_other = int(amount_all_other / len(other_rds))
    #print ('Target: {}, other: {}, {}x{}'.format(amount_target, amount_all_other, amount_other, len(other_rds)))

    set_target = roundabouts_data[targetkey][0:amount_target]
    set_other = roundabouts_data[other_rds[0]][0:amount_other]
    #print ('Initial {} entries from {}'.format(amount_other, other_rds[0]))
    for i in range(1, len(other_rds)):
        #print ('Adding {} entries from {}'.format(amount_other, other_rds[i]))
        set_other = np.append(
            set_other, roundabouts_data[other_rds[i]][0:amount_other], axis=0)

    return np.append(set_target, set_other, axis=0)


def generate_training_sets(
        roundabouts_data,
        targetkey,
        similar,
        distant,
        other,
        target_proportion):
    return (
        generate_training_set(
            roundabouts_data,
            targetkey,
            similar,
            target_proportion),
        generate_training_set(
            roundabouts_data,
            targetkey,
            distant,
            target_proportion),
        generate_training_set(
            roundabouts_data,
            targetkey,
            other,
            target_proportion))


def train_and_evaluate(training_set, validation_set):
    # Extract validation and training sets.

    x_training, y_training = training_set[:, [0, 1, 2]], training_set[:, 3]
    x_validation, y_validation = validation_set[:, [
        0, 1, 2]], validation_set[:, 3]

    #print('{}: len training: {}, valid: {}'.format(key, len(y_training), len(y_validation)))

    if len(y_validation) < 100:
        raise ValueError(
            'not enough validation data ({}, {})'.format(
                key, len(y_validation)))

    # Perform regression
    regression = LogisticRegression()
    regression.fit(x_training, y_training)

    prediction = regression.predict(x_validation)

    return {'accuracy': accuracy_score(prediction, y_validation),
            'recall': recall_score(prediction, y_validation),
            'precision': precision_score(prediction, y_validation),
            'f1': f1_score(prediction, y_validation)}


def shuffle_data(roundabouts_data):
    for key in roundabouts_data:
        np.random.shuffle(roundabouts_data[key])


def start_evaluation(
    roundabouts_data,
    geometry,
    targetkey,
    proportions=[
        0.0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0]):
    (similar, distant) = get_similarity_groups(geometry, targetkey)
    other = pd.concat([similar, distant])

    validation_set = roundabouts_data[targetkey][TRAINING_SIZE +
                                                 1:TRAINING_SIZE + EVALUATION_SIZE + 1]

    results = pd.DataFrame(
        columns=[
            'target',
            'target_proportion',
            'mode',
            'accuracy',
            'recall',
            'precision',
            'f1'])
    for p in proportions:
        (training_similar, training_distant, training_other) = generate_training_sets(
            roundabouts_data, targetkey, similar.index, distant.index, other.index, p)

        data_similar = train_and_evaluate(training_similar, validation_set)
        data_similar['target'] = targetkey
        data_similar['target_proportion'] = p
        data_similar['mode'] = 'similar'

        data_distant = train_and_evaluate(training_distant, validation_set)
        data_distant['target'] = targetkey
        data_distant['target_proportion'] = p
        data_distant['mode'] = 'distant'

        data_other = train_and_evaluate(training_other, validation_set)
        data_other['target'] = targetkey
        data_other['target_proportion'] = p
        data_other['mode'] = 'other'

        results = results.append(data_similar, ignore_index=True)
        results = results.append(data_distant, ignore_index=True)
        results = results.append(data_other, ignore_index=True)

    return results


def plot_result(results, mode, metric, color):
    df_mode = results[results['mode'] == mode]

    df_mean = df_mode.groupby('target_proportion').mean()
    df_std = df_mode.groupby('target_proportion').std()
    df_count = df_mode.groupby('target_proportion').count()

    x = df_mode['target_proportion'].drop_duplicates()
    y = df_mean[metric]

    error = []

    for proportion in df_count.index:
        n = df_count.at[proportion, metric]
        std = df_std.at[proportion, metric]
        t = stats.t.ppf(0.95, n - 1)
        error.append(t * std / np.sqrt(n))

    plt.fill_between(x, y - error, y + error, alpha=0.2, color=color)
    plt.plot(x, y, label='{} - {}'.format(mode, metric), color=color)


def compute_and_plot(roundabouts_data, geo_df, targetkey, metrics):
    results = pd.DataFrame(
        columns=[
            'target',
            'target_proportion',
            'mode',
            'accuracy',
            'recall',
            'precision',
            'f1'])
    for i in range(20):
        results = pd.concat([results, start_evaluation(
            roundabouts_data, geo_df, targetkey)])
        shuffle_data(roundabouts_data)

    for metric in metrics:

        title_text = 'Exit Probability Model Training Performance on {}\nusing Fractions of Training Data Extracted from Distinct Roundabouts'.format(
            reformat_index(targetkey))
        metric_text = 'Model Accuracy Score'
        if metric == 'f1':
            metric_text = 'Model F1-Score'
        elif metric == 'precision':
            metric_text = 'Precision Score'

        plt.suptitle(title_text, y=1.012)
        plt.title(
            'Similarity Condition: ΔEntries = 0 AND ΔRadius ≤ 6.0m',
            fontsize=11)
        plot_result(results, 'similar', metric, 'blue')
        plot_result(results, 'distant', metric, 'red')
        plot_result(results, 'other', metric, 'green')
        plt.legend()
        plt.ylabel(metric_text)
        plt.xlabel(
            'Proportion of training data extracted from {}'.format(
                reformat_index(targetkey)))
        plt.savefig(
            'training_stats/{}_{}.svg'.format(targetkey, metric))
        plt.close()

if __name__ == '__main__':
    create_dir('training_stats')

    TRAINING_SIZE = 5000
    EVALUATION_SIZE = 1000

    acc_matrix = load_matrix('mi_data/mi_acc_5000')
    recall_matrix = load_matrix('mi_data/mi_recall_5000')
    precision_matrix = load_matrix('mi_data/mi_precision_5000')
    f1_matrix = load_matrix('mi_data/mi_f1_5000')

    mi_matrix = load_matrix('mi_data/mi_5000')
    cluster_matrix = mi_matrix.applymap(lambda x: 1.0 / x)
    indexes = cluster_matrix.index
    for i in range(len(indexes)):
        for j in range(i, len(indexes)):
            cluster_matrix.at[indexes[j], indexes[i]
                              ] = cluster_matrix.at[indexes[i], indexes[j]]
            cluster_matrix.at[indexes[i], indexes[i]] = 0.0
            cluster_matrix.at[indexes[j], indexes[j]] = 0.0

    # Loading geometry data
    geo_df = get_roundabouts_geometry()
    geo_df.to_csv('geometry.csv')

    # Training data
    randseed = int(time.time())

    roundabouts = get_roundabouts_inputs()
    roundabouts_data = {}
    for key in roundabouts:
        # Preprocess paths for interaction roundabouts
        if roundabouts[key]['interaction']:

            roundabouts[key]['input'] = []
            input_raw = get_input_interaction(roundabouts[key]['name'])
            for value in input_raw:
                roundabouts[key]['input'].append(
                    roundabouts[key]['name'] + '_' + os.path.basename(value))
        else:
            roundabouts[key]['input'] = get_input_for_location(
                roundabouts[key]['name'])

        # Extract training data
        basepath = '../round/'
        if roundabouts[key]['interaction']:
            basepath += '../interaction/'

        data = np.array(model.gather_training_data(roundabouts[key]['input'],
                                                   randseed,
                                                   roundabouts[key]['interaction'],
                                                   basepath, filterout=False))

        # Normalize training data
        data = np.delete(data, 2, axis=1)  # Remove absolute distance data
        data[:, 0] /= data[:, 0].max()

        df = pd.DataFrame(data=data, index=None, columns=['Lane',
                                                          'Heading',
                                                          'DistanceRel',
                                                          'MeanApproachSpeed',
                                                          'MeanDensity',
                                                          'Flow',
                                                          'Capacity_German',
                                                          'Capacity_HCM2010',
                                                          'FOC_German',
                                                          'FOC_HCM2010',
                                                          'NextExit'])

        training_cols = ['Lane', 'Heading', 'DistanceRel', 'NextExit']

        data_np = df[training_cols].to_numpy()
        np.random.shuffle(data_np)

        roundabouts_data[key] = data_np


    for key in roundabouts_data:

        (similar, distant) = get_similarity_groups(geo_df, key)
        if len(similar.index) == 0:
            print('{}: No similar roundabouts.'.format(key))
            continue
        elif len(distant.index) == 0:
            print('{}: No distant roundabouts.'.format(key))
            continue

        print('== {} =='.format(key))
        print(similar)
        print(distant)
        print(
            'similar: mean radius diff: {}, mean entries diff: {}, mean width diff: {}'.format(
                similar['RADIUS_DIFF'].mean(),
                similar['ENTRIES_DIFF'].mean(),
                similar['WIDTH_DIFF'].mean()))
        print(
            'distant: mean radius diff: {}, mean entries diff: {}, mean width diff: {}'.format(
                distant['RADIUS_DIFF'].mean(),
                distant['ENTRIES_DIFF'].mean(),
                distant['WIDTH_DIFF'].mean()))

        compute_and_plot(
            roundabouts_data, geo_df, key, [
                'accuracy', 'f1', 'precision'])
