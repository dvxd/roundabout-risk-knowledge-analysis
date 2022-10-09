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

import os
import time
import numpy as np
import scipy.stats
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from EDGE.EDGE_4_4_1 import EDGE
import sys
sys.path.append('..')
from tools.locations import get_topology_interaction, get_topology_for_location, get_input_interaction, get_input_for_location
import tools.exit_model as model


def get_roundabouts_inputs():
    return {
        'INT_USA_FT': {
            'interaction': True,
            'input_raw': get_input_interaction('DR_USA_Roundabout_FT'),
            'name': 'DR_USA_Roundabout_FT'},
        'INT_USA_SR': {
            'interaction': True,
            'input_raw': get_input_interaction('DR_USA_Roundabout_SR'),
            'name': 'DR_USA_Roundabout_SR'},
        'INT_USA_EP': {
            'interaction': True,
            'input_raw': get_input_interaction('DR_USA_Roundabout_EP'),
            'name': 'DR_USA_Roundabout_EP'},
        'INT_CHN_LN': {
            'interaction': True,
            'input_raw': get_input_interaction('DR_CHN_Roundabout_LN'),
            'name': 'DR_CHN_Roundabout_LN'},
        'INT_DEU_OF': {
            'interaction': True,
            'input_raw': get_input_interaction('DR_DEU_Roundabout_OF'),
            'name': 'DR_DEU_Roundabout_OF'},
        'RD_0': {
            'interaction': False,
            'input': get_input_for_location(0)},
        'RD_1': {
            'interaction': False,
            'input': get_input_for_location(1)},
        'RD_2': {
            'interaction': False,
            'input': get_input_for_location(2)}}


def get_roundabouts_geometry():
    roundabouts = {'INT_USA_FT': {'COUNTRY': 'USA', 'interaction': True, 'name': 'DR_USA_Roundabout_FT'},
                   'INT_USA_SR': {'COUNTRY': 'USA', 'interaction': True, 'name': 'DR_USA_Roundabout_SR'},
                   'INT_USA_EP': {'COUNTRY': 'USA', 'interaction': True, 'name': 'DR_USA_Roundabout_EP'},
                   'INT_CHN_LN': {'COUNTRY': 'CHN', 'interaction': True, 'name': 'DR_CHN_Roundabout_LN'},
                   'INT_DEU_OF': {'COUNTRY': 'DEU', 'interaction': True, 'name': 'DR_DEU_Roundabout_OF'},
                   'RD_0': {'COUNTRY': 'DEU', 'interaction': False, 'name': 0},
                   'RD_1': {'COUNTRY': 'DEU', 'interaction': False, 'name': 1},
                   'RD_2': {'COUNTRY': 'DEU', 'interaction': False, 'name': 2}}

    for key in roundabouts:
        topology = None
        if roundabouts[key]['interaction']:
            topology = get_topology_interaction(roundabouts[key]['name'])
        else:
            topology = get_topology_for_location(roundabouts[key]['name'])

        roundabouts[key]['ENTRIES_COUNT'] = len(topology.entry_lanescount)
        roundabouts[key]['CIRCULAR_LANES_COUNT'] = topology.real_lanes_count
        roundabouts[key]['RADIUS'] = topology.circular_lanes[0].radius_begin
        roundabouts[key]['WIDTH'] = topology.circular_lanes[-1].radius_end - \
            roundabouts[key]['RADIUS']

        del roundabouts[key]['interaction']
        del roundabouts[key]['name']

    return pd.DataFrame.from_dict(roundabouts, orient='index')

def reformat_index(item):
    if item == 'RD_0':
        return 'RounD_0'
    elif item == 'RD_1':
        return 'RounD_1'
    elif item == 'RD_2':
        return 'RounD_2'
    elif item.startswith('INT_'):
        return item[4:]

def get_similarity_groups(geometry, key):
    entries = geometry.at[key, 'ENTRIES_COUNT']
    country = geometry.at[key, 'COUNTRY']
    radius = geometry.at[key, 'RADIUS']
    width = geometry.at[key, 'WIDTH']

    df = geometry.drop(key)
    df['ENTRIES_DIFF'] = (df['ENTRIES_COUNT'] - entries).abs()
    df['RADIUS_DIFF'] = (df['RADIUS'] - radius).abs()
    df['WIDTH_DIFF'] = (df['WIDTH'] - width).abs()
    df['SAME_COUNTRY'] = (df['COUNTRY'] == country)

    df_similar = df[(df['ENTRIES_DIFF'] < 1) & (df['RADIUS_DIFF'] <= 6.0)]
    #df_similar = pd.concat( [df[(df['ENTRIES_DIFF'] < 1) & (df['RADIUS_DIFF'] < 8.12)], df[(df['ENTRIES_DIFF'] >= 1) & (df['WIDTH_DIFF'] < 1.125)]] ).drop_duplicates()
    df_distant = pd.concat([df, df_similar]).drop_duplicates(keep=False)

    return (df_similar, df_distant)

def create_dir(path):
    try:
        os.mkdir(path)
    except FileExistsError:
        pass


def entropy(labels, base=2):
    value, counts = np.unique(labels, return_counts=True)
    return scipy.stats.entropy(counts, base=base)


def prepare_data(roundabouts, key, training_size, evaluation_size):
    # Extract validation and training sets.
    (x_training, y_training, x_validation, y_validation) = model.process_training_data(
        roundabouts[key]['data'], training_size, evaluation_size)
    print('{}: len training: {}, valid: {}'.format(
        key, len(y_training), len(y_validation)))

    if len(y_validation) < 100:
        del roundabouts[key]
        return

    roundabouts[key]['validation'] = (x_validation, y_validation)

    # Perform regression
    regression = LogisticRegression()
    regression.fit(x_training, y_training)
    #true_class = [i for i in range(len(regression.classes_)) if regression.classes_[i] == True][0]

    accuracy = regression.score(x_validation, y_validation)
    print("{} accuracy: {}".format(key, accuracy))

    roundabouts[key]['classifier'] = regression
    return (x_training, y_training)


def classify(items, classifier):
    res = []

    for item in items:
        sample = np.array([item])
        proba = classifier.predict_proba(sample)
        res.append([np.round(proba[0][1], 1)])

    return np.array(res)


if __name__ == '__main__':

    TRAINING_SIZE = 5000
    EVALUATION_SIZE = 1000

    # Create the target directories if needed
    create_dir('mi_data')
    create_dir('mi_data/mi_MI_{}'.format(TRAINING_SIZE))
    create_dir('mi_data/mi_{}'.format(TRAINING_SIZE))
    create_dir('mi_data/mi_acc_{}'.format(TRAINING_SIZE))
    create_dir('mi_data/mi_mse_{}'.format(TRAINING_SIZE))
    create_dir('mi_data/mi_recall_{}'.format(TRAINING_SIZE))
    create_dir('mi_data/mi_precision_{}'.format(TRAINING_SIZE))
    create_dir('mi_data/mi_f1_{}'.format(TRAINING_SIZE))

    # 1. Training classifiers.
    randseed = int(time.time())
    print('\nUsing seed {}...'.format(randseed))

    roundabouts = get_roundabouts_inputs()
    roundabouts_data = {}

    all_training_data = []
    for key in roundabouts:
        # Preprocess paths for interaction roundabouts
        if roundabouts[key]['interaction']:

            roundabouts[key]['input'] = []
            for value in roundabouts[key]['input_raw']:
                roundabouts[key]['input'].append(
                    roundabouts[key]['name'] + '_' + os.path.basename(value))

            del roundabouts[key]['input_raw']
            del roundabouts[key]['name']

        # Extract training data
        basepath = '../round/'
        if roundabouts[key]['interaction']:
            basepath = '../interaction/'

        data = np.array(
            model.gather_training_data(
                roundabouts[key]['input'],
                randseed,
                roundabouts[key]['interaction'],
                basepath,
                filterout=False))

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

        roundabouts_data[key] = {'data': data_np}
        (x_training, y_training) = prepare_data(
            roundabouts_data, key, TRAINING_SIZE, EVALUATION_SIZE)

        training_data = np.append(
            x_training, np.transpose(
                [y_training]), axis=1)
        if len(all_training_data) == 0:
            all_training_data = training_data
        else:
            all_training_data = np.append(
                all_training_data, training_data, axis=0)

    for shuffled in ['SHUFFLED']:
        np.random.shuffle(all_training_data)
        roundabouts_data[shuffled] = {'data': all_training_data}
        prepare_data(
            roundabouts_data,
            shuffled,
            TRAINING_SIZE,
            EVALUATION_SIZE)

    # 2. Computing MI for all pairs of roundabouts
    r_names = list(roundabouts_data.keys())
    MI_matrix = pd.DataFrame(columns=r_names, index=r_names)
    mi_matrix = pd.DataFrame(columns=r_names, index=r_names)
    mse_matrix = pd.DataFrame(columns=r_names, index=r_names)
    acc_matrix = pd.DataFrame(columns=r_names, index=r_names)
    recall_matrix = pd.DataFrame(columns=r_names, index=r_names)
    precision_matrix = pd.DataFrame(columns=r_names, index=r_names)
    f1_matrix = pd.DataFrame(columns=r_names, index=r_names)

    ''' overall evaluation set
    evaluation_set = np.array(roundabouts_data[r_names[0]]['validation'][0])
    for i in range(1, len(r_names)):
        evaluation_set = np.append(evaluation_set, roundabouts_data[r_names[i]]['validation'][0], axis=0)

    np.random.shuffle(evaluation_set)
    evaluation_set = evaluation_set[0:DATASET_SIZE]
    '''

    for i in range(len(r_names)):
        for j in range(i + 1, len(r_names)):
            key1, key2 = r_names[i], r_names[j]
            print('{} / {}'.format(key1, key2))

            evaluation_set = np.append(
                roundabouts_data[key1]['validation'][0],
                roundabouts_data[key2]['validation'][0],
                axis=0)
            #print (evaluation_set.shape)

            classif1 = classify(
                evaluation_set,
                roundabouts_data[key1]['classifier'])
            classif2 = classify(
                evaluation_set,
                roundabouts_data[key2]['classifier'])

            pred_1_1 = roundabouts_data[key1]['classifier'].predict(
                roundabouts_data[key1]['validation'][0])
            pred_2_1 = roundabouts_data[key2]['classifier'].predict(
                roundabouts_data[key1]['validation'][0])
            pred_2_2 = roundabouts_data[key2]['classifier'].predict(
                roundabouts_data[key2]['validation'][0])
            pred_1_2 = roundabouts_data[key1]['classifier'].predict(
                roundabouts_data[key2]['validation'][0])

            acc_1_1 = accuracy_score(
                pred_1_1, roundabouts_data[key1]['validation'][1])
            acc_2_1 = accuracy_score(
                pred_2_1, roundabouts_data[key1]['validation'][1])
            acc_2_2 = accuracy_score(
                pred_2_2, roundabouts_data[key2]['validation'][1])
            acc_1_2 = accuracy_score(
                pred_1_2, roundabouts_data[key2]['validation'][1])

            recall_1_1 = recall_score(
                pred_1_1, roundabouts_data[key1]['validation'][1])
            recall_2_1 = recall_score(
                pred_2_1, roundabouts_data[key1]['validation'][1])
            recall_2_2 = recall_score(
                pred_2_2, roundabouts_data[key2]['validation'][1])
            recall_1_2 = recall_score(
                pred_1_2, roundabouts_data[key2]['validation'][1])

            precision_1_1 = precision_score(
                pred_1_1, roundabouts_data[key1]['validation'][1])
            precision_2_1 = precision_score(
                pred_2_1, roundabouts_data[key1]['validation'][1])
            precision_2_2 = precision_score(
                pred_2_2, roundabouts_data[key2]['validation'][1])
            precision_1_2 = precision_score(
                pred_1_2, roundabouts_data[key2]['validation'][1])

            f1_1_1 = f1_score(
                pred_1_1, roundabouts_data[key1]['validation'][1])
            f1_2_1 = f1_score(
                pred_2_1, roundabouts_data[key1]['validation'][1])
            f1_2_2 = f1_score(
                pred_2_2, roundabouts_data[key2]['validation'][1])
            f1_1_2 = f1_score(
                pred_1_2, roundabouts_data[key2]['validation'][1])

            f1_1_1_test = f1_score(
                roundabouts_data[key1]['validation'][1], pred_1_1)
            f1_2_1_test = f1_score(
                roundabouts_data[key1]['validation'][1], pred_2_1)
            f1_2_2_test = f1_score(
                roundabouts_data[key2]['validation'][1], pred_2_2)
            f1_1_2_test = f1_score(
                roundabouts_data[key2]['validation'][1], pred_1_2)

            print("{}/{}".format(f1_1_1, f1_1_1_test))
            print("{}/{}".format(f1_2_1, f1_2_1_test))
            print("{}/{}".format(f1_2_2, f1_2_2_test))
            print("{}/{}".format(f1_1_2, f1_1_2_test))

            mi = EDGE(classif1, classif2, gamma=[0.1, 0.1])
            entrop1 = entropy(classif1)
            entrop2 = entropy(classif2)

            #print (mi)
            mi_matrix.at[key1, key2] = mi / entrop1
            mi_matrix.at[key2, key1] = mi / entrop2

            MI_matrix.at[key1, key2] = mi
            MI_matrix.at[key2, key1] = mi

            mse = np.square(classif1 - classif2).mean()
            mse_matrix.at[key1, key2] = mse
            mse_matrix.at[key2, key1] = mse

            #print ("mse: {}".format(mse))

            acc_matrix.at[key1, key1] = acc_1_1
            acc_matrix.at[key2, key2] = acc_2_2
            acc_matrix.at[key1, key2] = acc_1_2
            acc_matrix.at[key2, key1] = acc_2_1

            recall_matrix.at[key1, key1] = recall_1_1
            recall_matrix.at[key2, key2] = recall_2_2
            recall_matrix.at[key1, key2] = recall_1_2
            recall_matrix.at[key2, key1] = recall_2_1

            precision_matrix.at[key1, key1] = precision_1_1
            precision_matrix.at[key2, key2] = precision_2_2
            precision_matrix.at[key1, key2] = precision_1_2
            precision_matrix.at[key2, key1] = precision_2_1

            f1_matrix.at[key1, key1] = f1_1_1
            f1_matrix.at[key2, key2] = f1_2_2
            f1_matrix.at[key1, key2] = f1_1_2
            f1_matrix.at[key2, key1] = f1_2_1

    MI_matrix.to_json(
        'mi_data/mi_MI_{}/{}.json'.format(TRAINING_SIZE, randseed))
    mi_matrix.to_json('mi_data/mi_{}/{}.json'.format(TRAINING_SIZE, randseed))
    mse_matrix.to_json(
        'mi_data/mi_mse_{}/{}.json'.format(TRAINING_SIZE, randseed))
    acc_matrix.to_json(
        'mi_data/mi_acc_{}/{}.json'.format(TRAINING_SIZE, randseed))
    recall_matrix.to_json(
        'mi_data/mi_recall_{}/{}.json'.format(TRAINING_SIZE, randseed))
    precision_matrix.to_json(
        'mi_data/mi_precision_{}/{}.json'.format(TRAINING_SIZE, randseed))
    f1_matrix.to_json(
        'mi_data/mi_f1_{}/{}.json'.format(TRAINING_SIZE, randseed))
