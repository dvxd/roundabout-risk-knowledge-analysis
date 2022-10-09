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
from sklearn.linear_model import LogisticRegression
from EDGE.EDGE_4_4_1 import EDGE
sys.path.append('..')
import tools.exit_model as model
from mi import entropy, classify, get_roundabouts_inputs


def prepare_data(roundabouts, key):

    # Extract validation and training sets.
    (x_training, y_training, x_validation,
     y_validation) = model.process_training_data(roundabouts[key]['data'])
    print('{}: len training: {}, valid: {}'.format(
        key, len(y_training), len(y_validation)))

    if len(y_validation) < 100:
        del roundabouts[key]
        return

    roundabouts[key]['validation'] = (
        x_validation[0:DATASET_SIZE // 2], y_validation[0:DATASET_SIZE // 2])

    # Perform regression
    regression = LogisticRegression()
    regression.fit(x_training, y_training)

    #print (regression.classes_)
    #true_class = [i for i in range(
    #    len(regression.classes_)) if regression.classes_[i]][0]
    #print (true_class)

    accuracy = regression.score(x_validation, y_validation)
    print("{} accuracy: {}".format(key, accuracy))

    roundabouts[key]['classifier'] = regression


if __name__ == '__main__':
    DATASET_SIZE = 10000
    # 1. Training classifiers.
    randseed = int(time.time())
    print('\nUsing seed {}...'.format(randseed))

    roundabouts = get_roundabouts_inputs()
    roundabouts_data = {}

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

        traffic_ranges = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3),
                          (0.3, 0.4), (0.4, 0.5), (0.5, 0.6)]
        for (tmin, tmax) in traffic_ranges:
            df_traffic = df[(df['FOC_German'] >= tmin) & (df['FOC_German'] < tmax)]
            if df_traffic.shape[0] >= 5000:  # tolerate 5000 entries
                df_traffic = df_traffic[training_cols]
                roundabouts_data['{}-{}_{}'.format(key, tmin, tmax)] = {
                    'data': df_traffic.to_numpy()}
                prepare_data(roundabouts_data, '{}-{}_{}'.format(key, tmin, tmax))


    r_names = list(roundabouts_data.keys())
    mi_matrix = pd.DataFrame(columns=r_names, index=r_names)

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
            print(evaluation_set.shape)

            classif1 = classify(
                evaluation_set,
                roundabouts_data[key1]['classifier'])
            classif2 = classify(
                evaluation_set,
                roundabouts_data[key2]['classifier'])

            mi = EDGE(classif1, classif2, gamma=[0.1, 0.1])
            entrop1 = entropy(classif1)
            entrop2 = entropy(classif2)

            #print (mi)
            mi_matrix.at[key1, key2] = mi / entrop1
            mi_matrix.at[key2, key1] = mi / entrop2


    mi_matrix.to_json('mi_traffic/{}.json'.format(randseed))
    print(mi_matrix)
