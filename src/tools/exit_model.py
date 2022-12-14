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

import json
import numpy as np
from sklearn.linear_model import LogisticRegression

def gather_training_data(
        input_ids,
        seed,
        interaction=True,
        basepath='',
        filterout=True):
    '''Extracts training data from the pickle files generated by parse.py'''
    training_data = []
    for id_str in input_ids:
        path = basepath + 'exit_parse/round_exit_{}.json'.format(id_str)
        if interaction:
            path = basepath + 'exit_parse/inter_exit_{}.json'.format(id_str)

        with open(path, 'r') as f:
            exit_data = json.load(f)

            for vehicle_id in exit_data.keys():
                filtered_data = []
                for item in exit_data[vehicle_id]:
                    if filterout:
                        filtered_data.append(
                            [item[0], item[1], item[2], item[-1]])
                    else:
                        filtered_data.append(item)

                training_data.extend(filtered_data)

    if seed != -1:
        np.random.seed(seed)
    np.random.shuffle(training_data)

    return training_data



def process_training_data(
        training_data,
        training_size=None,
        evaluation_size=None):
    '''Extracts training and validation sets compatible with sklearn from training_data'''
    training_set = None
    validation_set = None

    if training_size is None or evaluation_size is None:
        nb_samples = len(training_data)
        nb_samples_training = int(nb_samples * 0.8)

        training_set = training_data[0:nb_samples_training]
        validation_set = training_data[nb_samples_training:]
    else:
        training_set = training_data[0:training_size]
        validation_set = training_data[training_size:training_size +
                                       evaluation_size]

    (x_training, y_training) = ([], [])
    (x_validation, y_validation) = ([], [])

    for (laneid, heading, dist, label) in training_set:
        x_training.append([laneid, heading, dist])
        y_training.append(label)

    x_training = np.array(x_training)
    y_training = np.array(y_training)

    for (laneid, heading, dist, label) in validation_set:
        x_validation.append([laneid, heading, dist])
        y_validation.append(label)

    x_validation = np.array(x_validation)
    y_validation = np.array(y_validation)

    return (x_training, y_training, x_validation, y_validation)


def get_exit_proba_model(input_ids, seed, interaction=True):
    '''Trains a model to predict the probability of vehicles exiting from the roundabout based on the data of the given input file'''
    training_data = gather_training_data(input_ids, seed, interaction)
    (x_training, y_training, x_validation,
     y_validation) = process_training_data(training_data)

    model = LogisticRegression()
    model.fit(x_training, y_training)

    return (model, model.score(x_validation, y_validation))



def get_exit_probability(model, lane, heading, distance):
    '''Returns the probability of exit of a given vehicle using a trained model'''
    sample = np.array([[lane, heading, distance]])
    true_class_id = [i for i in range(
        len(model.classes_)) if model.classes_[i]][0]

    return model.predict_proba(sample)[0][true_class_id]
