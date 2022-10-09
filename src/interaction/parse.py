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
import json
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
sys.path.append('..')
from tools import locations
import tools.exit_model as exit_model_lib
from tools.ttc_correlation import TTCTimeline
from tools import consts
from tools.consts import FRAMERATE_INTERACTION
import tools.read_csv as rint
from tools.parse_utils_interaction import read_frames

# .......................................................................... #
# Browse all recordings associated with a given location to compute TTC data #
# The obtained ttc data is saved into a json result file to be reused by     #
# analysis.py .............................................................. #
# .......................................................................... #

if __name__ == '__main__':

    np.random.seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dst",
        help="Directory where the generated json files will be saved.",
        default='ttc_parse')
    parser.add_argument(
        "--location",
        help="The location ID to generate TTC data from.",
        type=str)
    parser.add_argument(
        "--begin",
        help="Beginning recording ix to generate.",
        type=int,
        default=-1)
    parser.add_argument(
        "--end",
        help="Last recording ix to generate.",
        type=int,
        default=-1)
    parser.add_argument(
        "--probability_weighting",
        help="Compute risk using vehicles' roundabout exiting probability.",
        action="store_true")
    argsparse = parser.parse_args()

    # Create the target directory if needed
    try:
        os.mkdir(argsparse.dst)
    except FileExistsError:
        pass

    print("Loading recordings for location {}".format(argsparse.location))

    input_str_all = locations.get_input_interaction(argsparse.location)
    input_str = input_str_all
    if argsparse.begin != -1 and argsparse.end != - \
            1 and argsparse.begin >= 0 and argsparse.end > argsparse.begin and argsparse.end <= len(input_str):
        input_str = input_str_all[argsparse.begin:argsparse.end]


    print("Input files for location {}: {}".format(argsparse.location, input_str))


    # Loading topology for the location.
    topology = locations.get_topology_interaction(argsparse.location)

    # .................................................................................... #
    # If requested, compute exiting probability estimation models to weight TTC risk with. #
    # .................................................................................... #


    exit_proba_models = {}
    if argsparse.probability_weighting:
        print("Computing exit probability models...")

        for input_str_val in input_str:

            model_training_inputs = []
            for value in input_str:
                if value != input_str_val:
                    model_training_inputs.append(
                        argsparse.location + '_' + os.path.basename(value))

            (exit_model, accuracy) = exit_model_lib.get_exit_proba_model(
                model_training_inputs, 101010)
            print("Model trained excluding {}_tracks, accuracy: {}".format(
                os.path.basename(input_str_val), accuracy))

            exit_proba_models[input_str_val] = exit_model

    # Positioning error simulation
    '''position_noises = [0, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3,
                          0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                          1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                          2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]'''
    position_noises = [0]

    for args in input_str:
        tracks, last_frame, last_track = rint.read_track_csv(args)

        print("Read tracks: {}".format(os.path.basename(args)))
        print("Nb tracks:{}, Last Frame: {}".format(last_track, last_frame))

        print("Loading frames...")
        frames = read_frames(tracks, last_frame=last_frame, last_track=last_track)

        # A. TTC VALUES COMPUTATION
        risk_modes = [1, 2, 3, 4, 5, 6]

        timelines = []
        for noise in position_noises:
            timelines.append(
                (noise,
                 TTCTimeline(
                     risk_modes,
                     FRAMERATE_INTERACTION)))

        for frameId in range(len(frames)):
            print("frame {}/{}".format(frameId, last_frame), end='\r')
            for obj in frames[frameId]:

                # Fetch TTC data
                ttc = topology.getTTC(frames[frameId], obj, position_noises)

                # Compute risk likelihood if requested
                risk_probability = None
                if ttc is not None and args in exit_proba_models:
                    risk_probability = topology.get_risk_probability(
                        obj, ttc[0], exit_proba_models[args])

                for (i, (noise, timeline)) in enumerate(timelines):

                    # Notify the presence of the object inside the roundabout
                    if topology.is_inside_roundabout(obj):
                        timeline.add_occupancy(frameId, obj[consts.TRACK_ID])

                    # Add TTC data
                    if ttc is not None and ttc[1][i][1] > 0:

                        if ttc[1][i][0] != noise:
                            raise ValueError(
                                'TTC noise / timeline noise value mismatch')

                        # Add TTC data to the timeline
                        timeline.add(frameId,
                                     obj[consts.TRACK_ID],
                                     ttc[0][consts.TRACK_ID],
                                     ttc[1][i][1],
                                     risk_probability)

        tl_dicts = []
        for tl in timelines:
            tl_dicts.append((tl[0], tl[1].export_dict()))

        with open('{}/inter_ttc_{}_{}.json'.format(argsparse.dst, argsparse.location, os.path.basename(args)), 'w') as handle:
            json.dump(tl_dicts, handle)
