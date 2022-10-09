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
sys.path.append('..')
import tools.locations as locations
import tools.exit_model as exit_model_lib
from tools.ttc_correlation import TTCTimeline
from tools.parse_utils import read_frames, get_input_paths_for_recordings
from tools import consts
import tools.read_csv_round as rd


def compute_exit_probability_models(input_ids, input_ids_all):
    '''compute exiting probability estimation models to weight TTC risk with'''
    exit_proba_models = {}
    print("Computing exit probability models...")

    for input_str in input_ids:
        model_training_inputs = input_ids_all.copy()
        # Do not train the model using the data it will be applied to.
        model_training_inputs.remove(input_str)

        (exit_model, accuracy) = exit_model_lib.get_exit_proba_model(
            model_training_inputs, 101010, interaction=False)
        print(
            "Model trained excluding {}_tracks, accuracy: {}".format(
                input_str, accuracy))
        print('{} -> {}'.format(input_str, model_training_inputs))

        exit_proba_models[input_str] = exit_model

    return exit_proba_models


def parse(input_args, topology, exit_proba_models, dst_path):
    # Positioning error simulation
    position_noises = [0, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3,
                       0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                       1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                       2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]

    for args in input_args:
        tracks = rd.read_track_csv(args)
        tracks_meta = rd.read_static_info(args)
        recordings_meta = rd.read_meta_info(args)

        last_frame = int(recordings_meta[rd.FRAME_RATE]
                         * recordings_meta[rd.DURATION]) - 1
        last_track = int(recordings_meta[rd.NUM_TRACKS]) - 1

        print("Read tracks: {}".format(args['input_path']))
        print("Nb tracks:{}, Last Frame: {}".format(last_track, last_frame))

        print("Loading frames...")
        frames = read_frames(
            tracks,
            tracks_meta,
            last_frame=last_frame,
            last_track=last_track)

        # A. TTC VALUES COMPUTATION
        risk_modes = [1, 2, 3, 4, 5, 6]

        timelines = []
        for noise in position_noises:
            timelines.append(
                (noise, TTCTimeline(risk_modes, recordings_meta[rd.FRAME_RATE])))

        for frame_id in range(len(frames)):
            print("frame {}/{}".format(frame_id, last_frame), end='\r')
            for obj in frames[frame_id]:

                # Fetch TTC data
                ttc = topology.getTTC(frames[frame_id], obj, position_noises)

                # Compute risk likelihood if requested
                risk_probability = None
                if ttc is not None and args['id'] in exit_proba_models:
                    risk_probability = topology.get_risk_probability(
                        obj, ttc[0], exit_proba_models[args['id']])

                for (i, (noise, timeline)) in enumerate(timelines):

                    # Notify the presence of the object inside the roundabout
                    if topology.is_inside_roundabout(obj):
                        timeline.add_occupancy(frame_id, obj[consts.TRACK_ID])

                    # Add TTC data
                    if ttc is not None and ttc[1][i][1] > 0:

                        if (ttc[1][i][0] != noise):
                            raise ValueError(
                                'TTC noise / timeline noise value mismatch')

                        # Add TTC data to the timeline
                        timeline.add(frame_id,
                                     obj[consts.TRACK_ID],
                                     ttc[0][consts.TRACK_ID],
                                     ttc[1][i][1],
                                     risk_probability)

        tl_dicts = []
        for tl in timelines:
            tl_dicts.append((tl[0], tl[1].export_dict()))

        with open('{}/round_ttc_{}.json'.format(dst_path, args['id']), 'w') as handle:
            json.dump(tl_dicts, handle)


# .......................................................................... #
# Browse all recordings associated with a given location to compute TTC data #
# The obtained ttc data is saved into a json file to be reused by            #
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
        type=int)
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

    input_ids_all = locations.get_input_for_location(argsparse.location)
    input_ids = input_ids_all
    if argsparse.begin != -1 and argsparse.end != - \
            1 and argsparse.begin >= 0 and argsparse.end > argsparse.begin and argsparse.end <= len(input_ids):
        input_ids = input_ids_all[argsparse.begin:argsparse.end]

    print("Input files for location {}: {}".format(argsparse.location, input_ids))

    input_args = get_input_paths_for_recordings(input_ids)
    topology = locations.get_topology_for_location(argsparse.location)

    exit_proba_models = {}
    if argsparse.probability_weighting:
        compute_exit_probability_models(input_ids, input_ids_all)

    parse(input_args, topology, exit_proba_models, argsparse.dst)
