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
import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
sys.path.append('..')
from tools import locations
import tools.read_csv_round as rd
from tools.parse_utils import read_frames, get_input_paths_for_location
from tools.flow_measure import EntryFlow
from tools.exit_tracking import ExitTracking


# .......................................................................... #
# Browse all recordings associated with a given location to compute TTC data #
# The obtained ttc data is saved into a pickle file to be reused by          #
# analysis.py .............................................................. #
# .......................................................................... #

def parse_exits(location_id):
    input_args = get_input_paths_for_location(location_id)
    topology = locations.get_topology_for_location(location_id)

    flow_period = 30.0
    framerate = 25.0

    for args in input_args:
        tracks = rd.read_track_csv(args)
        tracks_meta = rd.read_static_info(args)
        recordings_meta = rd.read_meta_info(args)

        last_frame = int(recordings_meta[rd.FRAME_RATE]
                         * recordings_meta[rd.DURATION]) - 1
        last_track = int(recordings_meta[rd.NUM_TRACKS]) - 1

        print("Read tracks: {}".format(args['input_path']))
        print("Nb tracks:{}, Last Frame: {}".format(last_track, last_frame))

        # Loading flow stats
        entry_flow = EntryFlow(topology, framerate)
        entry_flow.read_json('flow_parse/round_flow_{}.json'.format(args['id']))
        flow_stats = entry_flow.analyze_complete(flow_period)

        print("Loading frames...")
        frames = read_frames(tracks, tracks_meta, last_frame=last_frame, last_track=last_track)

        # Initialize exit tracking
        exit_data = ExitTracking(topology, flow_stats)

        for frame_id in range(len(frames)):
            print("frame {}/{}".format(frame_id, last_frame), end='\r')
            for obj in frames[frame_id]:
                exit_data.update_vehicle(obj, frame_id / framerate)

        result = exit_data.training_data
        with open('exit_parse/round_exit_{}.json'.format(args['id']), 'w') as handle:
            json.dump(result, handle)

if __name__ == '__main__':

    # Create the target directory if needed
    try:
        os.mkdir('exit_parse')
    except FileExistsError:
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--location",
        help="The location ID to generate TTC data from.",
        type=int)
    argsparse = parser.parse_args()
    parse_exits(argsparse.location)
