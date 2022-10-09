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
import argparse
sys.path.append('..')
from tools.flow_measure import EntryFlow
from tools import locations
from tools.parse_utils import read_frames, get_input_paths_for_location
import tools.read_csv_round as rd


def parse_flows(location_id):

    input_args = get_input_paths_for_location(location_id)
    topology = locations.get_topology_for_location(location_id)

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
        frames = read_frames(tracks, tracks_meta, last_frame=last_frame, last_track=last_track)

        # Initialize entry flow tracking
        flow = EntryFlow(topology, framerate=25.0)
        flow.load_flow(frames)
        flow.save('flow_parse/round_flow_{}.json'.format(args['id']))


if __name__ == '__main__':

    # Create the target directory if needed
    try:
        os.mkdir('flow_parse')
    except FileExistsError:
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--location",
        help="The location ID to generate TTC data from.",
        type=int)
    argsparse = parser.parse_args()
    parse_flows(argsparse.location)
