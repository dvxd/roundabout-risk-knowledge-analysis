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
import argparse
import json
import os
import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
sys.path.append('..')
import tools.read_csv as rint
from tools.flow_measure import EntryFlow
from tools.exit_tracking import ExitTracking
from tools import locations
from tools import consts
from tools.parse_utils_interaction import read_frames


# .......................................................................... #
# Browse all recordings associated with a given location to compute TTC data #
# The obtained ttc data is saved into a json file to be reused by          #
# analysis.py .............................................................. #
# .......................................................................... #

def parse_exits(location_id):
    print("Loading recordings for location {}".format(location_id))
    input_args = locations.get_input_interaction(location_id)
    topology = locations.get_topology_interaction(location_id)

    flow_period = 30.0
    framerate = consts.FRAMERATE_INTERACTION

    for args in input_args:
        tracks, last_frame, last_track = rint.read_track_csv(args)

        print("Read tracks: {}".format(args))
        print("Nb tracks:{}, Last Frame: {}".format(last_track, last_frame))

        # Loading flow stats
        entry_flow = EntryFlow(topology, framerate)
        entry_flow.read_json(
            'flow_parse/inter_flow_{}_{}.json'.format(location_id, os.path.basename(args)))
        flow_stats = entry_flow.analyze_complete(flow_period)

        print("Loading frames...")
        frames = read_frames(tracks, last_frame=last_frame, last_track=last_track)

        # Initialize exit tracking
        exit_data = ExitTracking(topology, flow_stats)

        for frameId in range(len(frames)):
            print("frame {}/{}".format(frameId, last_frame), end='\r')
            for obj in frames[frameId]:
                exit_data.update_vehicle(obj, frameId / framerate)

        result = exit_data.training_data
        with open('exit_parse/inter_exit_{}_{}.json'.format(location_id, os.path.basename(args)), 'w') as handle:
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
        help="The interaction roundabout name to generate TTC data from.",
        type=str)
    argsparse = parser.parse_args()

    parse_exits(argsparse.location)
