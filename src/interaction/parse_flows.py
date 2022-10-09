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
import argparse
import sys
sys.path.append('..')
from tools.flow_measure import EntryFlow
import tools.locations as locations
import tools.read_csv as rint
from tools.parse_utils_interaction import read_frames


def parse_flows(location_id):
    print("Loading recordings for location {}".format(location_id))
    input_args = locations.get_input_interaction(location_id)
    topology = locations.get_topology_interaction(location_id)

    for args in input_args:
        tracks, last_frame, last_track = rint.read_track_csv(args)

        print("Read tracks: {}".format(args))
        print("Nb tracks:{}, Last Frame: {}".format(last_track, last_frame))

        print("Loading frames...")
        frames = read_frames(tracks, last_frame=last_frame, last_track=last_track)

        # Initialize entry flow tracking
        flow = EntryFlow(topology, framerate=10.0)
        flow.load_flow(frames)
        flow.save('flow_parse/inter_flow_{}_{}.json'.format(location_id,
                  os.path.basename(args)))

if __name__ == '__main__':
    # Create the target directory if needed
    try:
        os.mkdir('flow_parse')
    except FileExistsError:
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--location",
        help="The interaction roundabout name to generate TTC data from.",
        type=str)
    argsparse = parser.parse_args()

    parse_flows(argsparse.location)
