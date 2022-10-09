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

from os import listdir
from os.path import isfile, join

import tools.read_csv_round
import tools.topology as topology
import tools.consts as consts


def get_input_interaction(location_str):
    base_path = consts.INTER_PATH + "/" + location_str
    data_files = [f for f in listdir(base_path) if isfile(join(base_path, f))]

    res = []
    for f in data_files:
        if f.startswith('vehicle'):
            res.append(consts.INTER_PATH + "/" + location_str + '/' + f)

    return res


def get_topology_interaction(location_str):
    if location_str == "DR_USA_Roundabout_FT":
        return topology.Topology.interaction_USA_FT_Topology()
    elif location_str == "DR_USA_Roundabout_SR":
        return topology.Topology.interaction_USA_SR_Topology()
    elif location_str == "DR_USA_Roundabout_EP":
        return topology.Topology.interaction_USA_EP_Topology()
    elif location_str == "DR_CHN_Roundabout_LN":
        return topology.Topology.interaction_CHN_LN_Topology()
    elif location_str == "DR_DEU_Roundabout_OF":
        return topology.Topology.interaction_DEU_OF_Topology()
    else:
        raise Exception(
            "The topology for location {} has not been defined.".format(location_str))


def get_input_for_location(location):
    input_ids = []

    data_files = [
        f for f in listdir(
            consts.ROUND_PATH) if isfile(
            join(
                consts.ROUND_PATH,
                f))]

    for filename in data_files:
        if filename.endswith('recordingMeta.csv'):
            meta_info = tools.read_csv_round.read_meta_info(
                {'input_meta_path': join(consts.ROUND_PATH, filename)})

            if int(meta_info[tools.read_csv_round.LOCATION_ID]) == location:
                split = filename.split('_')
                if len(split) == 0:
                    print(
                        "Warning: filename {} is not separated by '_'".format(filename))
                else:
                    input_ids.append(split[0])

    return input_ids


def get_topology_for_location(location):
    if location == 0:
        return topology.Topology.roundDLocation0Topology()
    elif location == 1:
        return topology.Topology.roundDLocation1Topology()
    elif location == 2:
        return topology.Topology.roundDLocation2Topology()
    else:
        raise Exception(
            "The topology for location {} has not been defined.".format(location))
