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

'''
Read csv files from the INTERACTION dataset.
'''

import pandas
import numpy as np
import tools.consts as consts

# TRACK FILE
BBOX = "bbox"
FRAME = "frame_id"
TRACK_ID = "track_id"
X = "x"
Y = "y"
WIDTH = "width"
HEIGHT = "length"
HEADING = "psi_rad"
X_VELOCITY = "vx"
Y_VELOCITY = "vy"

INITIAL_FRAME = "frame_init"
LAST_FRAME = "frame_last"


def minus_one(value):
    return value - 1


def heading_process(heading):
    return np.rad2deg(heading)


def read_track_csv(input_path):
    '''Read INTERACTION track recordings'''
    # Read the csv file, convert it into a useful data structure
    df = pandas.read_csv(input_path)

    # Use groupby to aggregate track info. Less error prone than iterating
    # over the data.
    grouped = df.groupby(TRACK_ID, sort=False)
    # Efficiently pre-allocate an empty list of sufficient size
    tracks = [None] * grouped.ngroups
    current_track = 0
    for group_id, rows in grouped:
        bounding_boxes = np.transpose(np.array([rows[X].values,
                                                rows[Y].values,
                                                rows[WIDTH].values,
                                                rows[HEIGHT].values]))
        tracks[current_track] = {consts.TRACK_ID: current_track,  # for compatibility, int would be more space efficient
                                 consts.FRAME: list(map(minus_one, rows[FRAME].values)),
                                 consts.BBOX: bounding_boxes,
                                 consts.HEADING: list(map(heading_process, rows[HEADING].values)),
                                 consts.X_VELOCITY: rows[X_VELOCITY].values,
                                 consts.Y_VELOCITY: rows[Y_VELOCITY].values,
                                 INITIAL_FRAME: rows[FRAME].min() - 1,
                                 LAST_FRAME: rows[FRAME].max() - 1
                                 }
        current_track = current_track + 1

    return tracks, df[FRAME].max() - 1, current_track - 1
