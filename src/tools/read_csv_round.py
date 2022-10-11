#!/usr/bin/env python3
'''
Adapted from the HighD dataset tools: https://github.com/RobertKrajewski/highD-dataset/blob/master/Python/src/data_management/read_csv.py

Copyright (c) 2017-2020, Institute for Automotive Engineering of RWTH Aachen University and fka GmbH
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
3. All advertising materials mentioning features or use of this software must display the following acknowledgement: “This product includes software developed by IKA RWTH Aachen and fka GmbH and its contributors.”
4. Neither the name of the Institute for Automotive Engineering of RWTH Aachen University and fka GmbH nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS “AS IS” AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''

import pandas
import numpy as np
import tools.consts as consts

# TRACK FILE
BBOX = "bbox"
FRAME = "frame"
TRACK_ID = "trackId"
X = "xCenter"
Y = "yCenter"
WIDTH = "width"
HEIGHT = "length"
HEADING = "heading"
X_VELOCITY = "xVelocity"
Y_VELOCITY = "yVelocity"
X_ACCELERATION = "xAcceleration"
Y_ACCELERATION = "yAcceleration"
LON_VELOCITY = "lonVelocity"
LAT_VELOCITY = "latVelocity"
LON_ACCELERATION = "lonAcceleration"
LAT_ACCELERATION = "latAcceleration"


# STATIC FILE
INITIAL_FRAME = "initialFrame"
FINAL_FRAME = "finalFrame"
NUM_FRAMES = "numFrames"
CLASS = "class"

# VIDEO META
RECORDING_ID = "recordingId"
FRAME_RATE = "frameRate"
LOCATION_ID = "locationId"
SPEED_LIMIT = "speedLimit"
WEEKDAY = "weekday"
START_TIME = "startTime"
DURATION = "duration"
NUM_TRACKS = "numTracks"
# ...


def read_track_csv(arguments):
    """
    This method reads the tracks file from highD data.

    :param arguments: the parsed arguments for the program containing the input path for the tracks csv file.
    :return: a list containing all tracks as dictionaries.
    """
    # Read the csv file, convert it into a useful data structure
    df = pandas.read_csv(arguments["input_path"])

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
        tracks[current_track] = {consts.TRACK_ID: np.int64(group_id),  # for compatibility, int would be more space efficient
                                 consts.FRAME: rows[FRAME].values,
                                 consts.BBOX: bounding_boxes,
                                 consts.HEADING: rows[HEADING].values,
                                 consts.X_VELOCITY: rows[X_VELOCITY].values,
                                 consts.Y_VELOCITY: rows[Y_VELOCITY].values,
                                 X_ACCELERATION: rows[X_ACCELERATION].values,
                                 Y_ACCELERATION: rows[Y_ACCELERATION].values,
                                 LON_VELOCITY: rows[LON_VELOCITY].values,
                                 LAT_VELOCITY: rows[LAT_VELOCITY].values,
                                 LON_ACCELERATION: rows[LON_ACCELERATION].values,
                                 LAT_ACCELERATION: rows[LAT_ACCELERATION].values
                                 }
        current_track = current_track + 1
    return tracks


def read_static_info(arguments):
    """
    This method reads the static info file from highD data.

    :param arguments: the parsed arguments for the program containing the input path for the static csv file.
    :return: the static dictionary - the key is the track_id and the value is the corresponding data for this track
    """
    # Read the csv file, convert it into a useful data structure
    df = pandas.read_csv(arguments["input_static_path"])

    # Declare and initialize the static_dictionary
    static_dictionary = {}

    # Iterate over all rows of the csv because we need to create the bounding
    # boxes for each row
    for i_row in range(df.shape[0]):
        track_id = int(df[TRACK_ID][i_row])
        static_dictionary[track_id] = {
            TRACK_ID: track_id, WIDTH: int(
                df[WIDTH][i_row]), HEIGHT: int(
                df[HEIGHT][i_row]), INITIAL_FRAME: int(
                df[INITIAL_FRAME][i_row]), FINAL_FRAME: int(
                    df[FINAL_FRAME][i_row]), NUM_FRAMES: int(
                        df[NUM_FRAMES][i_row]), CLASS: str(
                            df[CLASS][i_row])}
    return static_dictionary


def read_meta_info(arguments):
    """
    This method reads the video meta file from highD data.

    :param arguments: the parsed arguments for the program containing the input path for the video meta csv file.
    :return: the meta dictionary containing the general information of the video
    """
    # Read the csv file, convert it into a useful data structure
    df = pandas.read_csv(arguments["input_meta_path"])

    # Declare and initialize the extracted_meta_dictionary
    extracted_meta_dictionary = {RECORDING_ID: int(df[RECORDING_ID][0]),
                                 FRAME_RATE: int(df[FRAME_RATE][0]),
                                 LOCATION_ID: int(df[LOCATION_ID][0]),
                                 SPEED_LIMIT: float(df[SPEED_LIMIT][0]),
                                 WEEKDAY: str(df[WEEKDAY][0]),
                                 START_TIME: str(df[START_TIME][0]),
                                 DURATION: float(df[DURATION][0]),
                                 NUM_TRACKS: float(df[NUM_TRACKS][0])
                                 }
    return extracted_meta_dictionary
