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

import tools.read_csv as rint
from tools import consts

def get_frames(tracks, track_id):
    '''Gets the frames associated with an INTERACTION track id'''
    return (tracks[track_id][rint.INITIAL_FRAME],
            tracks[track_id][rint.LAST_FRAME])


def read_frames(tracks, last_frame=-1, last_track=-1):
    '''Read the track data of a subset of INTERACTION frames'''
    res = {}
    for frame_id in range(last_frame + 1):
        res[frame_id] = []

        for track_id in range(last_track + 1):
            frames = get_frames(tracks, track_id)
            # The object exists in that frame
            if frame_id >= frames[0] and frame_id <= frames[1]:
                ix = list(tracks[track_id][consts.FRAME]).index(frame_id)
                pos = tracks[track_id][consts.BBOX][ix]
                res[frame_id].append({consts.TRACK_ID: track_id,
                                     consts.X: pos[0], consts.Y: pos[1],
                                     consts.WIDTH: pos[2], consts.HEIGHT: pos[3],
                                     consts.HEADING: tracks[track_id][consts.HEADING][ix],
                                     consts.X_VELOCITY: tracks[track_id][consts.X_VELOCITY][ix],
                                     consts.Y_VELOCITY: tracks[track_id][consts.Y_VELOCITY][ix]})
    return res
