import os
import argparse
import json
from pathlib import Path
import tqdm
import glob
import multiprocessing
import multiprocessing.dummy as mp
import numpy as np
import librosa
import pyroomacoustics as pra
import soundfile as sf

import sep.helpers.utils as utils
from sep.helpers.constants import SPEED_OF_SOUND


# Speaker signal peak value
FG_VOL_MIN = 0.2
FG_VOL_MAX = 0.5

# Speaker height (relative to desk)
MAX_SPEAKER_HEIGHT = 0.7
MIN_SPEAKER_HEIGHT = 0.1

# Minimum distance between speakers
MIN_SPEAKER_DIST = 0.51

# Height of robots
MIC_HEIGHT = 0.02

# Speaker orientation
SPEAKER_AZIMUTHAL_ANGLE_RANGE = 0 #np.pi * 0.8
SPEAKER_COALTITUDE_ANGLE_RANGE = 0 #np.pi / 4

# Reverb params
MIN_ABSORPTION = 0.1
MAX_ABSORPTION = 0.99

# Room dimensions
ROOM_LENGTH_MIN, ROOM_LENGTH_MAX = 6, 8
ROOM_WIDTH_MIN, ROOM_WIDTH_MAX = 6, 8
CEIL_MIN, CEIL_MAX = 2, 2.5

# Desk dimensions
DESK_LENGTH_MIN = 1.2
DESK_LENGTH_MAX = 2
DESK_WIDTH_MIN = 0.6
DESK_WIDTH_MAX = 1.2


# Speaker range to random sample the positions
WALL_KEEPOUT = 0.5
SPK_RANGE_W = 3
SPK_RANGE_H = 4.5


# Robot expansion perturbations
EXPAND_MAX_DEV = 0.08
THETA_MAX_DEV = np.deg2rad(6)

# Diameter of Amazon Echo Dot (https://www.dimensions.com/element/amazon-echo-dot-3rd-gen)
ECHO_DOT_DIAMETER = 0.1



def handle_error(e):
    print("Error happen " + "!"*30)
    print(e)

def get_voices(voices_list, n_voices, args):
    # Compute number of samples required
    total_samples = int(round(args.duration * args.sr))
    
    # Padding before and after speaker activity
    activity_pad = int(round(args.sr * 0.2))
    
    # Choose speakers
    voice_dirs = np.random.choice(voices_list, n_voices, replace=False)
    
    voices_data = []
    for voice_dir in voice_dirs:
        # Speaker must be relatively active for the duration
        # (1) Speaker activity least 0.5 seconds (after trimming silence)
        # Load audio for speaker until a satisfactory one is found
        success = False
        while not success:
            # Choose random speech sample for this speaker
            files = glob.glob(os.path.join(voice_dir, '*.wav'))
            voice_file = np.random.choice(files)
            
            # Load audio
            voice, _ = librosa.core.load(voice_file, sr=args.sr, mono=True)

            # Remove silence at beginning and end
            voice_trimmed, (begin, end) = librosa.effects.trim(voice, top_db=18)
            
            # Condition (1)
            if voice_trimmed.std() > 2e-4 and (end - begin) > int(round(args.sr * 0.5)):
                success = True
            else:
                print("Chosen voice sample does not meet the desired conditions, retrying ...")
        
        # Clip voice with padding
        begin_idx = max([begin - activity_pad, 0])
        end_idx = min([end + activity_pad, voice.shape[-1]])
        voice = voice[begin_idx:end_idx]

        if voice.shape[-1] < total_samples:
            # voice = np.pad(voice, (0, total_samples - voice.shape[-1]))
            #if np.random.random() > 0.5:
            voice = np.pad(voice, (0, total_samples - voice.shape[-1]))
            #else:
            #    voice = np.pad(voice, (total_samples - voice.shape[-1], 0))

        elif voice.shape[-1] > total_samples:
            min_bound = voice.shape[-1] - total_samples
            begin_i = np.random.choice(min_bound)
            voice = voice[begin_i:begin_i+total_samples]

        # Get speaker identity
        voice_identity = os.path.basename(voice_dir.strip('/'))
        
        # Store speaker audio and identity
        voices_data.append((voice, voice_identity))
    
    return voices_data

def point_in_box(pos, left, right, top, bottom):
    # Check if a pos is inside the bounding box left, right, top, bottom
    return pos[0] >= left and pos[0] <= right and pos[1] <= top and pos[1] >= bottom

def is_valid_mic_array(array, left, right, bottom, top):
    """
    Checks whether the current mic array is valid (i.e., physically realizable)
    Currently performs the following checks:
    1- Microphones are at least 6cm with the wall
    """
    valid = True

    theshold = 0.06
    for i in range(array.shape[0]):
        if (array[i, 0] <= left+theshold or array[i, 0] >= right-theshold\
            or array[i, 1] <= bottom+theshold or array[i, 1] >= top - theshold ):
            valid = False
            return
    return valid
    


def get_random_mic_positions_three_desks(n_mics, left, right, bottom, top, args):
    # Minimum distance to the wall (length side) is 10cm
    DESK_WALL_MIN_DIST = 0.1
    min_x = left + DESK_WALL_MIN_DIST
    max_x = right - DESK_WALL_MIN_DIST
    min_y = bottom + DESK_WALL_MIN_DIST
    max_y = top - DESK_WALL_MIN_DIST
    

    ### size of the large table
    Desk_length = np.random.uniform(low=1.9, high=2)
    Desk_width = np.random.uniform(low=1.1, high=1.2)

    ### size of the middle table
    Desk_length_middle = np.random.uniform(low=1.4, high=1.5)
    Desk_width_middle = np.random.uniform(low=0.8, high=0.9)
    

    ### size of the small table
    Desk_length_small = np.random.uniform(low=1, high=1.1)
    Desk_width_small = np.random.uniform(low=0.5, high=0.6)

    ### robot expand in the desk
    angle_list = np.linspace(0, np.pi, n_mics - 1) - np.pi/2
    

    #### large desk
    middle_angle = np.arctan(Desk_length/2/Desk_width)
    mic_positions = np.zeros((n_mics, 2))
    for i in range(0, n_mics - 1):
        # Perturb robot move angle
        move_angle_err = np.random.uniform(low = -THETA_MAX_DEV, high = THETA_MAX_DEV)  #np.random.normal(0, THETA_STD)
        move_angle = angle_list[i] + move_angle_err

        if move_angle < middle_angle and move_angle > -middle_angle:
            expand_r = Desk_width/np.cos(move_angle)
        elif move_angle > middle_angle:
            expand_r = Desk_length/2/np.sin(move_angle)
        else:
            expand_r = Desk_length/2/np.sin(-move_angle)
        #print(i, move_angle/np.pi*180, expand_r)
        assert(expand_r >= 0)

        # Robot backoff
        expand_r = expand_r - 0.04
        # print("large expand_r=", expand_r)
        #print(np.rad2deg(move_angle))
        temp_pos_x = expand_r*np.cos(move_angle) + np.random.uniform(low=-EXPAND_MAX_DEV, high=EXPAND_MAX_DEV)
        temp_pos_y = expand_r*np.sin(move_angle) + np.random.uniform(low=-EXPAND_MAX_DEV, high=EXPAND_MAX_DEV)

        mic_positions[i+1] = np.array([temp_pos_x, temp_pos_y])


    # middle table
    mic_positions_middle = np.zeros((n_mics, 2))
    middle_angle = np.arctan(Desk_length_middle/2/Desk_width_middle)
    for i in range(0, n_mics - 1):
        move_angle_err = np.random.uniform(low = -THETA_MAX_DEV, high = THETA_MAX_DEV)  #np.random.normal(0, THETA_STD)
        #print(angle_list[i]/np.pi*180, move_angle_err/np.pi*180)
        move_angle = angle_list[i] + move_angle_err
        if move_angle < middle_angle and move_angle > -middle_angle:
            expand_r = Desk_width_middle/np.cos(move_angle)
        elif move_angle > middle_angle:
            expand_r = Desk_length_middle/2/np.sin(move_angle)
        else:
            expand_r = Desk_length_middle/2/np.sin(-move_angle)
        assert(expand_r >= 0)
        expand_r = expand_r - 0.04

        #print(np.rad2deg(move_angle))
        temp_pos_x = expand_r*np.cos(move_angle) + np.random.uniform(low=-EXPAND_MAX_DEV, high=EXPAND_MAX_DEV)
        temp_pos_y = expand_r*np.sin(move_angle) + np.random.uniform(low=-EXPAND_MAX_DEV, high=EXPAND_MAX_DEV)
        mic_positions_middle[i+1] = np.array([temp_pos_x, temp_pos_y])


    # small table
    mic_positions_small = np.zeros((n_mics, 2))
    middle_angle = np.arctan(Desk_length_small/2/Desk_width_small)
    for i in range(0, n_mics - 1):
        move_angle_err = np.random.uniform(low = -THETA_MAX_DEV, high = THETA_MAX_DEV)  #np.random.normal(0, THETA_STD)
        #print(angle_list[i]/np.pi*180, move_angle_err/np.pi*180)
        move_angle = angle_list[i] + move_angle_err
        if move_angle < middle_angle and move_angle > -middle_angle:
            expand_r = Desk_width_small/np.cos(move_angle)
        elif move_angle > middle_angle:
            expand_r = Desk_length_small/2/np.sin(move_angle)
        else:
            expand_r = Desk_length_small/2/np.sin(-move_angle)
        assert(expand_r >= 0)
        expand_r = expand_r - 0.04

        temp_pos_x = expand_r*np.cos(move_angle) + np.random.uniform(low=-EXPAND_MAX_DEV, high=EXPAND_MAX_DEV)
        temp_pos_y = expand_r*np.sin(move_angle) + np.random.uniform(low=-EXPAND_MAX_DEV, high=EXPAND_MAX_DEV)
        mic_positions_small[i+1] = np.array([temp_pos_x, temp_pos_y])


    ### place the desk, all three desks are placed in the same way
    # Maximum distance to the wall (length side) is 35cm
    DIS_WALL_DESK = 0.35

    # Minimum distance to the side walls is 180cm
    DIS_WALL_DESK2 = 1.8

    pickup_wall = np.random.choice(4)
    # Maximum desk rotation
    MAX_ROT = np.pi/8

    if pickup_wall == 0:
        X_range = [min_x, min_x + DIS_WALL_DESK]
        Y_range = [min_y + DIS_WALL_DESK2, max_y - DIS_WALL_DESK2 ]
        center_pos_x = np.random.uniform(low=X_range[0], high=X_range[1])
        center_pos_y = np.random.uniform(low=Y_range[0], high=Y_range[1])
        if (center_pos_x - min_x) >= (Desk_length/2):
            theta = np.random.uniform(low = -MAX_ROT, high = MAX_ROT)
        else:
            theta_bound = np.arcsin((center_pos_x - min_x)/(Desk_length/2))
            if theta_bound > MAX_ROT:
                theta = np.random.uniform(low = -MAX_ROT, high = MAX_ROT)
            else:
                theta = np.random.uniform(low = -theta_bound, high = theta_bound)


    elif pickup_wall == 1:
        Y_range = [min_y, min_y + DIS_WALL_DESK]
        X_range = [min_x + DIS_WALL_DESK2, max_x- DIS_WALL_DESK2 ]
        center_pos_x = np.random.uniform(low=X_range[0], high=X_range[1])
        center_pos_y = np.random.uniform(low=Y_range[0], high=Y_range[1])

        if (center_pos_y - min_y) >= (Desk_length/2):
            theta = np.random.uniform(low = -MAX_ROT+ np.pi/2, high = MAX_ROT+ np.pi/2)
        else:
            theta_bound = np.arcsin((center_pos_y - min_y)/(Desk_length/2))
            if theta_bound > MAX_ROT:
                theta = np.random.uniform(low = -MAX_ROT+ np.pi/2, high = MAX_ROT+ np.pi/2)
            else:
                theta = np.random.uniform(low = -theta_bound + np.pi/2, high = theta_bound + np.pi/2)

    elif pickup_wall == 2:
        X_range = [max_x - DIS_WALL_DESK, max_x ]
        Y_range = [min_y + DIS_WALL_DESK2, max_y - DIS_WALL_DESK2 ]
        center_pos_x = np.random.uniform(low=X_range[0], high=X_range[1])
        center_pos_y = np.random.uniform(low=Y_range[0], high=Y_range[1])
        if (max_x - center_pos_x) >= (Desk_length/2):
            theta = np.random.uniform(low = -MAX_ROT, high = MAX_ROT)
        else:
            theta_bound = np.arcsin((max_x - center_pos_x)/(Desk_length/2))
            if theta_bound > MAX_ROT:
                theta = np.random.uniform(low = -MAX_ROT+ np.pi, high = MAX_ROT+ np.pi)
            else:
                theta = np.random.uniform(low = -theta_bound + np.pi, high = theta_bound + np.pi)

    elif pickup_wall == 3:
        Y_range = [max_y - DIS_WALL_DESK , max_y ]
        X_range = [min_x + DIS_WALL_DESK2, max_x - DIS_WALL_DESK2 ]
        center_pos_x = np.random.uniform(low=X_range[0], high=X_range[1])
        center_pos_y = np.random.uniform(low=Y_range[0], high=Y_range[1])
        if (max_y - center_pos_y) >= (Desk_length/2):
            theta = np.random.uniform(low = -MAX_ROT- np.pi/2, high = MAX_ROT- np.pi/2)
        else:
            theta_bound = np.arcsin((max_y - center_pos_y)/(Desk_length/2))
            if theta_bound > MAX_ROT:
                theta = np.random.uniform(low = -MAX_ROT- np.pi/2, high = MAX_ROT- np.pi/2)
            else:
                theta = np.random.uniform(low = -theta_bound - np.pi/2, high = theta_bound - np.pi/2)
    #theta = np.pi/2
    Rot_matrix = np.array([
        [np.cos(theta), np.sin(theta)],
        [-np.sin(theta), np.cos(theta)]
    ])

    mic_positions = mic_positions.dot(Rot_matrix) + np.array([center_pos_x, center_pos_y])
    mic_positions_middle = mic_positions_middle.dot(Rot_matrix) + np.array([center_pos_x, center_pos_y])
    mic_positions_small = mic_positions_small.dot(Rot_matrix) + np.array([center_pos_x, center_pos_y])

    if is_valid_mic_array(mic_positions, left, right, bottom, top) and is_valid_mic_array(mic_positions_middle, left, right, bottom, top) and is_valid_mic_array(mic_positions_small, left, right, bottom, top):        
        if args.dimensions == 3:
            heights = MIC_HEIGHT * np.ones((mic_positions.shape[0], 1))
            mic_positions = np.concatenate([mic_positions, heights], axis=1)
            mic_positions_middle = np.concatenate([mic_positions_middle, heights], axis=1)
            mic_positions_small = np.concatenate([mic_positions_small, heights], axis=1)
        return mic_positions, mic_positions_middle ,mic_positions_small, [[Desk_length, Desk_width],[Desk_length_middle, Desk_width_middle],[Desk_length_small, Desk_width_small] ], pickup_wall
    
    else:
        print("Generated mic array violates preset rules. Re-generating ...")
        #return mic_positions
        return get_random_mic_positions_three_desks(n_mics, left, right, bottom, top, args)




def get_random_mic_positions_desk(n_mics, left, right, bottom, top, args):
    # Randomize desk dimensions
    Desk_length = np.random.uniform(low=DESK_LENGTH_MIN, high=DESK_LENGTH_MAX)
    Desk_width = np.random.uniform(low=DESK_WIDTH_MIN, high=DESK_WIDTH_MAX)
    
    middle_angle = np.arctan(Desk_length/2/Desk_width)
    
    # Get angles the robots will move in
    angle_list = np.linspace(0, np.pi, n_mics - 1) - np.pi/2
    
    # Choose position for each microphone
    mic_positions = np.zeros((n_mics, 2))
    for i in range(0, n_mics - 1):
        # Perturb robot move angle
        move_angle_err = np.random.uniform(low = -THETA_MAX_DEV, high = THETA_MAX_DEV)
        move_angle = angle_list[i] + move_angle_err

        # Robot stops at length edge
        if move_angle < middle_angle and move_angle > -middle_angle:
            expand_r = Desk_width/np.cos(move_angle)
        # Robot stops at upper width edge
        elif move_angle > middle_angle:
            expand_r = Desk_length/2/np.sin(move_angle)
        # Robot stops at lower width edge
        else:
            expand_r = Desk_length/2/np.sin(-move_angle)
        
        assert(expand_r >= 0)

        # Robot backoff
        expand_r = expand_r - 0.04

        # Compute relative position offset after array expansion
        temp_pos_x = expand_r*np.cos(move_angle) + np.random.uniform(low=-EXPAND_MAX_DEV, high=EXPAND_MAX_DEV)
        temp_pos_y = expand_r*np.sin(move_angle) + np.random.uniform(low=-EXPAND_MAX_DEV, high=EXPAND_MAX_DEV)
        mic_positions[i+1] = np.array([temp_pos_x, temp_pos_y])

    # Choose a wall to place the desk by
    pickup_wall = np.random.choice(4)

    # Minimum distance to the wall (length side) is 10cm
    DESK_WALL_MIN_DIST = 0.1
    min_x = left + DESK_WALL_MIN_DIST
    max_x = right - DESK_WALL_MIN_DIST
    min_y = bottom + DESK_WALL_MIN_DIST
    max_y = top - DESK_WALL_MIN_DIST
    
    # Maximum distance to the wall (length side) is 35cm
    DIS_WALL_DESK = 0.35

    # Minimum distance to the side walls is 180cm
    DIS_WALL_DESK2 = 1.8
    
    # Maximum desk rotation
    MAX_ROT = np.pi/8

    # If chosen wall is left wall
    if pickup_wall == 0:
        X_range = [min_x, min_x + DIS_WALL_DESK]
        Y_range = [min_y + DIS_WALL_DESK2, max_y - DIS_WALL_DESK2 ]
        center_pos_x = np.random.uniform(low=X_range[0], high=X_range[1])
        center_pos_y = np.random.uniform(low=Y_range[0], high=Y_range[1])
        if (center_pos_x - min_x) >= (Desk_length/2):
            theta = np.random.uniform(low = -MAX_ROT, high = MAX_ROT)
        else:
            theta_bound = np.arcsin((center_pos_x - min_x)/(Desk_length/2))
            if theta_bound > MAX_ROT:
                theta = np.random.uniform(low = -MAX_ROT, high = MAX_ROT)
            else:
                theta = np.random.uniform(low = -theta_bound, high = theta_bound)

    # If chosen wall is bottom wall
    elif pickup_wall == 1:
        Y_range = [min_y, min_y + DIS_WALL_DESK]
        X_range = [min_x + DIS_WALL_DESK2, max_x- DIS_WALL_DESK2 ]
        center_pos_x = np.random.uniform(low=X_range[0], high=X_range[1])
        center_pos_y = np.random.uniform(low=Y_range[0], high=Y_range[1])

        if (center_pos_y - min_y) >= (Desk_length/2):
            theta = np.random.uniform(low = -MAX_ROT+ np.pi/2, high = MAX_ROT+ np.pi/2)
        else:
            theta_bound = np.arcsin((center_pos_y - min_y)/(Desk_length/2))
            if theta_bound > MAX_ROT:
                theta = np.random.uniform(low = -MAX_ROT+ np.pi/2, high = MAX_ROT+ np.pi/2)
            else:
                theta = np.random.uniform(low = -theta_bound + np.pi/2, high = theta_bound + np.pi/2)

    # If chosen wall is right wall
    elif pickup_wall == 2:
        X_range = [max_x - DIS_WALL_DESK, max_x ]
        Y_range = [min_y + DIS_WALL_DESK2, max_y - DIS_WALL_DESK2 ]
        center_pos_x = np.random.uniform(low=X_range[0], high=X_range[1])
        center_pos_y = np.random.uniform(low=Y_range[0], high=Y_range[1])
        if (max_x - center_pos_x) >= (Desk_length/2):
            theta = np.random.uniform(low = -MAX_ROT, high = MAX_ROT)
        else:
            theta_bound = np.arcsin((max_x - center_pos_x)/(Desk_length/2))
            if theta_bound > MAX_ROT:
                theta = np.random.uniform(low = -MAX_ROT+ np.pi, high = MAX_ROT+ np.pi)
            else:
                theta = np.random.uniform(low = -theta_bound + np.pi, high = theta_bound + np.pi)

    # If chosen wall is top wall
    elif pickup_wall == 3:
        Y_range = [max_y - DIS_WALL_DESK , max_y ]
        X_range = [min_x + DIS_WALL_DESK2, max_x - DIS_WALL_DESK2 ]
        center_pos_x = np.random.uniform(low=X_range[0], high=X_range[1])
        center_pos_y = np.random.uniform(low=Y_range[0], high=Y_range[1])
        if (max_y - center_pos_y) >= (Desk_length/2):
            theta = np.random.uniform(low = -MAX_ROT- np.pi/2, high = MAX_ROT- np.pi/2)
        else:
            theta_bound = np.arcsin((max_y - center_pos_y)/(Desk_length/2))
            if theta_bound > MAX_ROT:
                theta = np.random.uniform(low = -MAX_ROT- np.pi/2, high = MAX_ROT- np.pi/2)
            else:
                theta = np.random.uniform(low = -theta_bound - np.pi/2, high = theta_bound - np.pi/2)

    Rot_matrix = np.array([
        [np.cos(theta), np.sin(theta)],
        [-np.sin(theta), np.cos(theta)]
    ])

    # Rotate desk
    mic_positions = mic_positions.dot(Rot_matrix) + np.array([center_pos_x, center_pos_y])
    
    if is_valid_mic_array(mic_positions, left, right, bottom, top):        
        if args.dimensions == 3:
            heights = MIC_HEIGHT * np.ones((mic_positions.shape[0], 1))
            mic_positions = np.concatenate([mic_positions, heights], axis=1)

        return mic_positions, [Desk_length, Desk_width], pickup_wall
    else:
        print("Generated mic array violates preset rules. Re-generating ...")
        #return mic_positions
        return get_random_mic_positions_desk(n_mics, left, right, bottom, top, args)

def colocated_array(center, args, diameter = ECHO_DOT_DIAMETER):
    """
    Returns positions of a circular microphone at the center with given diameter
    """
    radius = diameter / 2

    # Choose random starting rotation
    random_phi0 = np.random.uniform(0, 2 * np.pi)
    mic_positions = pra.circular_2D_array(center[..., :2], M=args.n_mics, radius=radius, phi0=random_phi0)

    # Add height if simulation is 3 dimensional
    if args.dimensions == 3:
        heights = MIC_HEIGHT * np.ones(mic_positions.shape[-1]).reshape(1, -1)
        mic_positions = np.concatenate([mic_positions, heights])

    return mic_positions.T

def choose_point_with_keepout(left, right, up, down, btmright_x, btmright_y, h, w):
    voice_x = np.random.uniform(low=left, high=right)
    voice_y = np.random.uniform(low=down, high=up)

    if voice_x >= btmright_x and voice_x - btmright_x <= w and voice_y >= btmright_y and voice_y - btmright_y <= h:
        return choose_point_with_keepout(left, right, up, down, btmright_x, btmright_y, h, w)

    return np.array([voice_x, voice_y])

# Also, sampling rate and speed of sound should be global constants
def calculate_sample_offset(mic_positions, source_pos, sr):
    offsets = []
    for i in range(1, mic_positions.shape[0]):
        dis_offset = np.linalg.norm(source_pos - mic_positions[i]) - np.linalg.norm(source_pos - mic_positions[0])
        sample_offset = dis_offset/SPEED_OF_SOUND*sr
        offsets.append(sample_offset)
    return np.array(offsets)

def get_random_speaker_positions(n_voices, mic_positions, pickup_wall, left, right, up, down, args):
    # Get an expected bounding box of table
    mic_array_minx = np.min(mic_positions[:, 0])
    mic_array_miny = np.min(mic_positions[:, 1])
    mic_array_maxx = np.max(mic_positions[:, 0])
    mic_array_maxy = np.max(mic_positions[:, 1])

    h = mic_array_maxy - mic_array_miny
    w = mic_array_maxx - mic_array_minx
    
    # Place at least sources 25 cm away from table
    KEEPOUT = 0.25

    h = h + 2 * KEEPOUT
    w = w + 2 * KEEPOUT
    mic_array_minx -= KEEPOUT
    mic_array_miny -= KEEPOUT
    mic_center = mic_positions[0]


    ### place the desk near a wall 
    if pickup_wall == 0:
        SPEAK_MINX = max([mic_center[0] + KEEPOUT, left + WALL_KEEPOUT])
        SPEAK_MAXX = min([mic_center[0] + SPK_RANGE_H, right - WALL_KEEPOUT])
        SPEAK_MINY = max([mic_center[1] - SPK_RANGE_W, down + WALL_KEEPOUT])
        SPEAK_MAXY = min([mic_center[1] + SPK_RANGE_W, up - WALL_KEEPOUT])
    elif pickup_wall == 1:
        SPEAK_MINX = max([mic_center[0] - SPK_RANGE_W, left + WALL_KEEPOUT])
        SPEAK_MAXX = min([mic_center[0] + SPK_RANGE_W, right - WALL_KEEPOUT])
        SPEAK_MINY = max([mic_center[1] + KEEPOUT, down + WALL_KEEPOUT])
        SPEAK_MAXY = min([mic_center[1] + SPK_RANGE_H, up - WALL_KEEPOUT])
    elif pickup_wall == 2:
        SPEAK_MINX = max([mic_center[0] - SPK_RANGE_H, left + WALL_KEEPOUT])
        SPEAK_MAXX = min([mic_center[0] - KEEPOUT, right - WALL_KEEPOUT])
        SPEAK_MINY = max([mic_center[1] - SPK_RANGE_W, down + WALL_KEEPOUT])
        SPEAK_MAXY = min([mic_center[1] + SPK_RANGE_W, up - WALL_KEEPOUT])
    elif pickup_wall == 3:
        SPEAK_MINX = max([mic_center[0] - SPK_RANGE_W, left + WALL_KEEPOUT])
        SPEAK_MAXX = min([mic_center[0] + SPK_RANGE_W, right - WALL_KEEPOUT])
        SPEAK_MINY = max([mic_center[1] - SPK_RANGE_H, down + WALL_KEEPOUT])
        SPEAK_MAXY = min([mic_center[1] - KEEPOUT, up - WALL_KEEPOUT])
    
    ### the region of interest to put the speaker positions
    ROI = [SPEAK_MINX - 0.1, SPEAK_MAXX + 0.1, SPEAK_MINY - 0.1, SPEAK_MAXY + 0.1, MIN_SPEAKER_HEIGHT - 0.1, MIN_SPEAKER_HEIGHT + MAX_SPEAKER_HEIGHT + 0.1]
    voices = []
    offsets = []
    for i in range(n_voices):
        success = False
        while not success:
            success = True
            pos = choose_point_with_keepout(SPEAK_MINX, SPEAK_MAXX, SPEAK_MINY, SPEAK_MAXY, mic_array_minx, mic_array_miny, h, w)

            ### select the random height for speaker
            if args.dimensions == 3:
                z = np.random.random() * MAX_SPEAKER_HEIGHT + MIN_SPEAKER_HEIGHT
                pos = np.concatenate([pos, np.array([z])])

            offset = calculate_sample_offset(mic_positions, pos, args.sr)
            for j, pos2 in enumerate(voices):
                if np.linalg.norm(pos2 - pos) < MIN_SPEAKER_DIST:
                    print("Retrying random voice location generation ... because source too close")
                    success = False
                    break
        
        voices.append(pos)
        offsets.append(offset)
    return voices, offsets,ROI

def generate_data_scenario(mic_positions, voice_positions, voices_data, room_dimensions,
                           absorption, max_order, args):
    total_samples = voices_data[0][0].shape[-1]
    # Create room
    room = pra.ShoeBox(p=room_dimensions, fs=args.sr, max_order=max_order, absorption = absorption)
    
    # Add microphone array to rooom
    room.add_microphone_array(mic_positions.T)

    # Add speakers to room
    for voice_idx in range(len(voice_positions)):
        voice_loc = voice_positions[voice_idx]
        room.add_source(voice_loc, signal=voices_data[voice_idx][0])

    # Simulate room, return without premix to get ground truth
    premix_reverb = room.simulate(return_premix=True)

    # Get ground truth signals
    gt_signals = np.zeros((len(voice_positions), mic_positions.shape[0], total_samples))
    for i in range(len(voice_positions)):
        for j in range(mic_positions.shape[0]):
            gt_signals[i][j] = np.pad(premix_reverb[i][j], (0,total_samples))[:total_samples]

    # Get mixture signals
    mix_reverb = np.sum(premix_reverb, axis=0)
    input_signals = np.zeros((mic_positions.shape[0], total_samples))
    for i in range(mic_positions.shape[0]):
        input_signals[i] = np.pad(mix_reverb[i], (0, total_samples))[:total_samples]

    if args.generate_dereverb:
        # Create room again, this time without reverberation
        room = pra.ShoeBox(p=room_dimensions, fs=args.sr, max_order=0)
        
        # Add microphone array to rooom
        room.add_microphone_array(mic_positions.T)

        # Add speakers to room
        for voice_idx in range(len(voice_positions)):
            voice_loc = voice_positions[voice_idx]
            room.add_source(voice_loc, signal=voices_data[voice_idx][0])

        # Simulate room, return without premix to get ground truth
        premix_dereverb = room.simulate(return_premix=True)
        gt_signals_dereverb = np.zeros((len(voice_positions), mic_positions.shape[0], total_samples))
        for i in range(len(voice_positions)):
            for j in range(mic_positions.shape[0]):
                gt_signals_dereverb[i][j] = np.pad(premix_dereverb[i][j], (0,total_samples))[:total_samples]
        
        return input_signals, (gt_signals, gt_signals_dereverb)


    return input_signals, gt_signals

def save_scenario(output_prefix_dir, input_signals, gt_signals, mic_positions,
                  voice_positions, voice_offsets, voices_data, room_dimensions, 
                  desk_dimensions, Pick_wall, ROI, absorption, args, rt60 = None):
    
    if args.generate_dereverb:
        gt_signals, gt_signals_dereverb = gt_signals

    n_voices = len(voice_positions)
    total_samples = input_signals.shape[-1]
    included_sources = np.arange(len(voice_positions))

    # Save microphone recordings
    for mic_idx in range(args.n_mics):
        output_prefix = str(Path(output_prefix_dir) / "mic{:02d}_".format(mic_idx))
        
        # Go over each voice
        for voice_idx in range(n_voices):
            voice_at_mic = gt_signals[voice_idx][mic_idx]
            assert voice_at_mic.shape[-1] == total_samples

            # Save voice at reference microphone as ground truth
            if mic_idx == 0:
                if voice_idx in included_sources:
                    fname = os.path.join(output_prefix_dir, ('mic{:02d}_voice{:02d}'.format(mic_idx, voice_idx) + '.wav'))
                    utils.write_audio_file(fname, voice_at_mic, args.sr)
                    
                    if args.generate_dereverb:
                        voice_at_mic_dereverb = gt_signals_dereverb[voice_idx][mic_idx]
                        fname = os.path.join(output_prefix_dir, ('mic{:02d}_voice{:02d}_dereverb'.format(mic_idx, voice_idx) + '.wav'))
                        utils.write_audio_file(fname, voice_at_mic_dereverb, args.sr)

        # Save mixture signal at reference microphone
        assert input_signals.shape[-1] == total_samples
        sf.write(output_prefix + "mixed.wav", input_signals[mic_idx], args.sr)
    
    # Create metadata
    metadata = {}

    # Store speaker information
    for voice_idx in range(n_voices):
        metadata['voice{:02d}'.format(voice_idx)] = {
            'position': voice_positions[voice_idx].tolist(),
            'shifts': np.round(voice_offsets[voice_idx]).astype(np.int32).tolist(),
            'speaker_id': voices_data[voice_idx][1]
        }
    
    # Store microphone information
    for mic_idx in range(args.n_mics):
        metadata['mic{:02d}'.format(mic_idx)] = {
            'position': list(mic_positions[mic_idx])
        }
    if rt60 is not None:
        metadata["rt60"] = rt60
    # Store setting information
    metadata['Room_dimensions'] = room_dimensions
    metadata['Desk_size'] = desk_dimensions
    metadata['Pick_wall'] = Pick_wall
    metadata['ROI'] = ROI
    metadata['absorption'] = absorption
    
    # Indicate that this is a synthetically-generated sample
    metadata['real'] = False

    # Write metadata
    metadata_file = str(Path(output_prefix_dir) / "metadata.json")
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=4)

def generate_sample(voices_list: list, args: argparse.Namespace, subdir: str, idx: int) -> int:  
    # Create output directory
    output_prefix_dir = os.path.join(args.output_path, subdir, '{:05d}'.format(idx))
    Path(output_prefix_dir).mkdir(parents=True, exist_ok=True)

    # Choose a random number of speakers from the bounds
    n_voices = np.random.randint(args.n_voices_min, args.n_voices_max+1)
    voices_data = get_voices(voices_list, n_voices, args)

    # Generate room parameters, each scene has a random room and absorption
    # Choose room dimensions
    room_length = np.random.uniform(low=ROOM_LENGTH_MIN, high=ROOM_LENGTH_MAX)
    room_width = np.random.uniform(low=ROOM_WIDTH_MIN, high=ROOM_WIDTH_MAX)
    left, right, bottom, top = 0, room_length, 0, room_width
    ceiling = np.random.uniform(low=CEIL_MIN, high=CEIL_MAX)

    room_dimensions = [room_length, room_width]
    if args.dimensions == 3:
        room_dimensions.append(ceiling)
    
    # Choose room absorption
    absorption = np.random.uniform(low=MIN_ABSORPTION, high=MAX_ABSORPTION)

    # Randomize microphone positions
    mic_positions, desk_dimensions, pickup_wall = \
        get_random_mic_positions_desk(n_mics = args.n_mics, left=left, right=right,
                                      bottom=bottom, top=top, args=args)
    
    # Randomize speaker positions
    voice_positions, voice_offsets, ROI = \
        get_random_speaker_positions(n_voices, mic_positions, pickup_wall, left=left, 
                                     right=right, up=top, down=bottom, args=args)
 
    mic_positions = np.array(mic_positions)
    voice_positions = np.array(voice_positions)


    
    # Sanity check, make sure all microphones and sources are in the room 
    for pos in voice_positions:
        assert point_in_box(pos[:2], left, right, top, bottom)
    for pos in mic_positions:
        assert point_in_box(pos[:2], left, right, top, bottom)


    if args.sample_rt60:
        rt_60 = np.random.uniform(0.08, 0.7)
        not_finish = True
        while not_finish:
            try:
                absorption, max_order = pra.inverse_sabine(rt_60, room_dimensions)
            except ValueError:
                rt_60 = rt_60 + 0.02
                print("fail and try larger rt = ", rt_60)
                continue
            break
        if max_order > 150:
            max_order = 150
    else:
        max_order = args.max_order

    # Simulate room and get outputs
    input_signals, gt_signals = \
        generate_data_scenario(mic_positions=mic_positions,
                               voice_positions=voice_positions,
                               voices_data=voices_data,
                               room_dimensions=room_dimensions,
                               absorption=absorption,
                               max_order=max_order,
                               args=args)

    # Save results
    save_scenario(output_prefix_dir=output_prefix_dir,
                  input_signals=input_signals,
                  gt_signals=gt_signals,
                  mic_positions=mic_positions,
                  voice_positions=voice_positions,
                  voice_offsets=voice_offsets,
                  voices_data=voices_data,
                  room_dimensions=room_dimensions,
                  desk_dimensions=desk_dimensions,
                  Pick_wall=pickup_wall,
                  ROI = ROI,
                  absorption=absorption,
                  args=args)
    
    # If we also want to generate a colocated array (Echo Dot, for example)
    if args.generate_colocated:
        # Create ouput directory
        colocated_output_prefix_dir = \
            os.path.join(args.output_path.rstrip('/') + '_colocated', subdir, '{:05d}'.format(idx))
        Path(colocated_output_prefix_dir).mkdir(parents=True, exist_ok=True)

        # Get colocated microphone positions
        colocated_mic_positions = colocated_array(center=np.mean(mic_positions, axis=0), args=args)

        # Simulate
        input_signals, gt_signals = \
            generate_data_scenario(mic_positions=colocated_mic_positions,
                                   voice_positions=voice_positions,
                                   voices_data=voices_data,
                                   room_dimensions=room_dimensions,
                                   absorption=absorption,
                                   max_order=max_order,
                                   args=args)
        
        # Compute voice offsets in the colocated case
        colocated_voice_offsets = [calculate_sample_offset(colocated_mic_positions, v, args.sr) for v in voice_positions]

        # Save results
        save_scenario(output_prefix_dir=colocated_output_prefix_dir,
                      input_signals=input_signals,
                      gt_signals=gt_signals,
                      mic_positions=colocated_mic_positions,
                      voice_positions=voice_positions,
                      voice_offsets=colocated_voice_offsets,
                      voices_data=voices_data,
                      room_dimensions=room_dimensions,
                      desk_dimensions=desk_dimensions,
                      Pick_wall=pickup_wall,
                      ROI = ROI,
                      absorption=absorption,
                      args=args)


def generate_sample_size(voices_list: list, args: argparse.Namespace, subdir: str, idx: int) -> int:  


    filenames = ["large", "middle", "small"]

    # Choose a random number of speakers from the bounds
    n_voices = np.random.randint(args.n_voices_min, args.n_voices_max+1)
    voices_data = get_voices(voices_list, n_voices, args)

    # Generate room parameters, each scene has a random room and absorption
    # Choose room dimensions
    room_length = np.random.uniform(low=ROOM_LENGTH_MIN, high=ROOM_LENGTH_MAX)
    room_width = np.random.uniform(low=ROOM_WIDTH_MIN, high=ROOM_WIDTH_MAX)
    left, right, bottom, top = 0, room_length, 0, room_width
    ceiling = np.random.uniform(low=CEIL_MIN, high=CEIL_MAX)

    room_dimensions = [room_length, room_width]
    if args.dimensions == 3:
        room_dimensions.append(ceiling)
    
    # Choose room absorption
    absorption = np.random.uniform(low=0.1, high=0.99)

    # Randomize microphone positions
    mic_positions, mic_positions_middle, mic_positions_small, desk_dimensions, pickup_wall = \
        get_random_mic_positions_three_desks(n_mics = args.n_mics, left=left, right=right,
                                      bottom=bottom, top=top, args=args)
    pos_list = [mic_positions, mic_positions_middle, mic_positions_small]

    # Randomize speaker positions
    voice_positions, voice_offsets, ROI = \
        get_random_speaker_positions(n_voices, mic_positions, pickup_wall, left=left, 
                                     right=right, up=top, down=bottom, args=args)
    
    ### calculate the sample offset ground-truth for 3 desks
    offsets_large = []
    for pos in voice_positions:
        offset = calculate_sample_offset(mic_positions, pos, args.sr)
        offsets_large.append(offset)
    offsets_large = np.array(offsets_large)

    offsets_middle = []
    for pos in voice_positions:
        offset = calculate_sample_offset(mic_positions_middle, pos, args.sr)
        offsets_middle.append(offset)
    offsets_middle = np.array(offsets_middle)

    offsets_small = []
    for pos in voice_positions:
        offset = calculate_sample_offset(mic_positions_small, pos, args.sr)
        offsets_small.append(offset)
    offsets_small = np.array(offsets_small)
    offsets_list = [offsets_large, offsets_middle, offsets_small]

    mic_positions = np.array(mic_positions)
    mic_positions_middle = np.array(mic_positions_middle)
    mic_positions_small = np.array(mic_positions_small)
    voice_positions = np.array(voice_positions)

    # Sanity check, make sure all microphones and sources are in the room 
    for pos in voice_positions:
        assert point_in_box(pos[:2], left, right, top, bottom)
    for pos in mic_positions:
        assert point_in_box(pos[:2], left, right, top, bottom)
    for pos in mic_positions_middle:
        assert point_in_box(pos[:2], left, right, top, bottom)
    for pos in mic_positions_small:
        assert point_in_box(pos[:2], left, right, top, bottom)

    for i in range(len(filenames)):
        fname = filenames[i]
        mic_pos = pos_list[i]
        offsets = offsets_list[i]
        desk_dimension = desk_dimensions[i]
        output_prefix_dir = os.path.join(args.output_path, fname + '/{:05d}'.format(idx))
        Path(output_prefix_dir).mkdir(parents=True, exist_ok=True)
        # Simulate room and get outputs 
        input_signals, gt_signals = \
            generate_data_scenario(mic_positions=mic_pos,
                               voice_positions=voice_positions,
                               voices_data=voices_data,
                               room_dimensions=room_dimensions,
                               absorption=absorption,
                               max_order=args.max_order,
                               args=args)

        # Save results
        save_scenario(output_prefix_dir=output_prefix_dir,
                    input_signals=input_signals,
                    gt_signals=gt_signals,
                    mic_positions=mic_pos,
                    voice_positions=voice_positions,
                    voice_offsets=offsets,
                    voices_data=voices_data,
                    room_dimensions=room_dimensions,
                    desk_dimensions=desk_dimension,
                    Pick_wall=pickup_wall,
                    ROI = ROI,
                    absorption=absorption,
                    args=args)


def generate_sample_rt60(voices_list: list, args: argparse.Namespace, subdir: str, idx: int) -> int:  
    """
    Generate a single sample. Return 0 on success.

    Steps:
    - [1] Load voice
    - [2] Sample background with the same length as voice. //may not
    - [3] Pick background location
    - [4] Create a scene
    - [5] Render sound
    - [6] Save metadata
    """
    global SPK_RANGE_W, SPK_RANGE_H
    SPK_RANGE_W = 1.6
    SPK_RANGE_H = 2.4
    # [1] load voice
    n_voices = np.random.randint(args.n_voices_min, args.n_voices_max+1)
    voices_data = get_voices(voices_list, n_voices, args)

    # [3]
    # Generate room parameters, each scene has a random room and absorption
    # Choose room dimensions
    room_length = np.random.uniform(low=4, high=4.2)
    room_width = np.random.uniform(low=4, high=4.2)
    left, right, bottom, top = 0, room_length, 0, room_width
    ceiling = np.random.uniform(low=1.6, high=1.7)

    room_dimensions = [room_length, room_width]
    if args.dimensions == 3:
        room_dimensions.append(ceiling)

    # [4]
    # Compute mic positions

    # Randomize microphone positions
    mic_positions, desk_dimensions, pickup_wall = \
        get_random_mic_positions_desk(n_mics = args.n_mics, left=left, right=right,
                                      bottom=bottom, top=top, args=args)
    
    # Randomize speaker positions
    voice_positions, voice_offsets, ROI = \
        get_random_speaker_positions(n_voices, mic_positions, pickup_wall, left=left, 
                                     right=right, up=top, down=bottom, args=args)
    
    mic_positions = np.array(mic_positions)
    voice_positions = np.array(voice_positions)
    # Sanity check, make sure all microphones and sources are in the room 
    for pos in voice_positions:
        assert point_in_box(pos[:2], left, right, top, bottom)
    for pos in mic_positions:
        assert point_in_box(pos[:2], left, right, top, bottom)


    rt60_lists = [[0.08, 0.2], [0.2, 0.4], [0.4, 0.6], [0.6, 0.8]]
    #rt60_lists = [ [0.2, 0.4], [0.6, 0.8]]
    for i in range(len(rt60_lists)):
        #if i <= 1: continue
        rt_60 = np.random.uniform(rt60_lists[i][0], rt60_lists[i][1])
        ### convert rt60 to absorption and max order
        not_finish = True
        while not_finish:
            try:
                e_absorption, max_order = pra.inverse_sabine(rt_60, room_dimensions)
            except ValueError:
                rt_60 = rt_60 + 0.02
                print("fail and try larger rt = ", rt_60)
                continue
            break
        #print(e_absorption, max_order, rt_60)
        ### set maximum order to avoid too long processing time
        if max_order > 150:
            max_order = 150

        input_signals, gt_signals = \
        generate_data_scenario(mic_positions=mic_positions,
                               voice_positions=voice_positions,
                               voices_data=voices_data,
                               room_dimensions=room_dimensions,
                               absorption=e_absorption,
                               max_order=max_order,
                               args=args)

        output_prefix_dir = os.path.join(args.output_path, 'rt_60_{}'.format(i), '{:05d}'.format(idx))
        Path(output_prefix_dir).mkdir(parents=True, exist_ok=True)
        #print("Saving.....", output_prefix_dir)
        # Save results
        save_scenario(output_prefix_dir=output_prefix_dir,
                    input_signals=input_signals,
                    gt_signals=gt_signals,
                    mic_positions=mic_positions,
                    voice_positions=voice_positions,
                    voice_offsets=voice_offsets,
                    voices_data=voices_data,
                    room_dimensions=room_dimensions,
                    desk_dimensions=desk_dimensions,
                    Pick_wall=pickup_wall,
                    ROI = ROI,
                    absorption=e_absorption,
                    args=args,
                    rt60 = rt_60)



def main(args: argparse.Namespace):
    # Seed everything
    utils.seed_all(args.seed)
    
    # Load train/test/val split
    with open(args.split_path, 'rb') as f:
        split_data = json.load(f)

    # Generate train/test/va; sets
    for subdir, voices in split_data.items():
        voices_list = [os.path.join(args.input_voice_dir, x) for x in voices]
        
        if len(voices_list) == 0:
            raise ValueError("No voice files found")

        n_outputs = getattr(args, "n_outputs_" + subdir)
        pbar = tqdm.tqdm(total=n_outputs)

        
        # Generate samples on multiple processes
        num_workers = min(multiprocessing.cpu_count(), args.n_workers)
        pool = mp.Pool(num_workers)
        if args.generate_rt60:
            for i in range(n_outputs):
                generate_sample_rt60(voices_list, args, subdir, i)
                pbar.update()
        elif args.generate_size:
            for i in range(n_outputs):
                generate_sample_size(voices_list, args, subdir, i)
                pbar.update()            
        else:
            for i in range(n_outputs):
                generate_sample(voices_list, args, subdir, i)
                pbar.update()

        pool.close()
        pool.join()
        pbar.close()
        
    
    # Save arguments used to generate dataset
    with open(os.path.join(args.output_path, 'args.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_voice_dir',
                        type=str,
                        help="Directory with voice wav files")
    parser.add_argument('output_path', type=str, help="Output directory to write the synthetic dataset")
    parser.add_argument('--split_path',
                        type=str,
                        default='datasets/vctk_split.json')
    parser.add_argument('--n_mics', type=int, default=7)

    parser.add_argument('--n_voices_min', type=int, default=3)
    parser.add_argument('--n_voices_max', type=int, default=5)

    parser.add_argument('--n_outputs_train', type=int, default=0)
    parser.add_argument('--n_outputs_test', type=int, default=0)
    parser.add_argument('--n_outputs_val', type=int, default=0)

    parser.add_argument('--n_workers', type=int, default=16)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--sr', type=int, default=48000)
    parser.add_argument('--start_index', type=int, default=0)

    parser.add_argument('--dimensions', type=int, default=3, choices=(2, 3))
    parser.add_argument('--generate_colocated',
                        action='store_true',
                        help='Whether or not to generate each scenario with a\
                              colocated microphone array as well.')

    parser.add_argument('--generate_rt60',
                        action='store_true',
                        help='Whether or not to generate each scenario with \
                              different rt60 as well.')

    parser.add_argument('--generate_size',
                        action='store_true',
                        help='Whether or not to generate each scenario with \
                              different desk sizes as well.')
    
    parser.add_argument('--generate_dereverb',
                        action='store_true',
                        help='Whether or not to generate each scenario\
                            dereverberated ground truth as well.')


    parser.add_argument('--sample_rt60',
                        action='store_true',
                        help='Whether or not to use fixed max-order or random rt60.')

    parser.add_argument('--max_order',
                        type=int,
                        default=15,
                        help="Order of reflections to consider during simulation.\
                              Used to speed up dataset generation at the cost of\
                              simulation accuracy.")

    parser.add_argument('--duration', type=float, default=3.0)
    main(parser.parse_args())
    
