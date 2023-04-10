# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 16:23:47 2023

Contains all major functions to be used in this module

@author: Tanjim
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

from scipy.signal import find_peaks
from scipy.signal import savgol_filter
from matplotlib.animation import FuncAnimation

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

def get_fps(path):
    """
    Returns fps of video at path

    Parameters
    ----------
    path : str
        path to video.

    Returns
    -------
    fps : float
        frames per second of video at path.

    """
    
    # load video at path
    cap = cv2.VideoCapture('test_footage/tricep_pushdown_test.mp4')
    
    # obtain fps
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    return fps


fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 60.0, (640, 480))


def generate_landmarks(path, show_video=False):
    """
    Generates a list of landmarks for each frame in video located at path.

    Parameters
    ----------
    path : str
        Path to video to be analysed.
    show_video : bool, optional
        Boolean switch to show video with pose estimation on top. 
        Press key "q" to close video.
        The default is False.

    Returns
    -------
    landmarks : list
        List of lists of landmarks for each frame in video

    """
    
    # load video 
    cap = cv2.VideoCapture(path)
    
    # results stored below
    landmarks = []
    
    # initiate mp pose
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
    
      # loop through each frame in video  
      while cap.isOpened():
        success, image = cap.read()
        
        # loop ends when all frames have been scanned
        if not success:
            break

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        
        # apply pose estimation and save results to list landmarks
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        landmarks.append(results)
        
        if show_video:
            # Optionally, video will be shown
            # Draw the pose annotation on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            cv2.imshow('MediaPipe Pose', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # Write the current frame to the video file
            out.write(image)
      
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    return landmarks

def get_points(result_list):
    """
    Takes a list of pose landmark objects and returns points.
    Also returns the visibility of each point
    
    The hands and feet have multiple landmarks such as the wrist and thumb. The
    position returned is the average position of these landmarks.

    Parameters
    ----------
    result_list : LIST
        List of mediapipe.python.solution_base.SolutionOutputs objects as output
        by generate_landmarks

    Returns
    -------
    Tuple of (x, y, z, visibility)

    """

    # Define the landmark indices to extract
    # these are        [LS, RS, LE, RE, LP, RP, LK, RK, LH            , RH            , LF        , RF        ]
    landmark_indices = [11, 12, 13, 14, 23, 24, 25, 26, 15, 17, 19, 21, 16, 18, 20, 22, 27, 29, 31, 28, 30, 32]
    
    x = []
    y = []
    z = []
    visibility = []

    for count, frame in enumerate(result_list):
        
        # make sure this frame has a landmark
        if frame.pose_landmarks is not None:
            # Extract the landmark coordinates and visibility
            landmarks = np.array([[lmk.x, lmk.y, lmk.z, lmk.visibility] for lmk in frame.pose_landmarks.landmark])
            selected_landmarks = landmarks[landmark_indices]
            
            # Save values for shoulders, pelvis, knees and elbows
            single_landmarks = selected_landmarks[:8]
            
            x_coords = single_landmarks[:, 0]
            y_coords = single_landmarks[:, 1]
            z_coords = single_landmarks[:, 2]
            vis = single_landmarks[:, 3]
            
            # Compute the mean values for each hand and each foot
            remaining_landmarks = selected_landmarks[8:]
            grouped_landmarks = np.split(remaining_landmarks, [4, 8, 11])
            mean_values = np.array([np.mean(group, axis=0) for group in grouped_landmarks])
            
            # put all the values into one list
            x_coords = np.append(x_coords, mean_values[:, 0])
            y_coords = np.append(y_coords, mean_values[:, 1])
            z_coords = np.append(z_coords, mean_values[:, 2])
            vis = np.append(vis, mean_values[:, 3])
            
        elif count == 0:
            # No landmark detected in first frame then just set all to 0
            x_coords = [0]*12
            y_coords = [0]*12
            z_coords = [0]*12
            vis = [0]*12           
            
        else:
            # No landmark detected in this frame so just copy the last set of
            # points
            x_coords = x[-1]
            y_coords = y[-1]
            z_coords = z[-1]
            vis = visibility[-1]
            
        # add as an extra frame
        x.append(x_coords)
        y.append(y_coords)
        z.append(z_coords)
        visibility.append(vis)

    
    # convert the lists to arrays
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    visibility = np.array(visibility)
    
    
    return (x, y, z, visibility)

def part_list():
    """
    Returns part list
    """
    return ['ls', 'rs', 'le', 're', 'lp', 'rp', 'lk', 'rk', 'lh', 'rh', 'lf', 'rf']

def part_code(part):
    """
    Get code associated with string body part

    Parameters
    ----------
    part : str
        The body part you want the code for. E.g:
        rh = right-hand
        le = left-elbow
        lk = left-knee
        rf = right-foot

    Returns
    -------
    code : int
        The code associated with input body part

    """
    # these are        [LS, RS, LE, RE, LP, RP, LK, RK, LH            , RH            , LF        , RF        ]
    part_codes = {"ls":0,"rs":1,"le":2,"re":3,"lp":4,"rp":5,"lk":6,"rk":7,"lh":8,"rh":9,"lf":10,"rf":11}
    
    return (part_codes[part])

def length_part(part, dim):
    """
    Outputs the approximate distance travelled by a body part in the dimension 
    dim.

    Parameters
    ----------
    part : str
        The body part you want distance travelled to be measured.
    dim : np.array
        The dimension of motion which can be x, y or z.
        These are np arrays that contain coordinates

    Returns
    -------
    total_diff : float
        The total distance travelled by the body part in dimension dim

    """
    
    # get the column number for this body part
    code = part_code(part)
    
    # reduce to just this column
    arr = dim[:,code]
    
    # remove anomalies
    mean = np.nanmean(arr)
    std = np.nanstd(arr)
    arr = arr[np.where((arr > mean - 2 * std) & (arr < mean + 2 * std))]
    
    diff_arr = np.abs(np.diff(arr))
    total_diff = np.sum(diff_arr)
    
    return total_diff

def moving_part(x, y, visibility):
    """
    Returns the visible body part that has moved the most

    Parameters
    ----------
    x, y, visibility : numpy array
        Numpy arrays of points in space for each body part/visibility as output
        by get_points()

    Returns
    -------
    result : str
        The str description of body part that is moving.

    """
    
    # create a dataframe containing the distance in each dimension each part
    # travels. also include the visibility 
    
    distances_dict = {'part': [], 'dim': [], 'distance': []}
    
    for part in part_list():
        for count, dim in enumerate([x, y]):
            distances_dict["part"].append(part)
            distances_dict["dim"].append('xy'[count])
            distances_dict["distance"].append(length_part(part, dim))
    
    distances_df = pd.DataFrame(distances_dict)
    
    mean_visibility = np.mean(visibility, axis=0)
    distances_df["visibility"] = np.repeat(mean_visibility, 2)
    
    # filter out any parts that have a visibility under 0.8
    visible_df = distances_df[distances_df.visibility > 0.8]
    
    # get part and dim with the maximum distance
    max_index = visible_df['distance'].idxmax()
    result = visible_df.loc[max_index, ['part', 'dim']]
    
    return result

def moving_part_points(x, y, visibility):
    """
    Returns the points in space of visible body part that has moved the most

    Parameters
    ----------
    x, y, visibility : numpy array
        Numpy arrays of points in space for each body part/visibility as output
        by get_points()

    Returns
    -------
    points: numpy array
        Numpy array of the points in space for body part that has travelled the
        furthest distance.

    """
    
    moving_result = moving_part(x, y, visibility)
    
    moving_dim = moving_result.dim
    moving_part_code = part_code(moving_result.part)
    
    # Define a dictionary to map the dimension codes to the corresponding arrays
    dim_dict = {"x": x, "y": y}
    
    # extract the points of the desired part from desired dimension array
    points = dim_dict[moving_dim][:, moving_part_code]
    
    return points 

def fix_data(x):
    """ 
    For get_reps to work, you need the general starting position to have a 
    low value before oscillations begin.
    
    Some exercises this is not the case so we simply have to flip the data
    
    This function detects if this is needed and then applies the flip
    
    Also applies other fixes like normalisation and anomaly removal
    """
    
    # remove any anomalous points
    mean = np.nanmean(x)
    std = np.nanstd(x)

    x = x[np.where((x > mean - 2 * std) & (x < mean + 2 * std))]
    
    # normalise the points
    x = np.interp(x, (x.min(), x.max()), (0, 1))
    
    # get average value of first few data points
    starting_avg = np.mean(x[30:70])
    
    # now get the maximum and minimum values
    max_val = max(x)
    min_val = min(x)
    
    # the desired situation is that max - avg > avg - min
    if (max_val - starting_avg) < (starting_avg - min_val):
        return -x+max_val
    else:
        return x
    
def get_rep_df(x, fps, height_tolerance=0.7, period_tolerance = 0.7):
    """
    Return the rep start, rep peak and rep end for each rep in array x
    
    These are given in unit frame number (index of x)
    
    The tolerance refers to how leniant the definition of a peak is

    Parameters
    ----------
    x : numpy array
        A 1D array of points for a body part as output by moving_part_points().
    fps: float
        The fps of the source video
    height_tolerance : float, optional
        Leniancy in height of rep peaks. The default is 0.7.
    period_tolerance : float, optional
        Leniancy in separation between rep peaks. The default is 0.7.

    Returns
    -------
    rep_df : pandas DataFrame
        Contains rep start, rep peak and rep end information.

    """
    
    
    # make sure the starting position is at the bottom
    x = fix_data(x)
    
    # smooth the data
    x = savgol_filter(x,71,2)
    
    # set the min period as 1.5s which is fps*1.5
    period = int(1.5*fps)
    
    # find the peaks in the signal
    rep_peaks, _ = find_peaks(x, height = height_tolerance, distance = period * period_tolerance)
    
    # now find the troughs on smoothed data that has been normalised
    flipped_data = -x
    flipped_data = np.interp(flipped_data, (flipped_data.min(), flipped_data.max()), (0, 1))
    # height tolerance is a bit more leniant for troughs
    troughs, _ = find_peaks(flipped_data, distance  = period * period_tolerance, height=height_tolerance-0.1)
    
    # some consecutive peaks are just flat regions 
    # make sure there is a trough between peaks else the two peaks merge
    # iterate through the peaks list
    i = 0
    while i < len(rep_peaks) - 1:
        # check if there are any troughs between the current and next peak
        if not any(rep_peaks[i] < t < rep_peaks[i+1] for t in troughs):
            # if there are no troughs, remove the next peak
            rep_peaks = np.delete(rep_peaks, i+1)
        else:
            # otherwise, move to the next peak
            i += 1
    
    
    # troughs either side of rep peaks are the rep starts and ends
    # Create boolean masks indicating valid troughs for each peak
    lower_mask = troughs < rep_peaks.reshape(-1, 1)
    upper_mask = troughs > rep_peaks.reshape(-1, 1)

    # Find the largest trough value below each peak
    max_below = np.nanmax(np.where(lower_mask, troughs, np.nan), axis=1)
    # if no trough before first peak, then fill 0
    max_below = np.nan_to_num(max_below, nan=0)
    
    # Find the smallest trough value above each peak
    min_above = np.nanmin(np.where(upper_mask, troughs, np.nan), axis=1)
    # if no peak after last peak, then fill final value
    min_above = np.nan_to_num(min_above, nan=len(x))
    
    # Create dataframe from peaks and troughs data
    rep_data = {"rep_start": max_below, "rep_peak": rep_peaks,  "rep_end": min_above}
    rep_df = pd.DataFrame(rep_data).astype(int)

    return rep_df

def get_rep_timing_df(rep_df, fps, concentric_first = False):
    """
    Converts rep_df to a DataFrame with timings based on fps provided

    Parameters
    ----------
    rep_df : pandas DataFrame
        DataFrame of rep timings as output by get_rep_df.
    fps : float
        The fps of the video used.
    concentric_first : bool, optional
        Set this to true if the first portion of the exercise is the concentric.
        This is the case for tricep pushdown for example but not for squats
        The default is False.

    Returns
    -------
    rep_timing_df : pandas DataFrame
        DataFrame of each rep, its eccentric and concentric.

    """
    
    
    # add a rep counter column to the rep df
    rep_df["rep"] = np.arange(1, len(rep_df)+1)
    
    # add the eccentric time for each rep
    rep_df["eccentric"] = (rep_df.rep_peak - rep_df.rep_start) / fps
    
    # and concentric
    rep_df["concentric"] = (rep_df.rep_end - rep_df.rep_peak) / fps
    
    if concentric_first:
        rep_df = rep_df.rename(columns={'eccentric': 'concentric', 'concentric': 'eccentric'})
    
    rep_timing_df = rep_df[["rep","eccentric", "concentric"]]

    return rep_timing_df
    
def intensity_check(rep_timing_df):
    """
    Returns ratio of final concentric and first concentric (as a measure of 
    intensity)
    """
    
    ratio = rep_timing_df.concentric.iloc[-1] / rep_timing_df.concentric.iloc[0]
    
    return ratio

def pretty_plot(points, fps, save=False):
    """
    Plots a nice graph of the workout
    """
    # define the time axis
    t = np.arange(len(points))
    
    # smooth out plot with lowess
    smooth_arr = sm.nonparametric.lowess(points, t, frac=0.01)
    
    # create plot
    fig, ax = plt.subplots()
    line, = ax.plot([], [], lw=2, color='#001933')
    
    # Remove bounding box
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Set font
    plt.rcParams['font.family'] = 'Impact'
    
    # Remove y-axis
    ax.get_yaxis().set_visible(False)
    
    # Set x-axis label and thick ticks
    ax.set_xlabel('Time')
    ax.tick_params(axis='x', which='both', width=2)
    
    # Define the update function for the animation
    def update(frame):
        xdata = t[:frame+1]
        ydata = smooth_arr[:,1][:frame+1]
        line.set_data(xdata, ydata)
        ax.set_xlim(0, t[-1])  # set x-axis limits
        ax.set_ylim(smooth_arr[:,1].min(), smooth_arr[:,1].max())  # set y-axis limits
        return line,
    
    # plt.plot(t, smooth_arr[:,1])
    
    # Create the animation
    ani = FuncAnimation(fig, update, frames=len(t), blit=True, interval=int(1000/fps))
    if save:
        ani.save("animation.mp4")
    plt.show()

def analyse_video(path, concentric_first = False, show_video = False, height_tolerance = 0.7, period_tolerance = 0.7, debug=False):
    
    # first generate the landmarks for each body part for each frame
    landmarks = generate_landmarks(path, show_video)
    
    # extract the points in space and visibility for each body part
    x, y, z, visibility = get_points(landmarks)
    
    # get the points of the moving visible body part from the video
    points = moving_part_points(x, y, visibility)
    
    # get the rep DataFrame which has start, peak and end of rep frames
    rep_df = get_rep_df(points, height_tolerance, period_tolerance)
    
    # get the fps of the video
    fps = get_fps(path)
    
    # convert the rep_df from frame to time
    rep_time_df = get_rep_timing_df(rep_df, fps, concentric_first)
    
    if debug:
        return rep_time_df, points
    
    return rep_time_df

#%%
path = tricep_path = 'test_footage/tricep_pushdown_test.mp4' # 7 reps
path = leg_press_path = 'test_footage/leg_press_test.mp4' # 7 reps
path = lat_pulldown_path = 'test_footage/lat_pulldown_test.mp4' # 6 reps
path = pendulum_squat_path = 'test_footage/pendulum_squat_test.mp4' # 4 reps
path = lateral_raise_path = "test_footage/lateral_raise_test.mp4" # 4 reps
path = machine_press_path = "test_footage/machine_press_test.mp4" # 7 reps
path = rdl_path = "test_footage/rdl_test.mp4" # 5 reps

#%%
# first generate the landmarks for each body part for each frame
landmarks = generate_landmarks(path, False)

# extract the points in space and visibility for each body part
x, y, z, visibility = get_points(landmarks)

# get the points of the moving visible body part from the video
points = moving_part_points(x, y, visibility)

# get the fps of the video
fps = get_fps(path)

# get the rep DataFrame which has start, peak and end of rep frames
rep_df = get_rep_df(points, fps)

# convert the rep_df from frame to time
rep_time_df = get_rep_timing_df(rep_df, fps)

print(rep_time_df)

# period leniancy should be adjusted based on fps


#%%

# define the time axis
t = np.arange(len(points))

# smooth out plot with lowess
smooth_arr = sm.nonparametric.lowess(points, t, frac=0.01)

# create plot
fig, ax = plt.subplots()
line, = ax.plot([], [], lw=2, color='#001933')

# Remove bounding box
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)

# Set font
plt.rcParams['font.family'] = 'Impact'

# Remove y-axis
ax.get_yaxis().set_visible(False)

# Set x-axis label and thick ticks
ax.set_xlabel('Time')
ax.tick_params(axis='x', which='both', width=2)

# Define the update function for the animation
def update(frame):
    xdata = t[:frame+1]
    ydata = smooth_arr[:,1][:frame+1]
    line.set_data(xdata, ydata)
    ax.set_xlim(0, t[-1])  # set x-axis limits
    ax.set_ylim(smooth_arr[:,1].min(), smooth_arr[:,1].max())  # set y-axis limits
    return line,

# plt.plot(t, smooth_arr[:,1])

# Create the animation
ani = FuncAnimation(fig, update, frames=len(t), blit=True, interval=int(1000/fps))
ani.save("animation.gif")
plt.show()