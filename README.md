# auto-set
auto-set is a Python library that analyzes workout videos and returns a DataFrame of rep time information.


https://user-images.githubusercontent.com/48531291/231286382-23db1e1a-f254-495e-810d-65a7e9069f4a.mp4


## Usage
`
from auto-set import analyze_video

path = 'path/to/workout.mp4'
result = auto_set.analyze_video(path)

`

The function will return a pandas DataFrame with the columns:
- "**rep**": The rep number
- "**eccentric**": The time taken for the eccentric portion of the rep
- "**concentric**": The time taken for the concentric portion of the rep

You can also pass optional arguments to "**analyze_video()**" such as 
- "**concentric_first**": Set to true if the first portion of the exercise is the concentric like for lat pulldowns but not for squats
- "**show_video**": Set to true if you want to see the pose estimation overlayed on video of exercise


## Example
To obtain the rep timing for the lat-pulldown video above:

`
from auto-set import analyze_video

path = 'path/to/lat_pulldown_video.mp4'
result = auto_set.analyze_video(path, concentric_first = True, show_video = True)

`

which returns the DataFrame

`
   rep  eccentric  concentric
0    1   0.650597    0.784053
1    2   0.784053    0.717325
2    3   0.817417    0.750689
3    4   0.917508    0.734007
4    5   0.934190    0.750689
5    6   1.301194    1.334558
`

## Dependencies
Main dependencies are OpenCV, mediapipe, pandas, numpy, scipy, statsmodels, matplotlib

## Credits
auto-set was made by Tanjim Chowdhury.
