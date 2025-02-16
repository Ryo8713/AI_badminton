from moviepy.editor import VideoFileClip
from pathlib import Path
from datetime import datetime

# Define input and output folders
video_folder = "C:\Match_video"  # Replace with the actual folder containing videos
timeline_folder = "final_result"  # Replace with the actual folder containing timeline files
output_folder = "rally_video"  # Replace with the actual folder to save rally videos

# Ensure the output folder exists
Path(output_folder).mkdir(parents=True, exist_ok=True)

# List of match video filenames (replace with actual filenames)
match_videos = [f"match{i}" for i in range(1, 29)]  # Assuming videos are named match1.mp4, match2.mp4, ..., match29.mp4

# Function to convert timestamp to seconds
def timestamp_to_seconds(timestamp):
    time_obj = datetime.strptime(timestamp, "%H:%M:%S.%f")
    return time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second + time_obj.microsecond / 1e6

# Process each match video
for match_video in match_videos:
    video_path = f"{video_folder}/{match_video}/{match_video}.mp4"
    timeline_file = f"{video_folder}/{match_video}/{match_video}.txt"  # Assuming timeline files are named match1.txt, match2.txt, ..., match29.txt

    # Load video
    video = VideoFileClip(video_path)

    # Read rally timestamps
    with open(timeline_file, 'r', encoding='utf-8') as f:
        timestamps = f.readlines()

    # Ensure timestamps are in pairs (start, end)
    if len(timestamps) % 2 != 0:
        print(f"Warning: Odd number of timestamps in {timeline_file}. Ignoring last unpaired timestamp.")
        timestamps = timestamps[:-1]

    # Process each pair of timestamps
    for i in range(0, len(timestamps), 2):
        start_time = timestamp_to_seconds(timestamps[i].strip())
        end_time = timestamp_to_seconds(timestamps[i + 1].strip())
        rally_clip = video.subclip(start_time, end_time)
        rally_clip.write_videofile(f"{output_folder}/{match_video}_rally_{i//2 + 1}.mp4", codec="libx264")

    print(f"Processed {match_video}")