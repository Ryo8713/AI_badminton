import os
import sys
import traceback
from subprocess import run, CalledProcessError, DEVNULL
from multiprocessing import Pool
from tqdm import tqdm

detect_pose_script = r"C:\Badminton\Hit_detection\mmpose\detect_pose.py"
court_detection_bin = r"C:\Users\User\source\repos\Project1\x64\Debug\Project1.exe"
prdict_script = r"C:\Badminton\Hit_detection\TrackNetV3-main\predict.py"
show_trajectory_script = r"C:\Badminton\Hit_detection\TrackNetV3-main\show_trajectory.py"

# Virtual environment paths
MMPose_ENV_WINDOWS = r"C:\Badminton\Hit_detection\mmpose\myenv\Scripts\Activate.ps1"
TrackNet_ENV_WINDOWS = r"C:\Badminton\Hit_detection\TrackNetV3-main\env\Scripts\Activate.ps1"  # Change path as needed

# Activation commands
activate_mmpose_cmd = f'powershell -ExecutionPolicy Bypass "& {MMPose_ENV_WINDOWS};"'
activate_tracknet_cmd = f'powershell -ExecutionPolicy Bypass "& {TrackNet_ENV_WINDOWS};"'



VENV_PATH_LINUX = "~/venvs/mmpose_env/bin/activate"
VENV_PATH_WINDOWS = r"C:\Badminton\Hit_detection\mmpose\myenv\Scripts\Activate.ps1"

# Detect operating system and set the correct activation command
if os.name == "nt":  # Windows
    activate_cmd = f'powershell -ExecutionPolicy Bypass "& \"{VENV_PATH_WINDOWS}\"; "'
else:  # Linux/macOS
    activate_cmd = f"source {VENV_PATH_LINUX} && "

# Match data directory
data_dir = r"C:\Match_video"
matches = [f'match{i}' for i in range(1, 27)] + [f'test_match{i}' for i in range(1, 4)]




def process_match(match):
    """ Process a single match by running court detection and pose estimation on its rally videos """
    match_path = os.path.join(data_dir, match, "rally_video")
    court_output_dir = os.path.join(data_dir, match, "court")
    court_img_dir = os.path.join(data_dir, match, "court_images")
    pose_output_dir = os.path.join(data_dir, match, "poses")

    # Ensure output directories exist
    os.makedirs(court_output_dir, exist_ok=True)
    os.makedirs(court_img_dir, exist_ok=True)
    os.makedirs(pose_output_dir, exist_ok=True)

    # Check if the match folder exists
    if not os.path.exists(match_path):
        print(f"Warning: {match_path} does not exist. Skipping {match}.")
        return []

    # Collect rally videos
    rally_videos = [f for f in os.listdir(match_path) if f.endswith(('.mp4'))]

    if not rally_videos:
        print(f"Warning: No video files found in {match_path}. Skipping {match}.")
        return []

    return [(match, os.path.splitext(video)[0]) for video in rally_videos]

def create_court_detection_cmd(video_path, output_txt, output_img):
    """ Create command to run court detection """
    return f'"{court_detection_bin}" "{video_path}" "{output_txt}" "{output_img}"'


def create_pose_estimation_cmd(video_path, csv_output_dir):
    """ Create command to run MMPose inside the virtual environment """
    return f'{activate_mmpose_cmd} python "{detect_pose_script}" --video_path "{video_path}" --csv_output_dir "{csv_output_dir}"'


def create_tracknet_cmd(video_path, model_file, output_dir):
    """ Create command to run TrackNet for shuttlecock tracking """
    return f'{activate_tracknet_cmd} python "{prdict_script}" --video_file "{video_path}" --model_file "{model_file}" --save_dir "{output_dir}"'

def create_tracknet_video_cmd(video_path, csv_file, output_dir):
    """ Create command to produce tracjectory video"""
    return f'{activate_tracknet_cmd} python "{show_trajectory_script}" --video_file "{video_path}" --csv_file "{csv_file}" --save_dir "{output_dir}"'

def mapper(args):
    """ Run both court detection and pose estimation for a given rally video """
    match, rally = args
    video_path = os.path.join(data_dir, match, "rally_video", f"{rally}.mp4")

    output_txt_path = os.path.join(data_dir, match, "court", f"{rally}.txt")
    output_image_path = os.path.join(data_dir, match, "court_images", f"{rally}.png")
    csv_output_dir = os.path.join(data_dir, match, "poses")
    tracknet_output_dir = os.path.join(data_dir, match, "TrackNet")
    model_file = r"C:\Badminton\Hit_detection\TrackNetV3-main\exp\model_best.pt"  # Adjust as needed
    
    os.makedirs(csv_output_dir, exist_ok=True)
    os.makedirs(tracknet_output_dir, exist_ok=True)

    try:
        print(f"Processing {match}/{rally}...")

        # Run Court Detection
        court_cmd = create_court_detection_cmd(video_path, output_txt_path, output_image_path)
        print(f"Running court detection: {court_cmd}")
        run(court_cmd, shell=True, check=True)

        # Run Pose Estimation (MMPose)
        pose_cmd = create_pose_estimation_cmd(video_path, csv_output_dir)
        print(f"Running pose estimation: {pose_cmd}")
        run(pose_cmd, shell=True, check=True)

        # Run TrackNet (Shuttlecock Tracking)
        tracknet_cmd = create_tracknet_cmd(video_path, model_file, tracknet_output_dir)
        trajectory_cmd = create_tracknet_video_cmd(video_path, os.path.join(tracknet_output_dir, f"{rally}.csv"), tracknet_output_dir)
        print(f"Running TrackNet: {tracknet_cmd}")
        run(tracknet_cmd, shell=True, check=True)
        run(trajectory_cmd, shell=True, check=True)

        print(f"Finished processing {match}/{rally}")

    except CalledProcessError as e:
        print(f"Error processing {video_path}: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    all_videos = []

    # Collect all videos across matches
    for match in tqdm(matches, desc="Collecting videos"):
        all_videos.extend(process_match(match))

    if not all_videos:
        print("No videos found for processing.")
        sys.exit(1)

    # Process videos in parallel with a progress bar
    with Pool(4) as pool:
        list(tqdm(pool.imap_unordered(mapper, all_videos), total=len(all_videos), desc="Processing videos"))

    print("All matches processed successfully!")
