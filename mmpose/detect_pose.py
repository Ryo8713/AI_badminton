import cv2
import csv
import os
import argparse
import numpy as np
from mmpose.apis import MMPoseInferencer
from mayavi import mlab
from tqdm import tqdm
from pathlib import Path


def normalize_joints(
    arr: np.ndarray,
    bbox: np.ndarray,
    v_height=None,
    center_align=False,
):
    '''
    - `arr`: (m, J, 2), m=2.
    - `bbox`: (m, 4), m=2.
    
    Output: (m, J, 2), m=2.
    '''
    # If v_height == None and center_align == False,
    # this normalization method is same as that used in TemPose.
    if v_height:
        dist = v_height / 4
    else:  # bbox diagonal dist
        dist = np.linalg.norm(bbox[:, 2:] - bbox[:, :2], axis=-1, keepdims=True)
    
    arr_x = arr[:, :, 0]
    arr_y = arr[:, :, 1]
    x_normalized = np.where(arr_x != 0.0, (arr_x - bbox[:, None, 0]) / dist, 0.0)
    y_normalized = np.where(arr_y != 0.0, (arr_y - bbox[:, None, 1]) / dist, 0.0)

    if center_align:
        center = (bbox[:, :2] + bbox[:, 2:]) / 2
        c_normalized = (center - bbox[:, :2]) / dist
        x_normalized -= c_normalized[:, None, 0]
        y_normalized -= c_normalized[:, None, 1]

    return np.stack((x_normalized, y_normalized), axis=-1)


def demo_human_2d_and_3d(img_path):
    inferencer_2d = MMPoseInferencer('human')
    inferencer_3d = MMPoseInferencer(pose3d='human3d')
    result_generator_2d = inferencer_2d(img_path, show=False)
    result_generator_3d = inferencer_3d(img_path, show=False)
    for result_2d, result_3d in zip(result_generator_2d, result_generator_3d):
        for e_2d, e_3d in zip(result_2d['predictions'][0],
                              result_3d['predictions'][0]):  # batch_size=1 (default)
            
            ## 2d
            keypoints_2d = np.array(e_2d['keypoints'])[None, :]
            bbox = np.concatenate([
                keypoints_2d.min(1),
                keypoints_2d.max(1)
            ], axis=-1)
            
            keypoints_2d_normalized = normalize_joints(
                keypoints_2d, bbox,
                center_align=True
            )[0]

            coords = np.concatenate([
                keypoints_2d_normalized[:, 0:1],
                keypoints_2d_normalized[:, 1:],
                np.zeros((len(e_2d['keypoints']), 1)),
            ], axis=1)
            
            # 創建一個 3D 圖形
            fig = mlab.figure(figure='My Figure')

            # 繪製散點圖
            pts = mlab.points3d(coords[:, 0], coords[:, 1], coords[:, 2], scale_factor=0.03)

            # 為每個點添加文字標籤
            for i, (x, y, z) in enumerate(coords):
                mlab.text3d(x, y, z, str(i), scale=0.02)

            # 顯示圖形
            mlab.show(stop=True)

            ## 3d
            coords = np.array(e_3d['keypoints'])

            # x
            print('0 -> 1 :', coords[1] - coords[0])
            print('0 -> 4 :', coords[4] - coords[0])

            # y
            print('12 -> 13 :', coords[13] - coords[12])

            # z
            print('0 -> 7 :', coords[7] - coords[0])

            # 創建一個 3D 圖形
            fig = mlab.figure(figure='My Figure')

            # 繪製散點圖
            pts = mlab.points3d(coords[:, 0], coords[:, 1], coords[:, 2], scale_factor=0.1)

            # 為每個點添加文字標籤
            for i, (x, y, z) in enumerate(coords):
                mlab.text3d(x, y, z, str(i), scale=0.1)

            # 顯示圖形
            mlab.show(stop=True)


def demo_human_2d(img_path):
    inferencer = MMPoseInferencer('human')
    x = []
    y = []
    result_generator = inferencer(img_path, show=False)
    for result in result_generator:
        for e in result['predictions'][0]:  # batch_size=1 (default)
            keypoints_2d = np.array(e['keypoints'])[None, :]
            bbox = np.concatenate([
                keypoints_2d.min(1),
                keypoints_2d.max(1)
            ], axis=-1)
            
            keypoints_2d_normalized = normalize_joints(
                keypoints_2d, bbox,
                center_align=True
            )[0]

            coords = np.concatenate([
                keypoints_2d_normalized[:, 0:1],
                keypoints_2d_normalized[:, 1:],
                np.zeros((len(e['keypoints']), 1)),
            ], axis=1)
            
            # 創建一個 3D 圖形
            fig = mlab.figure(figure='My Figure')

            # 繪製散點圖
            pts = mlab.points3d(coords[:, 0], coords[:, 1], coords[:, 2], scale_factor=0.03)

            # 為每個點添加文字標籤
            for i, (x, y, z) in enumerate(coords):
                mlab.text3d(x, y, z, str(i), scale=0.02)

            # 顯示圖形
            mlab.show(stop=True)


def demo_human_3d(img_path):
    inferencer = MMPoseInferencer(pose3d='human3d')
    result_generator = inferencer(img_path, show=False)

    for result in result_generator:
        for e in result['predictions'][0]:  # batch_size=1 (default)
            coords = np.array(e['keypoints'])

            # x
            print('0 -> 1 :', coords[1] - coords[0])
            print('0 -> 4 :', coords[4] - coords[0])

            # y
            print('12 -> 13 :', coords[13] - coords[12])

            # z
            print('0 -> 7 :', coords[7] - coords[0])

            # 創建一個 3D 圖形
            fig = mlab.figure(figure='My Figure')

            # 繪製散點圖
            pts = mlab.points3d(coords[:, 0], coords[:, 1], coords[:, 2], scale_factor=0.05)

            # 為每個點添加文字標籤
            for i, (x, y, z) in enumerate(coords):
                mlab.text3d(x, y, z, str(i), scale=0.05)

            # 顯示圖形
            mlab.show(stop=True)


def test_bug(inferencer, p):
    result_generator = inferencer(str(p), show=False)
    for result in result_generator:
        pass


def no_bug(p):
    inferencer = MMPoseInferencer(pose3d='human3d')
    result_generator = inferencer(str(p), show=False)
    for result in result_generator:
        pass

def save_player_keypoints(player_data, output_path):
    with open(output_path, mode='w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        headers = ['frame']
        for i in range(15):  # Assuming 15 keypoints
            headers += [f'x{i}', f'y{i}']  # x, y, confidence
        csv_writer.writerow(headers)
        
        for frame_data in player_data:
            frame_idx = frame_data['frame']
            keypoints = frame_data['keypoints']
            row = [frame_idx]
            for kp in keypoints:
                row.extend(kp)  # Append x, y, and confidence
            csv_writer.writerow(row)

    print(f"Keypoints saved to {output_path}")

def visualize_video_estimated(in_path, out_dir='vis_out', csv_output_dir='pose_data.csv', frame_width=1280):
    inferencer = MMPoseInferencer('human')
    result_generator = inferencer(in_path, vis_out_dir=out_dir)

    video_name = os.path.splitext(os.path.basename(in_path))[0]

    os.makedirs(csv_output_dir, exist_ok=True)
    output_video_path = os.path.join(csv_output_dir, f"{video_name}_visualized_output.mp4")

    #Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Set up video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # Data for CSVs
    bottom_player_data = []
    top_player_data = []

    # Process each frame
    for frame_idx, result in enumerate(result_generator):
        ret, frame = cap.read()
        predictions = result['predictions'][0]
        if len(predictions) < 2:
            print(f"Frame {frame_idx}: Less than two players detected.")
            continue
        players = []
        for pred in predictions:  
            bbox = pred['bbox']
            keypoints = pred['keypoints']

            # Calculate bbox center
            center_x = (bbox[0][0] + bbox[0][2]) / 2  # (x_min + x_max) / 2
            center_y = (bbox[0][1] + bbox[0][3]) / 2  # (y_min + y_max) / 2
            
            players.append({'center_x': center_x, 'center_y': center_y, 'bbox': bbox, 'keypoints': keypoints})
        
        # Sort players by x-axis (center_x)
        players = sorted(players, key=lambda p: p['center_x'])

        # Remove the 4 players with smallest x-axis values and largest x-axis value
        players = players[4:-1]

        # Sort by y-axis and select top and bottom players
        players = sorted(players, key=lambda p: p['center_y'], reverse=True)
        bottom_player = players[0]
        top_player = players[1]

        # Save keypoints to respective lists
        bottom_player_data.append({'frame': frame_idx, 'keypoints': bottom_player['keypoints']})
        top_player_data.append({'frame': frame_idx, 'keypoints': top_player['keypoints']})
        # Draw bounding boxes and keypoints
        for player, color, label in [(top_player, (0, 255, 0), "Top Player"), (bottom_player, (0, 0, 255), "Bottom Player")]:
            bbox = player['bbox']
            keypoints = player['keypoints']

            # Draw keypoints
            for kp in keypoints:
                x, y= map(int, kp)
                cv2.circle(frame, (x, y), 5, color, -1)

        # Write frame to output video
        out_video.write(frame)

    # Release resources
    cap.release()
    out_video.release()
    print(f"Visualized video saved to {output_video_path}")

    bottom_file = os.path.join(csv_output_dir, f"{video_name}_bottom.csv")
    top_file = os.path.join(csv_output_dir, f"{video_name}_top.csv")

    save_player_keypoints(bottom_player_data, bottom_file)
    save_player_keypoints(top_player_data, top_file)  

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pose estimation and visualization")
    parser.add_argument('--video_path', type=str, required=True, help="Path to the input video file")
    parser.add_argument('--csv_output_dir', type=str, required=True, help="Path to the output CSV file")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
    else:
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        visualize_video_estimated(
            in_path=args.video_path,
            out_dir='vis_out',
            csv_output_dir = args.csv_output_dir,
            frame_width=frame_width
        )
    cap.release()
