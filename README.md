# AI_badminton
NTHU CS project

## Introduction
AI_badminton is a Python project for automated processing of badminton videos. This project includes tools for court and pose processing, homography for court line detection, model loaders for hit detection, pose tracking, shot tracking, and pipelines for processing video data.

## Installation
To set up the project, follow these steps:

1. **Install needed packages for mmpose and TrackNet-main**:

   - Navigate to the `mmpose` directory and create a virtual environment. And install the required packages:
     ```sh
     cd mmpose
     python -m venv myenv
     pip install -r requirements.txt
     ```
  - Navigate to the `TrackNetV3-main` directory and create a virtual environment. And install the required packages:
     ```sh
     cd TrackNetV3-main
     python -m venv env
     pip install -r requirements.txt
     ```
   - Navigate to the `setup` directory and create a virtual environment. And install the required packages:
     ```sh
     cd ../setup
     python -m venv myenv
     pip install -r requirements.txt
     ```

2. **Activate the virtual environment**:

   - Activate the virtual environment in the setup directory:

3. **Execute the setup script**:
   - Ensure you are in the `setup` directory and run the setup script:
     ```sh
     python setup.py
     ```

## Usage
Coming soon!

## Features
- Classes for Court and Pose processing
- Homography code for identifying court lines from 4 points
- Model loaders / wrappers for hit detection, pose tracking, and shot tracking
- Pipelines for processing pose / shot / hit data for input videos
- A framework and UI for shot searching

## Contributing
Contributions are welcome! Please follow the [contributing guidelines](CONTRIBUTING.md).

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
