# Yearner | Photo Position Detector

## Description

This repository contains the source code for the **Yearner** project. The goal of the project is to use real-time human pose detection through a webcam to create an interactive photo and audio experience.

The application detects the position of a person's upper body using **MediaPipe Pose**. When the person moves to the **left side of the camera**, a photograph is automatically displayed and a music track begins playing.

When the person moves back to the **right side**, the photograph disappears and the music stops.

The project combines computer vision, real-time pose detection, image processing, and audio playback using Python.

## Important Note About the Python Version

The project **MUST be executed using Python 3.10**, as this is the version used to create and test the project's environment and dependencies.

### Python Requirements

- **Python 3.10.x (required)**
- Python 3.11+ is not recommended
- Python 3.9 or earlier is not recommended

For consistency and compatibility, Python 3.10 should be used when creating the virtual environment.

## Technologies Used

The project uses the following libraries:

- **Python 3.10**
- **OpenCV** – webcam access, image processing, and displaying the camera
- **MediaPipe** – real-time human pose detection
- **Pygame** – music playback
- **uv** – Python version and virtual environment management

## How It Works

The application follows this process:

1. Open the computer's webcam.
2. Capture the video stream in real time.
3. Detect the user's body using MediaPipe Pose.
4. Calculate the approximate position of the user's upper body.
5. Detect when the user moves to the left side.
6. Display the configured photograph.
7. Start playing the configured music.
8. Detect when the user moves back to the right side.
9. Hide the photograph and stop the music.

## Example of Use

Using the **uv** project manager:

```bash
# Install Python 3.10 if it is not already installed
uv python install 3.10

# Create a virtual environment using Python 3.10
uv venv --python 3.10

# Activate the virtual environment
source .venv/bin/activate
# Linux

.venv\Scripts\activate
# Windows PowerShell

# Install dependencies
uv pip install mediapipe opencv-contrib-python pygame

# Run the project
uv run main.py
