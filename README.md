# Yearner | Photo Position Detector

An interactive computer vision project that uses a webcam and MediaPipe Pose to detect a person's upper-body position.

When the user moves to the **left side of the camera**, the system automatically displays a photograph and plays a selected music track. When the user moves back to the **right side**, the photograph disappears and the music stops.

The project is designed as a simple interactive camera installation using Python, OpenCV, MediaPipe, and Pygame.

---

## Features

- Real-time webcam input
- Upper-body detection using MediaPipe Pose
- Automatic left-side position detection
- Automatic photo triggering
- Background music playback
- Photo and music synchronization
- Photo automatically fills the camera window
- Photo disappears when the user moves to the right
- Music stops when the photo disappears
- Configurable trigger and display durations
- Runs locally without requiring an internet connection

---

## How It Works

The application continuously reads frames from the webcam and processes them using MediaPipe Pose.

The system calculates the approximate center position of the user's upper body using the following landmarks:

- Left shoulder
- Right shoulder
- Left hip
- Right hip

The calculated position is used to determine whether the user is standing on the left side of the camera.

### Interaction Flow

```text
                 START
                   |
                   v
             Open Camera
                   |
                   v
          Detect Upper Body
                   |
                   v
       Is the person on the LEFT?
              /          \
            YES           NO
             |             |
             v             v
      Start Trigger      Keep Camera
         Timer             Active
             |
             v
      Trigger Condition
         Satisfied
             |
             v
     Display Photograph
             +
        Play Music
             |
             v
       User moves RIGHT
             |
             v
     Hide Photograph
             +
        Stop Music
             |
             v
        Camera Resumes
