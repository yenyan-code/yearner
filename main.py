import cv2
import mediapipe as mp
import time
import pygame

from pathlib import Path


# ============================================================
# MEDIAPIPE POSE
# ============================================================

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

# ============================================================
# PHOTO
# ============================================================

photo_path = BASE_DIR / "assets" / "trigger_photo.jpg"

photo = cv2.imread(str(photo_path))

if photo is None:
    print("ERROR: Could not load photo:")
    print(photo_path)
    exit()


# ============================================================
# MUSIC
# ============================================================

music_path = BASE_DIR / "assets" / "part_1.mp3"

if not music_path.exists():
    print("ERROR: Could not find music:")
    print(music_path)
    exit()


# Initialize pygame audio
pygame.mixer.init()

photo_visible = False


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open camera.")
    exit()


# ============================================================
# VARIABLES
# ============================================================

left_start_time = None
photo_start_time = 0

# Person must remain on the left for this long
REQUIRED_TIME = 3.0

# Maximum time the photo can remain visible
PHOTO_DURATION = 40


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Mirror camera
    frame = cv2.flip(frame, 1)

    current_time = time.time()

    # Convert BGR → RGB
    frame_rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # ========================================================
    # ALWAYS DETECT PERSON
    # ========================================================

    result = pose.process(frame_rgb)

    person_on_left = False
    person_detected = False


    if result.pose_landmarks:

        person_detected = True

        landmarks = result.pose_landmarks.landmark


        # ----------------------------------------------------
        # Upper body landmarks
        # ----------------------------------------------------

        left_shoulder = landmarks[
            mp_pose.PoseLandmark.LEFT_SHOULDER
        ]

        right_shoulder = landmarks[
            mp_pose.PoseLandmark.RIGHT_SHOULDER
        ]

        left_hip = landmarks[
            mp_pose.PoseLandmark.LEFT_HIP
        ]

        right_hip = landmarks[
            mp_pose.PoseLandmark.RIGHT_HIP
        ]


        # ----------------------------------------------------
        # Calculate upper-body center
        # ----------------------------------------------------

        center_x = (
            left_shoulder.x +
            right_shoulder.x +
            left_hip.x +
            right_hip.x
        ) / 4


        # ----------------------------------------------------
        # Detect LEFT side
        # ----------------------------------------------------

        if center_x < 0.50:

            person_on_left = True


    # ========================================================
    # PHOTO IS NOT CURRENTLY VISIBLE
    # ========================================================

    if not photo_visible:

        if person_on_left:

            # Start timer when entering left side
            if left_start_time is None:

                left_start_time = current_time

            else:

                time_on_left = (
                    current_time -
                    left_start_time
                )


                # ============================================
                # TRIGGER PHOTO + MUSIC
                # ============================================

                if time_on_left >= REQUIRED_TIME:

                    photo_visible = True

                    photo_start_time = current_time

                    left_start_time = None


                    # Start music
                    pygame.mixer.music.load(
                        str(music_path)
                    )

                    pygame.mixer.music.play()

        else:

            # Person is not on left
            left_start_time = None


    # ========================================================
    # PHOTO IS CURRENTLY VISIBLE
    # ========================================================

    else:

        # ----------------------------------------------------
        # IMPORTANT:
        # If the person moves RIGHT, hide the photo
        # ----------------------------------------------------

        if person_detected and not person_on_left:

            photo_visible = False

            left_start_time = None

            # Stop music
            pygame.mixer.music.stop()


        # ----------------------------------------------------
        # Maximum photo duration
        # ----------------------------------------------------

        elif (
            current_time -
            photo_start_time
        ) >= PHOTO_DURATION:

            photo_visible = False

            # Stop music
            pygame.mixer.music.stop()


        # ========================================================
        # SHOW PHOTO
        # ========================================================

        if photo_visible:

            frame_height, frame_width = frame.shape[:2]

            photo_height, photo_width = photo.shape[:2]

            # ----------------------------------------------------
            # Resize photo so it completely covers the camera
            # ----------------------------------------------------

            scale = max(
                frame_width / photo_width,
                frame_height / photo_height
            )

            new_width = int(photo_width * scale)
            new_height = int(photo_height * scale)

            resized_photo = cv2.resize(
                photo,
                (new_width, new_height)
            )

            # ----------------------------------------------------
            # Crop the excess parts
            # ----------------------------------------------------

            crop_x = (
                new_width - frame_width
            ) // 2

            crop_y = (
                new_height - frame_height
            ) // 2

            cropped_photo = resized_photo[
                crop_y:crop_y + frame_height,
                crop_x:crop_x + frame_width
            ]

            # ----------------------------------------------------
            # Make photo exactly the same size as camera
            # ----------------------------------------------------

            frame = cropped_photo.copy()

    # ========================================================
    # DISPLAY CAMERA
    # ========================================================

    cv2.imshow(
        "Camera",
        frame
    )


    # ESC = EXIT
    if cv2.waitKey(1) & 0xFF == 27:
        break


# ============================================================
# CLEANUP
# ============================================================

pygame.mixer.music.stop()
pygame.mixer.quit()

cap.release()
cv2.destroyAllWindows()
pose.close()
