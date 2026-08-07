import cv2
import mediapipe as mp
import time

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
# PHOTO
# ============================================================

photo_path = Path(
    "/home/yen/Desktop/yearn/the man who can't be moved.jpg"
)

photo = cv2.imread(str(photo_path))

if photo is None:
    print("ERROR: Could not load photo.")
    print(photo_path)
    exit()

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

# How long the person must remain on the left
REQUIRED_TIME = 0.5

# How long the photo remains visible
PHOTO_DURATION = 3


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Mirror camera
    frame = cv2.flip(frame, 1)

    # Current time
    current_time = time.time()

    # Convert camera frame to RGB
    frame_rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # ========================================================
    # ONLY DETECT POSE WHILE PHOTO IS NOT VISIBLE
    # ========================================================

    if not photo_visible:

        result = pose.process(frame_rgb)

        person_on_left = False


        # ====================================================
        # CHECK IF A PERSON WAS DETECTED
        # ====================================================

        if result.pose_landmarks:

            landmarks = result.pose_landmarks.landmark


            # ------------------------------------------------
            # Upper body landmarks
            # ------------------------------------------------

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


            # ------------------------------------------------
            # Calculate upper-body center
            # ------------------------------------------------

            center_x = (
                left_shoulder.x +
                right_shoulder.x +
                left_hip.x +
                right_hip.x
            ) / 4


            # ------------------------------------------------
            # Check left side
            # ------------------------------------------------

            if center_x < 0.50:

                person_on_left = True


            # ------------------------------------------------
            # Draw pose
            # ------------------------------------------------

            mp_draw.draw_landmarks(
                frame,
                result.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )


        # ====================================================
        # POSITION TIMER
        # ====================================================

        if person_on_left:

            if left_start_time is None:

                left_start_time = current_time

            else:

                time_on_left = (
                    current_time -
                    left_start_time
                )

                # ============================================
                # TRIGGER PHOTO
                # ============================================

                if time_on_left >= REQUIRED_TIME:

                    photo_visible = True

                    photo_start_time = current_time

                    left_start_time = None


        else:

            # Person left the trigger area
            left_start_time = None


    # ========================================================
    # SHOW PHOTO
    # ========================================================

    if photo_visible:

        # Keep the original aspect ratio

        frame_height, frame_width = frame.shape[:2]

        photo_height, photo_width = photo.shape[:2]


        # Scale photo to fit the camera window

        scale = min(
            frame_width / photo_width,
            frame_height / photo_height
        )

        new_width = int(photo_width * scale)
        new_height = int(photo_height * scale)


        resized_photo = cv2.resize(
            photo,
            (new_width, new_height)
        )


        # Center the photo

        x = (
            frame_width -
            new_width
        ) // 2

        y = (
            frame_height -
            new_height
        ) // 2


        # Put photo onto camera frame

        frame[
            y:y + new_height,
            x:x + new_width
        ] = resized_photo


        # ====================================================
        # PHOTO TIMER
        # ====================================================

        if (
            current_time -
            photo_start_time
        ) >= PHOTO_DURATION:

            photo_visible = False


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

cap.release()
cv2.destroyAllWindows()
pose.close()