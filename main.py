
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import pygame
import time
import os
import urllib.request

# ---------------- CONFIGURATION ----------------

MODEL_PATH = "hand_landmarker.task"
SOUND_FOLDER = "sounds_swaras"

SOUND_MAPPING = {
    1: "1_Sa.mp3",
    2: "2_Re.mp3",
    3: "3_Ga.mp3",
    4: "4_Ma.mp3",
    5: "5_Pa.mp3",
    6: "6_Dha.mp3",
    7: "7_Ni.mp3",
    8: "8_Sa_upper.mp3"
}

SWARA_NAMES = {
    1: "Sa",
    2: "Re",
    3: "Ga",
    4: "Ma",
    5: "Pa",
    6: "Dha",
    7: "Ni",
    8: "Sa (Upper)"
}

# ---------------- DOWNLOAD MODEL ----------------

if not os.path.exists(MODEL_PATH):
    print("Downloading MediaPipe hand model...")
    url = (
        "https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/1/"
        "hand_landmarker.task"
    )
    urllib.request.urlretrieve(url, MODEL_PATH)
    print("Model downloaded!")

# ---------------- INITIALISE AUDIO ----------------

try:
    pygame.mixer.init()
    AUDIO_AVAILABLE = True
    print("Audio initialized successfully!")
except pygame.error:
    AUDIO_AVAILABLE = False
    print("No audio device found. Running without sound.")

available_sounds = {}

if AUDIO_AVAILABLE:
    for count, filename in SOUND_MAPPING.items():
        path = os.path.join(SOUND_FOLDER, filename)

        if os.path.exists(path):
            available_sounds[count] = pygame.mixer.Sound(path)

# ---------------- CREATE HAND LANDMARKER ----------------

base_options = mp_python.BaseOptions(
    model_asset_path=MODEL_PATH
)

options = mp_vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=mp_vision.RunningMode.VIDEO,
    num_hands=2
)

landmarker = mp_vision.HandLandmarker.create_from_options(options)

# ---------------- HAND CONNECTIONS ----------------

HAND_CONNECTIONS = mp.solutions.hands.HAND_CONNECTIONS

def draw_landmarks(frame, hand_landmarks):

    h, w, _ = frame.shape

    points = [
        (int(lm.x * w), int(lm.y * h))
        for lm in hand_landmarks
    ]

    for start_idx, end_idx in HAND_CONNECTIONS:
        cv2.line(
            frame,
            points[start_idx],
            points[end_idx],
            (0, 255, 0),
            2
        )

    for x, y in points:
        cv2.circle(
            frame,
            (x, y),
            4,
            (0, 0, 255),
            -1
        )


# ---------------- COUNT FINGERS ----------------

def count_fingers(hand_landmarks, handedness):

    count = 0

    # Thumb
    thumb_tip = hand_landmarks[4]
    thumb_joint = hand_landmarks[3]

    if handedness == "Right":
        if thumb_tip.x < thumb_joint.x:
            count += 1
    else:
        if thumb_tip.x > thumb_joint.x:
            count += 1

    # Index, Middle, Ring and Pinky
    finger_pairs = [
        (8, 6),
        (12, 10),
        (16, 14),
        (20, 18)
    ]

    for tip, joint in finger_pairs:
        if hand_landmarks[tip].y < hand_landmarks[joint].y:
            count += 1

    return count


# ---------------- OPEN WEBCAM ----------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    landmarker.close()
    pygame.quit()
    raise SystemExit

start_time = time.time()

previous_count = -1

print("Hand Gesture Swara Player started!")
print("Press Q to quit.")

# ---------------- MAIN WEBCAM LOOP ----------------

try:

    while cap.isOpened():

        success, frame = cap.read()

        if not success:
            print("Could not read webcam frame.")
            break

        # Mirror webcam
        frame = cv2.flip(frame, 1)

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # Wrap frame as MediaPipe Image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Timestamp in milliseconds
        timestamp_ms = int(
            (time.time() - start_time) * 1000
        )

        # Detect hands
        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

        # Reset total count every frame
        total_count = 0

        # Process each detected hand
        for i, hand_landmarks in enumerate(result.hand_landmarks):

            label = result.handedness[i][0].category_name

            total_count += count_fingers(
                hand_landmarks,
                label
            )

            draw_landmarks(frame, hand_landmarks)

        # Keep count between 0 and 10
        total_count = min(total_count, 10)

        # Display finger count
        cv2.putText(
            frame,
            f"Fingers: {total_count}",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 0),
            2
        )

        # Play sound only when count changes
        if total_count != previous_count:

            if total_count in available_sounds:
                available_sounds[total_count].play()

            previous_count = total_count

        # Display swara name
        if total_count in SWARA_NAMES:
            swara = SWARA_NAMES[total_count]
        else:
            swara = "Silence"

        cv2.putText(
            frame,
            f"Swara: {swara}",
            (30, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        # Display webcam
        cv2.imshow(
            "Hand Gesture Swara Player",
            frame
        )

        # Quit on Q
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()
    pygame.mixer.quit()

    print("Program closed successfully!")
