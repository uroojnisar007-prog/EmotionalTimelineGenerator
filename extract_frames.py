import cv2
import os

dataset_path = r"C:\Users\Urooj\Desktop\EmotionTimelineGenerator\dataset\Video_Dataset\Video_Dataset"
frames_path = r"C:\Users\Urooj\Desktop\EmotionTimelineGenerator\dataset\frames"

if not os.path.exists(frames_path):
    os.makedirs(frames_path)

emotion_map = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.lower().endswith('.mp4'):

            # Emotion from Actor folder
            actor_folder = os.path.basename(root)   # e.g. "Actor_04"
            actor_id = actor_folder.split('_')[-1]  # e.g. "04"
            emotion_code = actor_id.zfill(2)
            emotion = emotion_map.get(emotion_code, "unknown")

            # Create folder
            video_name = os.path.splitext(file)[0]
            video_frames_folder = os.path.join(frames_path, emotion, video_name)
            os.makedirs(video_frames_folder, exist_ok=True)

            # Extract frames
            cap = cv2.VideoCapture(os.path.join(root, file))
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                cv2.imwrite(os.path.join(video_frames_folder, f"frame_{frame_count:03d}.jpg"), frame)
                frame_count += 1

            print(f"Extracted {frame_count} frames -> {emotion} ({file})")
