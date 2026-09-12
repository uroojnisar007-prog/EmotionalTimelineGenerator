import cv2
import csv
import os

video_path = "videos/girl.mp4"  # apna video path
output_csv = "timeline_output/girl_ground_truth.csv"

cap = cv2.VideoCapture(video_path)
frame_number = 0

# CSV create karenge
with open(output_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["frame_number", "emotion"])

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(0)  # 0 means wait until key is pressed

        # Manual label input
        # Example: 'happy', 'sad', 'angry', etc.
        label = input(f"Frame {frame_number}: Enter emotion -> ")

        # Write to CSV
        writer.writerow([frame_number, label])

        frame_number += 1

cap.release()
cv2.destroyAllWindows()
print(f"Ground truth CSV saved at {output_csv}")
