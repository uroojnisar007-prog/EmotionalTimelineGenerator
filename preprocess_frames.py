import os
import cv2

# INPUT: frames (extracted)
input_path = r"C:\Users\Urooj\Desktop\EmotionTimelineGenerator\dataset\frames"

# OUTPUT: resized frames
output_path = r"C:\Users\Urooj\Desktop\EmotionTimelineGenerator\dataset\processed_frames"

img_size = 224  # you can change 112 or 224 based on your model

def preprocess_images():
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    actors = os.listdir(input_path)
    print("Actors found:", actors)

    for actor in actors:
        actor_folder = os.path.join(input_path, actor)
        if not os.path.isdir(actor_folder):
            continue

        # Create same folder in processed output
        output_actor_folder = os.path.join(output_path, actor)
        os.makedirs(output_actor_folder, exist_ok=True)

        for img_name in os.listdir(actor_folder):
            img_path = os.path.join(actor_folder, img_name)

            # Read image
            img = cv2.imread(img_path)
            if img is None:
                print("Skipping corrupted:", img_path)
                continue

            # Resize
            img = cv2.resize(img, (img_size, img_size))

            # Save processed image
            save_path = os.path.join(output_actor_folder, img_name)
            cv2.imwrite(save_path, img)

        print(f"Processed actor: {actor}")

    print("\n✅ All frames preprocessed successfully!")

if __name__ == "__main__":
    preprocess_images()
