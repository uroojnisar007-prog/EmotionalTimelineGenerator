import pandas as pd

# Load predicted CSV
pred_df = pd.read_csv('timeline_output/girl_timeline.csv')
predicted = pred_df['emotion'].tolist()

# Load ground truth CSV
gt_df = pd.read_csv('timeline_output/girl_ground_truth.csv')
ground_truth = gt_df['emotion'].tolist()

# Check frame count
if len(predicted) != len(ground_truth):
    print("Warning: Frame count mismatch!")
else:
    correct = sum([p == g for p, g in zip(predicted, ground_truth)])
    total = len(ground_truth)
    accuracy = correct / total
    print(f"Accuracy: {accuracy*100:.2f}%")
