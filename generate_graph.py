import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
timeline_output_path = os.path.join(BASE_DIR, "timeline_output")

if len(sys.argv) < 2:
    print("Usage: python generate_graph.py <video_filename>")
    sys.exit()

video_filename = sys.argv[1]
csv_path = os.path.join(timeline_output_path, f"{os.path.splitext(video_filename)[0]}_timeline.csv")

if not os.path.exists(csv_path):
    print(f"Timeline CSV not found: {csv_path}")
    sys.exit()

# --------- READ CSV ----------
df = pd.read_csv(csv_path)

# Naye column names ke hisaab se
x = df['frame_number']
y = df['emotion']

# --------- PLOT GRAPH ----------
plt.figure(figsize=(15,5))
plt.plot(x, y, marker='o', linestyle='-', markersize=3)
plt.title(f"Emotion Timeline: {video_filename}")
plt.xlabel("Frame Number")
plt.ylabel("Emotion")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()

# Save figure
graph_path = os.path.join(timeline_output_path, f"{os.path.splitext(video_filename)[0]}_timeline.png")
plt.savefig(graph_path)
plt.show()

print(f"Graph saved: {graph_path}")
