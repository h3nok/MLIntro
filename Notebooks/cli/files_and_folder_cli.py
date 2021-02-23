import os
import shutil

p = r"E:\viNet_RnD\Datasets\Feedback\Vat\White-Tailed-Eagle"
latest = os.path.join(p, 'Reviewed_post_9_28')
if not os.path.exists(latest):
    os.makedirs(latest)
frames = [frame for frame in os.listdir(p) if '2020.01' in frame or '2020.02' in frame or '2020.03' in frame]
for frame in frames:
    shutil.move(os.path.join(p, frame), os.path.join(latest, frame))
print(len(frames))
