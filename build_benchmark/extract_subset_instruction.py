import json
import random
import numpy as np

def setup_seeds():
    seed = 0
    random.seed(seed)
    np.random.seed(seed)

subset_size = 50
ANNOTATIONS_FILE_PATH = "/home/pirenjie/data/coco/annotations/train_2017_task_evaluation.json"
output_file = "/home/pirenjie/data/coco/annotations/train_2017_task_evaluation_subset50.json"
with open(ANNOTATIONS_FILE_PATH, 'r')  as f:
    full_file = json.load(f)

tasks = ["task1", "task2", "task3"]
images_new = random.sample(full_file['images'], subset_size)
img_ids_subset = [img['id'] for img in images_new]

annos_new = {task : [anno for anno in full_file[f'annotations_{task}'] if anno['image_id'] in img_ids_subset] for task in tasks}
print(f"number of images: {len(images_new)}, number of annos {len(annos_new)}")
full_file['images'] = images_new
for task in tasks:
    full_file[f'annotations_{task}'] = annos_new[task]

json_object = json.dumps(full_file, indent=4)

with open(output_file, "w") as outfile:
    outfile.write(json_object)
