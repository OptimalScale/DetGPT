import json
import random
import numpy as np

def setup_seeds():
    seed = 0
    random.seed(seed)
    np.random.seed(seed)

subset_size = 10
ANNOTATIONS_FILE_PATH = "/home/pirenjie/data/coco/annotations/instances_val2017.json"
output_file = "/home/pirenjie/data/coco/annotations/instances_val2017_subset10.json"
with open(ANNOTATIONS_FILE_PATH, 'r')  as f:
    full_file = json.load(f)

images_new = random.sample(full_file['images'], subset_size)
img_ids_subset = [img['id'] for img in images_new]
annos_new = [anno for anno in full_file['annotations'] if anno['image_id'] in img_ids_subset]
print(f"number of images: {len(images_new)}, number of annos {len(annos_new)}")
full_file['images'] = images_new
full_file['annotations'] = annos_new

json_object = json.dumps(full_file, indent=4)

with open(output_file, "w") as outfile:
    outfile.write(json_object)
