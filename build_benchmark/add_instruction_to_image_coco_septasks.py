import random
import numpy as np
import json
import re
import itertools
import copy
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--task", type=str, default="task1", help="path to annotation.")
    args = parser.parse_args()
    return args


args = parse_args()

def setup_seeds():
    seed = 0
    random.seed(seed)
    np.random.seed(seed)

def deleteLeadingZeros(inputString):
    # regex pattern for removing leading zeros from an input string
    regexPattern = "^0+(?!$)"
    # Replace the matched regex pattern with an empty string
    outputString = re.sub(regexPattern, "", inputString)
    # returning output string after removing leading 0s
    return outputString

subset_size = 10
coco_anno = "/home/pirenjie/data/coco/annotations/instances_val2017.json"
task = args.task
instruct_anno = f"/home/pirenjie/data/coco/annotations/{task}/coco_general_v2_cls_reorganized_{task}.json"
output_file = f"/home/pirenjie/data/coco/annotations/{task}/train_2017_{task}_evaluation.json"
tasks = ["task1", "task2", "task3"]
with open(coco_anno, 'r')  as f:
    full_file = json.load(f)
coco_new_anno = copy.deepcopy(full_file)
with open(instruct_anno, 'r')  as f:
    instruct_file = json.load(f)

all_classes = list(set([f["name"] for f in full_file['categories']]))

classid_to_name = {
    d['id'] : d['name'] for d in full_file["categories"]
}

classid_to_index = {
    d['id'] : all_classes.index(d['name']) for d in full_file["categories"]
}
# str(img["id"]) : [anno for anno in full_file["annotations"] if anno['image_id'] == img["id"]] for img in full_file["images"]
annos_per_image = {}
for anno in full_file["annotations"]:
    if str(anno['image_id']) not in annos_per_image:
        annos_per_image[str(anno['image_id'])] = []
    annos_per_image[str(anno['image_id'])].append(anno)
image_to_instructs = dict()

# reformat task annotation, extract categories for each instruct, filter out those that are not in the categories of coco
pattern1 = r"(?i)therefore,?\s+the\s+answer\s+is:?[\s\[\],]*(\w+[\s,]*)+([ ,]\w+[\s,]*)*"
pattern2 = r"(?i)therefore,?\s+the\s+target\s+objects?\s+are:?[\s\[\],]*(\w+[\s,]*)+([ ,]\w+[\s,]*)*"
new_task_annos = []
for anno in instruct_file['annotations']:
    add_to_new_anno = True
    # Use re.search() to find the match
    match1 = re.search(pattern1, anno['answer'])
    match2 = re.search(pattern2, anno['answer'])
    # Extract the matched substring
    if match1:
        substr = match1.group(0)
        # Remove the unnecessary characters
        substr = re.sub(r"(?i)therefore,?\s+the\s+answer\s+is:?[\s\[\],]*", "", substr)
        categories = re.sub(r"[\[\]]", "", substr)
        cat_list = [c.strip() for c in categories.split(',')]
        # remove duplicate
        cat_list = list(set(cat_list))
        cat_list_in_gt = [cat for cat in cat_list if cat in all_classes]
        if len(cat_list_in_gt)<=0:
            add_to_new_anno = False
    elif match2:
        substr = match2.group(0)
        # Remove the unnecessary characters
        substr = re.sub(r"(?i)therefore,?\s+the\s+target\s+objects?\s+are:?[\s\[\],]*", "", substr)
        categories = re.sub(r"[\[\]]", "", substr)
        cat_list = [c.strip() for c in categories.split(',')]
        # remove duplicate
        cat_list = list(set(cat_list))
        cat_list_in_gt = [cat for cat in cat_list if cat in all_classes]
        if len(cat_list_in_gt)<=0:
            add_to_new_anno = False
    else:
        print("no match")
        print(f"task:\n {anno['task']}")
        add_to_new_anno = False
    if add_to_new_anno:
        new_task_anno = {"image_id": deleteLeadingZeros(anno["image_id"]), "task": anno["task"], "categories": cat_list_in_gt}
        new_task_annos.append(new_task_anno)

instruct_related_classes = list(set(itertools.chain.from_iterable([anno["categories"] for anno in new_task_annos])))
print(f"{len(instruct_related_classes)} classes are related to instructs")
all_image_ids = list(set([f["image_id"] for f in new_task_annos]))
all_image_ids = [id for id in all_image_ids if id in annos_per_image]
# record instructions for each image
for anno in new_task_annos:
    if anno["image_id"] not in image_to_instructs:
        image_to_instructs[anno["image_id"]] = [anno]
    else:
        image_to_instructs[anno["image_id"]].append(anno)

# replace images
new_images = [img for img in full_file["images"] if str(img['id']) in all_image_ids]
coco_new_anno[f"images"] = new_images
# construct annotation for each task
for task in tasks:
    annos_for_task = []
    instruct_for_task = {}
    for img_id in all_image_ids:
        # randomly select an instruction from the instructs to img_id
        instruct_for_img = random.choice(image_to_instructs[img_id])
        instruct_for_task[img_id]=instruct_for_img
        # keep box annotations only for sampled instruct
        box_anno_for_img = [anno for anno in annos_per_image[img_id] if classid_to_name[anno['category_id']] in instruct_for_img["categories"]]
        annos_for_task.extend(box_anno_for_img)
    coco_new_anno[f"annotations_{task}"] = annos_for_task
    coco_new_anno[f"instruct_{task}"] = instruct_for_task

json_object = json.dumps(coco_new_anno, indent=4)

with open(output_file, "w") as outfile:
    outfile.write(json_object)

print("done preparing ... ")