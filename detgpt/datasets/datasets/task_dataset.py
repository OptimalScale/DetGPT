import os
from PIL import Image
import webdataset as wds
from detgpt.datasets.datasets.base_dataset import BaseDataset
from detgpt.datasets.datasets.caption_datasets import CaptionDataset



class CCSBUAlignDataset(CaptionDataset):

    def __getitem__(self, index):

        # TODO this assumes image input, not general enough
        ann = self.annotation[index]

        img_file = '{}.jpg'.format(ann["image_id"])
        image_path = os.path.join(self.vis_root, img_file)
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        task = ann["task"]
        answer = ann["answer"]

        return {
            "image": image,
            "task": task,
            "answer": answer,
            "image_id": self.img_ids[ann["image_id"]],
        }
