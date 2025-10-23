import json
import torch

from pathlib import Path
from torch.utils.data import Dataset
from torch.nn.functional import one_hot
from torchvision import transforms
from PIL import Image


def get_testing_labels(label_path, as_multi_hot=False):
    label_lists = json.load(open(label_path, "r"))
    object_ids = json.load(open("./data/objects.json", "r"))
    
    if as_multi_hot:
        return torch.stack([
            sum(
                [one_hot(torch.tensor(object_ids[tag]), num_classes=24) for tag in labels],
                torch.zeros(24, dtype=torch.long)
            )
            for labels in label_lists
        ])
    else:
        return torch.tensor([
            [object_ids[tag] for tag in labels] + [24] * (24 - len(labels))
            for labels in label_lists
        ])


class IclevrDataset(Dataset):
    def __init__(
        self,
        root_dir=Path("./data"),
    ):
        super().__init__()
        self.root = root_dir
        self.files = list(json.load(open(root_dir / "train.json", "r")).items())
        self.transform = transforms.Compose(
            [
                transforms.Resize((64, 64)),
                #transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        self.object_ids = json.load(open(root_dir / "objects.json", "r"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img_path = self.root / "iclevr" / self.files[index][0]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        tags = self.files[index][1]
        labels = [self.object_ids[tag] for tag in tags]
        padded_labels = labels + [24] * (24 - len(labels))

        return {"images": img, "labels": torch.tensor(padded_labels, dtype=torch.long)}


if __name__ == "__main__":
    # i = IclevrDataset()[0]
    # print(i["labels"])
    r = get_testing_labels(label_path="./testing/test.json", as_one_hot=True)
    print(r)
