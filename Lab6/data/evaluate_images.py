import os
from PIL import Image
import torch
from torchvision import transforms
from testing.evaluator import evaluation_model
from data.utils import get_testing_labels

def load_image_stack(directory):
    image_list = []
    transform = transforms.Compose([
        transforms.ToTensor(), # Converts to [0, 1] tensor of shape (3, H, W)
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    for i in range(32):
        filename = f"{i}.png"  # Adjust extension if needed (e.g., .jpg)
        path = os.path.join(directory, filename)

        if not os.path.exists(path):
            raise FileNotFoundError(f"Image {filename} not found in {directory}")

        image = Image.open(path).convert("RGB")
        image_tensor = transform(image)
        image_list.append(image_tensor)

    # Stack into a tensor of shape (32, 3, H, W)
    return torch.stack(image_list)

eval_model = evaluation_model()
t = "new_test"
images = load_image_stack(f"images/{t}/").to("cuda")
labels = get_testing_labels(f"./testing/{t}.json", as_multi_hot=True).to("cuda")
acc = eval_model.eval(images, labels)
print(acc)