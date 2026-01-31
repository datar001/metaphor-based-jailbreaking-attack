import argparse
import os
import pdb
import random
import sys
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from evaluators import config
from evaluators.baselines import SimClassifier, load_prompts

# Set device and seeds for reproducibility
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(2022)
random.seed(2022)
np.random.seed(2022)


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir):
        self.images_dir = images_dir
        self.image_locs = [os.path.join(images_dir, img_name) for img_name in os.listdir(images_dir)]

    def __getitem__(self, idx):
        return self.image_locs[idx]

    def __len__(self):
        return len(self.image_locs)


class q16_predict_image:
    def __init__(self, device):
        checkpoints_dir = "/home/zcy/attack/mayiwen/image_evaluation/Q16/q16"
        checkpoints = os.path.join(checkpoints_dir, os.listdir(checkpoints_dir)[0])
        model_name, pretrained = config.model_name, config.pretrained
        soft_prompts = load_prompts(checkpoints, device)
        self.classifier = SimClassifier(soft_prompts, model_name, pretrained)
        self.classifier.freeze()

        # 将模型移动到指定设备
        self.device = device
        self.classifier.to(self.device)

    def tensor_to_pil(self, tensor):
        unloader = transforms.ToPILImage()
        image = tensor.cpu().clone()  # clone the tensor to not change it directly
        image = image.squeeze(0)  # remove the batch dimension
        image = unloader(image)
        return image

    def preprocess_image(self, img):
        if isinstance(img, torch.Tensor):
            img_pil = self.tensor_to_pil(img)
            preprocessed = self.classifier.preprocess(images=img_pil, return_tensors="pt").pixel_values
        else:
            preprocessed = self.classifier.preprocess(images=img, return_tensors="pt").pixel_values
        return preprocessed.to(self.device)  # 确保预处理后的图像在同一设备上

    def get_image_result(self, images):

        if len(images) == 0:
            return False
        with torch.no_grad():
            images_tensor = self.preprocess_image(images)
            preds = self.classifier(images_tensor).view(len(images), -1)
            preds = torch.argmax(preds, dim=1).detach().cpu().numpy()
        return preds > 0.5  # 返回列表


def main(opt):
    dataset = ImageDataset(images_dir=opt.images_dir)
    loader = torch.utils.data.DataLoader(dataset, batch_size=50, drop_last=False, shuffle=False)

    q16_eva = q16_predict_image(device)
    preds = []

    for paths in loader:
        images = [Image.open(p).convert("RGB") for p in paths]
        image_tensors = [transforms.ToTensor()(img) for img in images]
        result = q16_eva.get_image_result(image_tensors)
        preds.append(result)

    avg_acc = np.mean(preds)
    print(f"The average accuracy of the malicious detection is: {avg_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--images_dir",
        type=str,
        required=True,
        help="Directory containing images for evaluation"
    )
    opt = parser.parse_args()
    main(opt)