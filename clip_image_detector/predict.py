import pdb
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import argparse
from transformers import CLIPModel, AutoProcessor
import os
import json
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


class Normalization(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.register_buffer('mean', torch.zeros(shape))
        self.register_buffer('variance', torch.ones(shape))

    def forward(self, x):
        return (x - self.mean) / self.variance.sqrt()


class L14_NSFW_Detector(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = Normalization([768])
        self.linear_1 = nn.Linear(768, 64)
        self.linear_2 = nn.Linear(64, 512)
        self.linear_3 = nn.Linear(512, 256)
        self.linear_4 = nn.Linear(256, 1)
        self.act = nn.ReLU()
        self.act_out = nn.Sigmoid()

    def forward(self, x):
        x = self.norm(x)
        x = self.act(self.linear_1(x))
        x = self.act(self.linear_2(x))
        x = self.act(self.linear_3(x))
        x = self.act_out(self.linear_4(x))
        return x

    def load_model(self, model_path='/home/zcy/attack/mayiwen/image_evaluation/clip_image_detector/L_14_nsfw.pth'):
        try:
            state_dict = torch.load(model_path, map_location=torch.device('cpu'))
            self.load_state_dict(state_dict)
        except Exception as e:
            print(f"Error loading L14 model: {e}")
            raise


class H14_NSFW_Detector(nn.Module):
    def __init__(self, input_size=1024):
        super().__init__()
        self.input_size = input_size
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 16),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.layers(x)

    def load_model(self, model_path='/home/zcy/attack/mayiwen/image_evaluation/clip_image_detector/h14_nsfw.pth'):
        try:
            state_dict = torch.load(model_path, map_location=torch.device('cpu'))
            self.load_state_dict(state_dict)
        except Exception as e:
            print(f"Error loading H14 model: {e}")
            raise


class nsfw_model(nn.Module):
    def __init__(self, model_id, device):
        super(nsfw_model, self).__init__()
        if model_id == 'L14_NSFW_Detector':
            self.nsfw_detector = L14_NSFW_Detector()
            self.model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
            self.processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
        elif model_id == 'H14_NSFW_Detector':
            self.nsfw_detector = H14_NSFW_Detector()
            self.model = CLIPModel.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
            self.processor = AutoProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
        else:
            raise ValueError('Nsfw detector must be L14_NSFW_Detector or H14_NSFW_Detector')

        self.nsfw_detector.load_model()
        self.nsfw_detector.to(device)
        self.model.to(device)
        self.device = device

    def forward_image(self, images):
        try:
            inputs = self.processor(images=images, return_tensors="pt")
            for key, value in inputs.items():
                if torch.is_tensor(value):
                    inputs[key] = value.to(self.device)
            image_features = self.model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            preds = self.nsfw_detector(image_features)
            return preds
        except Exception as e:
            print(f"Error in forward_image: {e}")
            raise

    def forward_prompt(self, prompts):
        try:
            inputs = self.processor(text=prompts, padding=True, return_tensors="pt", truncation=True)
            for k, v in inputs.items():
                inputs[k] = v.to(self.device)
            text_embeds = self.model.get_text_features(**inputs)
            text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
            preds = self.nsfw_detector(text_embeds)
            return preds
        except Exception as e:
            print(f"Error in forward_prompt: {e}")
            raise

    def get_prompt_feature(self, prompts, type="global"):
        inputs = self.processor(text=prompts, padding=True, return_tensors="pt", truncation=True)
        for k, v in inputs.items():
            inputs[k] = v.to(self.device)
        if type == "global":
            text_features = self.model.get_text_features(**inputs)
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)  # N * 1024
        else:
            text_features = self.model.text_model(**inputs)
            text_features = text_features[0]  # N * 77 * 1024
        return text_features

    def image_text_score(self, prompts, images):
        inputs = self.processor(text=prompts, padding=True, return_tensors="pt", truncation=True)
        for k, v in inputs.items():
            inputs[k] = v.to(self.device)
        text_features = self.model.get_text_features(**inputs)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

        inputs = self.processor(images=images, return_tensors="pt")
        for key, value in inputs.items():
            if torch.is_tensor(value):
                inputs[key] = value.to(self.device)
        image_features = self.model.get_image_features(**inputs)
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

        sims = text_features @ image_features.t()
        return sims.cpu().detach()

    def text_text_score(self, prompt1, prompt2):
        inputs = self.processor(text=prompt1, padding=True, return_tensors="pt", truncation=True)
        for k, v in inputs.items():
            inputs[k] = v.to(self.device)
        text_features_1 = self.model.get_text_features(**inputs)
        text_features_1 = text_features_1 / text_features_1.norm(p=2, dim=-1, keepdim=True)

        inputs = self.processor(text=prompt2, padding=True, return_tensors="pt", truncation=True)
        for k, v in inputs.items():
            inputs[k] = v.to(self.device)
        text_features_2 = self.model.get_text_features(**inputs)
        text_features_2 = text_features_2 / text_features_2.norm(p=2, dim=-1, keepdim=True)

        sims = text_features_1 @ text_features_2.t()
        return sims.cpu().detach()  # [1,m]


def load_images(img_path, root=None):
    images = []
    for path in img_path:
        if root is not None:
            path = os.path.join(root, path)
        try:
            image = Image.open(path).convert("RGB")
            images.append(image)
        except Exception as e:
            print(f"Error loading image {path}: {e}")
    return images


class clip_predict_image:
    def __init__(self, device):
        self.device = device
        self.nsfw_model = nsfw_model("L14_NSFW_Detector", device)
        self.image_text_score = self.nsfw_model.image_text_score
        self.text_text_score = self.nsfw_model.text_text_score

    def _process_image(self, image):
        image = np.array(image).astype(np.float32)
        if image.max() > 1.0:
            image = image / 255.0
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = np.transpose(image, (2, 0, 1))
        return image

    def get_image_result(self, images):
        if len(images) == 0:
            return False
        with torch.no_grad():
            preds = self.nsfw_model.forward_image(images)
            preds = preds.squeeze().detach().cpu()
        return preds > 0.5  # 返回列表[True, False, False, True]



def plot_auroc(y_true, y_pred):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mal_prompt_path', type=str, required=True, help='Path to the malicious prompts file')
    parser.add_argument('--clean_prompt_path', type=str, required=True, help='Path to the clean prompts file')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for processing')
    parser.add_argument('--gt', type=str, default='unsafe', help='Ground truth label for the image type')
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    nsfw_model = nsfw_model("H14_NSFW_Detector", device)

    # Load malicious prompts
    try:
        if args.mal_prompt_path.endswith('.txt'):
            with open(args.mal_prompt_path, 'r') as f:
                test_mal_data = f.readlines()
        else:
            with open(args.mal_prompt_path, 'r') as f:
                mal_data = json.load(f)
            test_mal_data = [info['prompt'] if isinstance(info['prompt'], list) else info['prompt'] for info in mal_data.values()]
    except Exception as e:
        print(f"Error loading malicious prompts: {e}")
        sys.exit(1)

    # Load clean prompts
    try:
        with open(args.clean_prompt_path, 'r') as f:
            clean_data = json.load(f)
        test_clean_data = [info['prompt'] if isinstance(info['prompt'], list) else info['prompt'] for info in clean_data.values()]
    except Exception as e:
        print(f"Error loading clean prompts: {e}")
        sys.exit(1)

    # Combine datasets
    test_data = test_mal_data + test_clean_data
    test_label = [1] * len(test_mal_data) + [0] * len(test_clean_data)

    # Predict
    clip_predictor = clip_predict_image(device)
    all_preds = []
    all_labels = []

    for i in range(0, len(test_data), args.batch_size):
        batch_prompts = test_data[i:i + args.batch_size]
        try:
            batch_preds = nsfw_model.forward_prompt(batch_prompts)
            batch_preds = batch_preds.squeeze().cpu().numpy()
            all_preds.extend(batch_preds)
            all_labels.extend(test_label[i:i + args.batch_size])
        except Exception as e:
            print(f"Error predicting batch {i // args.batch_size}: {e}")

    # Plot ROC curve
    plot_auroc(all_labels, all_preds)