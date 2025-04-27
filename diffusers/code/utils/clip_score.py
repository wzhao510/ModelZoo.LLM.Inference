import time

import open_clip
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


class ClipScore:
    def __init__(self,
                model_name, # open clip model name
                model_weights_path, # open clip model weights
                device = None):
        self.device = device
        self.model_name = model_name
        self.model_weights_path = model_weights_path

        if self.device is None:
            device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
        else:
            device = torch.device(self.device)
        
        t_b = time.time()
        print("Load clip model...")
        self.model_clip, _, self.preprocess = open_clip.create_model_and_transforms(
            self.model_name, pretrained=self.model_weights_path, device=device)
        self.model_clip.eval()
        print(f">done. elapsed time: {(time.time() - t_b):.3f} s")
        
        self.tokenizer = open_clip.get_tokenizer(self.model_name)
    
    def process(self, image_info):
        t_b = time.time()
        print("Calc clip score...")
        all_scores = []
        cat_scores = {}

        for i, info in enumerate(image_info):
            image_files = [info['image']]
            if 'category' in info:
                category = info['category']
            prompt = info['prompt']

            # print(f"[{i + 1}/{len(image_info)}] {prompt}")

            image_scores = self.clip_score(self.model_clip,
                                    self.tokenizer,
                                    self.preprocess,
                                    prompt,
                                    image_files,
                                    self.device)
            if len(image_files) > 1:
                best_score = max(image_scores)
            else:
                best_score = image_scores

            # print(f"image scores: {image_scores}")
            # print(f"best score: {best_score}")

            all_scores.append(best_score)
            if 'category' in info:
                if category not in cat_scores:
                    cat_scores[category] = []
                cat_scores[category].append(best_score)
        print(f">done. elapsed time: {(time.time() - t_b):.3f} s")

        average_score = np.average(all_scores)
        print("====================================")
        print(f"average score: {average_score:.3f}")
        if cat_scores :
            print("category average scores:")
            cat_average_scores = {}
            for category, scores in cat_scores.items():
                cat_average_scores[category] = np.average(scores)
                print(f"[{category}], average score: {cat_average_scores[category]:.3f}")
            print("====================================")
        return average_score

    @staticmethod
    def clip_score(model_clip, tokenizer, preprocess, prompt, image_files, device):
        imgs = []
        texts = []
        for image_file in image_files:
            img = preprocess(image_file).unsqueeze(0).to(device)
            imgs.append(img)
            text = tokenizer([prompt]).to(device)
            texts.append(text)

        img = torch.cat(imgs).to(device)   # [bs, 3, 224, 224]
        text = torch.cat(texts).to(device) # [bs, 77]

        with torch.no_grad():
            text_ft = model_clip.encode_text(text).float()
            img_ft = model_clip.encode_image(img).float()
            score = F.cosine_similarity(img_ft, text_ft).squeeze()  
        return score.cpu()