import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from src.utils import convert_text_to_feature

class Text2ImageDataset(Dataset):
    def __init__(self, dataset, transform=None, split=0):

        self.transform = transform
        self.dataset = dataset if isinstance(split, str) else dataset['train'] 
        self.split = split

        self.texts = self.dataset['text']
        self.images = self.dataset['image']
        self.labels = self.dataset['label']
        self.embeddings = self._compute_embeddings(self.texts)

    def _compute_embeddings(self, texts):
        batch_size = 8
        embeddings_list = []  

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]  

            batch_embeddings = convert_text_to_feature(batch_texts).detach().cpu()
            embeddings_list.append(batch_embeddings)
            print("Batch done")

        embeddings = torch.cat(embeddings_list, dim=0) 
        return embeddings

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        right_image = self.images[idx]  
        if isinstance(right_image, Image.Image):
            right_image = np.array(right_image) 
        if self.transform is not None:
            right_image = self.transform(right_image)

        txt = self.texts[idx]
        right_embed = self.embeddings[idx]  

        wrong_image = self.find_wrong_image(self.labels[idx])
        if self.transform is not None:
            wrong_image = self.transform(wrong_image)

        sample = {
            'right_images': torch.FloatTensor(right_image),  
            'right_embed': right_embed,  
            'wrong_images': torch.FloatTensor(wrong_image), 
            'txt': str(txt)  
        }
        return sample

    def find_wrong_image(self, category):

        wrong_indices = [i for i, label in enumerate(self.labels) if label != category]

        wrong_idx = np.random.choice(wrong_indices)
        wrong_image = self.images[wrong_idx]

        if isinstance(wrong_image, Image.Image):
            wrong_image = np.array(wrong_image)

        return wrong_image