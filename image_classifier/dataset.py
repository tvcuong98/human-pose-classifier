import os
from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
class IRDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []

        # Load all the images and labels
        for label, folder in enumerate([0,1,2,3,4,5,6,7,8]):
            folder_path = os.path.join(self.root_dir, str(folder))
            for image_name in os.listdir(folder_path):
                self.images.append(os.path.join(folder_path, image_name))  
                self.labels.append(label)

        # Use the correct set for this instance
        self.data = self.images 
        self.labels = self.labels 

    def __len__(self):
        return len(self.data)
    def get_labels(self):
        return self.labels
    def __getitem__(self, idx):
        image_path = self.data[idx]
        label = self.labels[idx]
        image = Image.open(image_path)

        if image.mode == 'P':
            image = image.convert('RGBA')

        image = image.convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label