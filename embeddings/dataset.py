from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

class ImageDataLoader:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.dataset = ImageFolder(self.data_dir, transform=self.transform, target_transform=self._get_class_name)
        
        self.paths = [path for path, _ in self.dataset.samples]
        self.labels = [target for _, target in self.dataset.samples]

        self.dataloader = DataLoader(self.dataset, shuffle=False)

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        return iter(self.dataloader)

    def _get_class_name(self, index):
        return index