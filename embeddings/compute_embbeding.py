from models.model_selector import get_model
from dataset import ImageDataLoader
from torch.utils.data import DataLoader
import numpy as np
import torch
import pickle 
from options import BaseOptions, load_parameters

## sh script to set differents embeddings models
# class Config:
#     def __init__(self) -> None:
#         self.model = "vgg16"
#         self.embedding_dim = 4096
#         self.pipeline = None
#         self.train = False
#         self.batch_size = 256
#         self.dataset_path = "C:/Users/Maods/Documents/Development/Mestrado/terumo/apps/renal-pathology-retrieval/data/01_raw/"
#         self.save_file_path = ""

class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)

# config = Config()

opt = BaseOptions().parse()
config = load_parameters(opt.config_file)
config = Config(**config)

# Load data
data = ImageDataLoader(config.dataset_path)
dataloader = DataLoader(data.dataset, batch_size=config.batch_size, shuffle=False)

# select embedding model training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model(config.model)
model.to(device)


# train embedings
if config.train:
    train_embedding = get_model(config.pipeline)
    train_embedding(model, dataloader)


# compute embeddings and save
feature_embeddings = np.empty((0, config.embedding_dim))
labels = []
for i, (x, y) in enumerate(dataloader):
    x = x.to(device=device)
    with torch.no_grad():
        batch_features = model(x)

    batch_features = batch_features.view(batch_features.size(0), -1).cpu().numpy()
    feature_embeddings = np.vstack((feature_embeddings, batch_features))
    labels.extend(list(y.cpu().detach().numpy()))


data_dict = {
    "model": config.model,
    "embedding":feature_embeddings,
    "labels":labels,
    "paths": data.paths,
    "classes":data.labels
}

with open(config.save_file_path, 'wb') as pickle_file:
    pickle.dump(data_dict, pickle_file)