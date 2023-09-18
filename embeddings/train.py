from dataset import ImageDataLoader
from torch.utils.data import DataLoader
from models.model_selector import get_model
from pipelines.pipeline_selector import get_pipeline
from options import BaseOptions, Config, load_parameters
from utils import save_checkpoint
import torch
import numpy as np
import pickle


opt = BaseOptions().parse()
config = Config(**load_parameters(opt.config_file))

model = get_model(config.model)
train_model = get_pipeline(config.pipeline)

optimizer, loss = train_model(
    model=model,
    config=config
)


# compute embeddings and save
data = ImageDataLoader(config.data_path)
dataloader = DataLoader(data.dataset, batch_size=config.inf_batch_size, shuffle=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

target = []
paths = []
labels = []
feature_embeddings = np.empty((0, config.embedding_dim))

for i, (x, y, path, label) in enumerate(dataloader):
    x = x.to(device=device)
    with torch.no_grad():
        batch_features = model(x)

    batch_features = batch_features.view(
        batch_features.size(0), -1).cpu().numpy()
    feature_embeddings = np.vstack((feature_embeddings, batch_features))
    target.extend(list(y.cpu().detach().numpy()))
    paths.extend(path)
    labels.extend(label)


data_dict = {
    "model": config.model,
    "embedding": feature_embeddings,
    "target": target,
    "paths": paths,
    "classes": labels
}

with open(config.save_file_path, 'wb') as pickle_file:
    pickle.dump(data_dict, pickle_file)


save_checkpoint(model, optimizer, loss, config)
