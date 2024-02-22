from datetime import datetime
import sys

sys.path.append('../embeddings/')
sys.path.append('./embeddings/')
sys.path.append('..')

from dataset import ImageDataLoader
from torch.utils.data import DataLoader
from utils import slice_image_paths
import matplotlib.pyplot as plt
import pickle
import torch
import torchvision
from efficientnet_pytorch import EfficientNet
import numpy as np

from lightly import loss

from lightly.data import LightlyDataset
from lightly.models.modules import heads

from lightly.transforms import SimCLRTransform

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class SimCLR(torch.nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.projection_head = heads.SimCLRProjectionHead(
            input_dim=1280,  # Efficientnet features have 1280 dimensions.
            hidden_dim=1280,
            output_dim=128,
        )

    def forward(self, x):
        features = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(features)
        return z


backbone = EfficientNet.from_pretrained('efficientnet-b0')
# Ignore the classification head as we only want the features.
backbone._fc = torch.nn.Identity()

# Build the SimCLR model.
model = SimCLR(backbone)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

transform = SimCLRTransform(input_size=32, cj_prob=0.5)

dataset = LightlyDataset(input_dir="../data/02_data_split/train_data/", transform=transform)


dataloader = torch.utils.data.DataLoader(
    dataset,  # Pass the dataset to the dataloader.
    batch_size=128,  # A large batch size helps with the learning.
    shuffle=True,  # Shuffling is important!
)

# Lightly exposes building blocks such as loss functions.
criterion = loss.NTXentLoss(temperature=0.5)

# Get a PyTorch optimizer.
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=1e-6)


loss_curve = np.array([])
num_epochs = 100
print("Starting Training")
for epoch in range(num_epochs):
    total_loss = 0
    for (view0, view1), targets, filenames in dataloader:
        z0 = model(view0)
        z1 = model(view1)
        loss = criterion(z0, z1)
        total_loss += loss.detach()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    avg_loss = total_loss / len(dataloader)
    loss_curve = np.append(loss_curve, avg_loss.cpu().numpy())
    print(f"epoch: {epoch:>02}, loss: {avg_loss:.5f}")

plt.plot(np.arange(1, num_epochs + 1), loss_curve, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
nome_arquivo = f"efficient_SimCLR_training_loss_curve_{timestamp}.jpg"

# Salvar o gráfico
plt.savefig(nome_arquivo)
# plt.show()


caminho_do_arquivo = 'efficient_SimCLR_model_test.pth'
torch.save(model, caminho_do_arquivo)





model.to(device)
model.eval()

# compute embeddings and save
data_paths = [('train', '../data/02_data_split/train_data/'), ('test', '../data/02_data_split/test_data')]

for method, path in data_paths:

    data = ImageDataLoader(path)
    dataloader = DataLoader(data.dataset, batch_size=50, shuffle=False)

    target = []
    paths = []
    labels = []
    feature_embeddings = np.empty((0, 1280))



    for i, (x, y, path, label) in enumerate(dataloader):
        x = x.to(device=device)
        with torch.no_grad():
            batch_features = model(x)

        batch_features_np = batch_features.view(batch_features.size(0), -1).cpu().numpy()
        feature_embeddings = np.vstack((feature_embeddings, batch_features_np))
        target.extend(list(y.cpu().detach().numpy()))
        paths.extend(slice_image_paths(path))
        labels.extend(label)


    data_dict = {
        "model": 'efficient_SimCLR',
        "embedding":feature_embeddings,
        "target":target,
        "paths": paths,
        "classes":labels
    }

    with open(f'./efficient_SimCLR_{method}.pickle', 'wb') as pickle_file:
        pickle.dump(data_dict, pickle_file)


