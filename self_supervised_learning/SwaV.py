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

from lightly.loss import SwaVLoss
from lightly.models.modules import SwaVProjectionHead, SwaVPrototypes
from lightly.transforms.swav_transform import SwaVTransform

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class SwaV(torch.nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.projection_head = SwaVProjectionHead(1280, 1280, 128)
        self.prototypes = SwaVPrototypes(128, n_prototypes=1280)

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        x = self.projection_head(x)
        x = torch.nn.functional.normalize(x, dim=1, p=2)
        p = self.prototypes(x)
        return p


# Use a resnet backbone.
backbone = EfficientNet.from_pretrained('efficientnet-b0')
# Ignore the classification head as we only want the features.
backbone._fc = torch.nn.Identity()


model = SwaV(backbone)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

transform = SwaVTransform()

dataset = LightlyDataset(input_dir="../data/02_data_split/test_data/", transform=transform)


dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=20,
    shuffle=True,
    drop_last=True,
    # num_workers=8,
)

criterion = SwaVLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

loss_curve = np.array([])
num_epochs = 1
print("Starting Training")
for epoch in range(num_epochs):
    total_loss = 0
    for batch in dataloader:
        views = batch[0]
        model.prototypes.normalize()
        multi_crop_features = [model(view.to(device)) for view in views]
        high_resolution = multi_crop_features[:2]
        low_resolution = multi_crop_features[2:]
        loss = criterion(high_resolution, low_resolution)
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
nome_arquivo = f"training_loss_curve_{timestamp}.jpg"

# Salvar o gráfico
plt.savefig(nome_arquivo)
# plt.show()


caminho_do_arquivo = 'efficientnet_SwaV_model_test.pth'
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
        "model": 'efficientnet_SwaV',
        "embedding":feature_embeddings,
        "target":target,
        "paths": paths,
        "classes":labels
    }

    with open(f'./efficientnet_SwaV_{method}.pickle', 'wb') as pickle_file:
        pickle.dump(data_dict, pickle_file)


