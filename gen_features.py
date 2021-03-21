import pytorch_lightning as pl
from data import lymph_datamodule
from model import TransferResNet
import numpy as np
import pandas as pd
import torch


pl.seed_everything(42)
    
#  train data
dm = lymph_datamodule("data", batch_size=8)
dm.setup("test")
print("Loaded dataset")

# Define model
model = TransferResNet.load_from_checkpoint("logs/tensorboard_logs/default/version_23/checkpoints/epoch=6-step=8748.ckpt")
model.eval()
print("Loaded model")



dataloaders = [
    torch.utils.data.DataLoader(dm.full_dataset,
                                batch_size=self.batch_size,
                                shuffle=True,
                                num_workers=8),
    dm.train_dataloader()
]


feats = []
IDs = []
for dl in dataloaders:
    for step, (im, label, patient_id) in enumerate(dl):
        feats.append(model.resnet(im).detach())
        IDs.append(patient_id)

# Concatenate to a single array
IDs = torch.cat(IDs)
preds = torch.cat(preds)
scores = torch.cat(scores)

# Reduction
# TODO max or mean reduction
uniq_ids = torch.unique(IDs)
uniq_ids = [f"P{i}" for i in uniq_ids]
uniq_features = torch.cat([features[IDs == p_id].mean(dim=0) for p_id in uniq_ids])
print("Aggregated Features")

# id_df = pd.DataFrame({"Id": uniq_ids})
# features_df = pd.DataFrame(uniq_features, columns=list(range(uniq_features.shape[1])))
full_df = pd.DataFrame({"Id": uniq_ids, **{i: uniq_features[:, i] for i in range(uniq_features.shape[1])}})
full_df.to_csv('data/features.csv', index=False)
print("Wrote results")
