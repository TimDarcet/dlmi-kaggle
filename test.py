import pytorch_lightning as pl
from data import lymph_datamodule
from model import TransferResNet
import numpy as np
import pandas as pd
import torch


pl.seed_everything(42)
    
# Handle the data
dm = lymph_datamodule("data", batch_size=8)
dm.setup("test")
print("Loaded dataset")

# Define model
model = TransferResNet.load_from_checkpoint("logs/tensorboard_logs/default/version_20/checkpoints/epoch=1-step=1343.ckpt")
model.eval()
print("Loaded model")

IDs = []
preds = []
scores = []
for step, (im, label, patient_id) in enumerate(dm.test_dataloader()):
    score = model(im)
    scores.append(score)
    preds.append((score > 0).int())
    IDs.append(patient_id)

IDs = torch.cat(IDs)
preds = torch.cat(preds)
scores = torch.cat(scores)

uniq_ids = torch.unique(IDs)
uniq_scores = []
uniq_scores = [scores[IDs == p_id].mean() for p_id in uniq_ids]


results = pd.DataFrame({'Id': uniq_ids,
                        'Predicted': uniq_scores})
results.to_csv('submission_viz.csv', index=False)
