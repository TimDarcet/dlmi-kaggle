import pytorch_lightning as pl
from data import lymph_datamodule
from model import TransferResNet
import numpy as np
import pandas as pd
import torch


pl.seed_everything(42)
    
# Handle the data
dm = lymph_datamodule("data", batch_size=8)
dm.setup()
print("Loaded dataset")

# Define model
model = TransferResNet.load_from_checkpoint("logs/tensorboard_logs/default/version_23/checkpoints/epoch=6-step=8748.ckpt")
model.eval()
print("Loaded model")

IDs = []
preds = []
scores = []
for step, (im, label, patient_id) in enumerate(dm.test_dataloader()):
    score = model(im).detach()
    scores.append(score)
    preds.append((score > 0).int())
    IDs.append(patient_id)
print("Calculated individual scores")

IDs = torch.cat(IDs)
preds = torch.cat(preds)
scores = torch.cat(scores)

uniq_ids = torch.unique(IDs)
uniq_ids = [f"P{i}" for i in uniq_ids]
# uniq_scores = [scores[IDs == p_id].mean().item() for p_id in uniq_ids]
uniq_preds = [int(scores[IDs == p_id].mean().item() > 0) for p_id in uniq_ids]
print("Aggregated scores")


results = pd.DataFrame({'Id': uniq_ids,
                        'Predicted': uniq_preds})
results.to_csv('submission_viz.csv', index=False)
print("Wrote results")
