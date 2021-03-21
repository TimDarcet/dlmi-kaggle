from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from data import lymph_datamodule
from model import *


pl.seed_everything(42)
    
# Handle the data
dm = lymph_datamodule("data", batch_size=8)

# Define model
model = TransferFeaturizer()

# Exp logger
logger = TensorBoardLogger('logs/tensorboard_logs')

# Define training
checkpointer = ModelCheckpoint(monitor='val_accbal',
                               save_top_k=3,
                               mode='min',
                               save_last=True,
                               filename='{epoch}-{val_loss:.2f}-{train_loss:.2f}')
trainer = pl.Trainer(gpus=1,
                     max_epochs=10,
                     #callbacks=[checkpointer],
                     logger=logger,
                     val_check_interval=0.5)

# Train
trainer.fit(model, dm)