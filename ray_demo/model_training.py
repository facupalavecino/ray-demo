import boto3
import lightning as L
import ray.train.lightning
import torch

from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from ray.train import Checkpoint, CheckpointConfig, RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryJaccardIndex, Dice
from torchvision.transforms import v2

from ray_demo.datasets import SegmentationS3Dataset
from ray_demo.models.unet import UNet

# --------------------------------------------------
# Load data
# --------------------------------------------------
s3 = boto3.client("s3")

bucket = "ray-demo-data"

image_filenames = []
mask_filenames = []

paginator = s3.get_paginator("list_objects_v2")

for result in paginator.paginate(
    Bucket=bucket, Prefix="football-players-recognizer/images/"
):
    for content in result.get("Contents", []):
        image_filenames.append(content["Key"])

for result in paginator.paginate(
    Bucket=bucket, Prefix="football-players-recognizer/masks/"
):
    for content in result.get("Contents", []):
        mask_filenames.append(content["Key"])

# --------------------------------------------------
# Split data
# --------------------------------------------------
images_train, images_test, masks_train, masks_test = train_test_split(
    image_filenames, mask_filenames, test_size=0.1, random_state=180
)
images_train, images_val, masks_train, masks_val = train_test_split(
    images_train, masks_train, test_size=0.1127, random_state=180
)  # 0.1 * 0.9 = 0.1127

# --------------------------------------------------
# Preprocessing functions
# --------------------------------------------------
image_transform = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ]
)

mask_transform = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32),
    ]
)

# --------------------------------------------------
# Datasets
# --------------------------------------------------
train_dataset = SegmentationS3Dataset(
    bucket=bucket,
    image_files=images_train,
    mask_files=masks_train,
    transform=image_transform,
    target_transform=mask_transform,
)

val_dataset = SegmentationS3Dataset(
    bucket=bucket,
    image_files=images_val,
    mask_files=masks_val,
    transform=image_transform,
    target_transform=mask_transform,
)

test_dataset = SegmentationS3Dataset(
    bucket=bucket,
    image_files=images_test,
    mask_files=masks_test,
    transform=image_transform,
    target_transform=mask_transform,
)

# --------------------------------------------------
# Dataloaders
# --------------------------------------------------
BATCH_SIZE = 8

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=False,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    drop_last=False,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    drop_last=False,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
)

# --------------------------------------------------
# Lightning Data module
# --------------------------------------------------
class PlayerImagesDataModule(L.LightningDataModule):
    def __init__(self, bucket_name: str, batch_size: int = 32):
        super().__init__()
        self.bucket_name = bucket_name
        self.batch_size = batch_size
        self.image_transform = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
            ]
        )
        self.mask_transform = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32),
            ]
        )

    def setup(self, stage: str):
        s3 = boto3.client("s3")
        image_filenames = []
        mask_filenames = []

        paginator = s3.get_paginator("list_objects_v2")

        for result in paginator.paginate(
            Bucket=self.bucket_name, Prefix="football-players-recognizer/images/"
        ):
            for content in result.get("Contents", []):
                image_filenames.append(content["Key"])

        for result in paginator.paginate(
            Bucket=self.bucket_name, Prefix="football-players-recognizer/masks/"
        ):
            for content in result.get("Contents", []):
                mask_filenames.append(content["Key"])
        
        images_train, images_test, masks_train, masks_test = train_test_split(
            image_filenames, mask_filenames, test_size=0.1, random_state=180
        )
        images_train, images_val, masks_train, masks_val = train_test_split(
            images_train, masks_train, test_size=0.1127, random_state=180
        )  # 0.1 * 0.9 = 0.1127

        
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=True, drop_last=False, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, shuffle=False, drop_last=False, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, shuffle=False, drop_last=False, batch_size=self.batch_size)

# --------------------------------------------------
# Lightning model
# --------------------------------------------------
class PlayerRecognizer(L.LightningModule):
    def __init__(self):
        super(PlayerRecognizer, self).__init__()
        self.model = UNet(n_channels=3, n_classes=1)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.dice = Dice()
        self.iou = BinaryJaccardIndex()
        self.example_input_array = torch.rand(BATCH_SIZE, 3, 216, 384)

        self.validation_step_outputs = {
            "val_loss": [],
            "val_dice": [],
            "val_iou": [],
        }

        self.testing_step_outputs = {
            "test_loss": [],
            "test_dice": [],
            "test_iou": [],
        }

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        images, masks = batch
        outputs = self.model(images)
        loss = self.loss_fn(outputs, masks)
        self.log("train_loss", loss)
        return loss

    def clear_validation_outputs(self):
        self.validation_step_outputs = {
            "val_loss": [],
            "val_dice": [],
            "val_iou": [],
        }

    def clear_testing_outputs(self):
        self.testing_step_outputs = {
            "test_loss": [],
            "test_dice": [],
            "test_iou": [],
        }

    def validation_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        images, masks = batch
        outputs = self.model(images)
        loss = self.loss_fn(outputs, masks)
        dice = self.dice(outputs, masks.int())
        iou = self.iou(outputs, masks.int())
        self.validation_step_outputs["val_loss"].append(loss)
        self.validation_step_outputs["val_dice"].append(dice)
        self.validation_step_outputs["val_iou"].append(iou)

        return loss

    def on_validation_epoch_end(self):
        """
        Computes average validation loss
        """
        avg_loss = torch.stack(self.validation_step_outputs["val_loss"]).mean()
        avg_dice = torch.stack(self.validation_step_outputs["val_dice"]).mean()
        avg_iou = torch.stack(self.validation_step_outputs["val_iou"]).mean()
        self.log_dict({"val_loss": avg_loss, "val_dice": avg_dice, "val_iou": avg_iou})

        self.clear_validation_outputs()

    def test_step(self, batch, batch_idx):
        images, masks = batch

        outputs = self.model(images)

        loss = self.loss_fn(outputs, masks)
        dice = self.dice(outputs, masks.int())
        iou = self.iou(outputs, masks.int())

        self.testing_step_outputs["test_loss"].append(loss)
        self.testing_step_outputs["test_dice"].append(dice)
        self.testing_step_outputs["test_iou"].append(iou)

        return loss

    def on_test_epoch_end(self):
        avg_test_loss = torch.stack(self.testing_step_outputs["test_loss"]).mean()
        avg_test_dice = torch.stack(self.testing_step_outputs["test_dice"]).mean()
        avg_test_iou = torch.stack(self.testing_step_outputs["test_iou"]).mean()

        self.log_dict(
            {
                "test_loss": avg_test_loss,
                "test_dice": avg_test_dice,
                "test_iou": avg_test_iou,
            },
            prog_bar=True,
        )

        self.clear_testing_outputs()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

    def forward(self, x):
        return self.model(x)


model = PlayerRecognizer()

checkpoint_callback = ModelCheckpoint(
    monitor="val_loss", save_top_k=3, every_n_epochs=5, mode="min"
)

early_stopping = EarlyStopping(monitor="val_loss", mode="min")

wandb_logger = WandbLogger(project="Football Players Segmentation", log_model="all")

wandb_logger.watch(model, log_freq=10)
wandb_logger.experiment.config["batch_size"] = BATCH_SIZE

trainer = L.Trainer(
    max_epochs=30,
    check_val_every_n_epoch=2,
    accelerator="auto",
    devices="auto",
    callbacks=[early_stopping, checkpoint_callback],
    logger=wandb_logger,
)

trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

trainer.test(model=model, dataloaders=test_loader, ckpt_path="best")
# --------------------------------------------------
# Ray Train
# --------------------------------------------------


def train_func():
    model = PlayerRecognizer()

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss", save_top_k=3, every_n_epochs=5, mode="min"
    )

    early_stopping = EarlyStopping(monitor="val_loss", mode="min")

    wandb_logger = WandbLogger(project="Football Players Segmentation", log_model="all")

    wandb_logger.watch(model, log_freq=10)
    wandb_logger.experiment.config["batch_size"] = BATCH_SIZE

    trainer = L.Trainer(
        max_epochs=30,
        check_val_every_n_epoch=2,
        accelerator="auto",
        devices="auto",
        strategy=ray.train.lightning.RayDDPStrategy(),
        plugins=[ray.train.lightning.RayLightningEnvironment()],
        callbacks=[ray.train.lightning.RayTrainReportCallback()],
        # [1a] Optionally, disable the default checkpointing behavior
        # in favor of the `RayTrainReportCallback` above.
        enable_checkpointing=False,
    )

    trainer = ray.train.lightning.prepare_trainer(trainer)
    trainer.fit(model, train_dataloaders=train_dataloader)

scaling_config = ScalingConfig(num_workers=2, use_gpu=True)

trainer = TorchTrainer(
    train_loop_per_worker=train_func,
    train_loop_config={},
)