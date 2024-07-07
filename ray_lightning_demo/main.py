import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchmetrics import Accuracy, Precision, Recall
from PIL import Image
import boto3
from io import BytesIO

# S3 Dataset
class S3ImageNetDataset(Dataset):
    def __init__(self, bucket_name, prefix, transform=None):
        self.s3 = boto3.client('s3')
        self.bucket_name = bucket_name
        self.prefix = prefix
        self.transform = transform
        self.image_list = self._get_image_list()

    def _get_image_list(self):
        paginator = self.s3.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=self.bucket_name, Prefix=self.prefix)
        image_list = []
        for page in pages:
            for obj in page['Contents']:
                if obj['Key'].lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_list.append(obj['Key'])
        return image_list

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_key = self.image_list[idx]
        response = self.s3.get_object(Bucket=self.bucket_name, Key=image_key)
        image = Image.open(BytesIO(response['Body'].read()))
        
        if self.transform:
            image = self.transform(image)
        
        # Extract label from the file path
        label = int(image_key.split('/')[-2])  # Assuming format: .../class_id/image.jpg
        return image, label

# Lightning Module
class LightningModel(pl.LightningModule):
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.resnet50(pretrained=True)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, num_classes)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.accuracy = Accuracy()
        self.precision = Precision(average='macro', num_classes=num_classes)
        self.recall = Recall(average='macro', num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.accuracy(preds, y)
        self.precision(preds, y)
        self.recall(preds, y)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_accuracy', self.accuracy, prog_bar=True)
        self.log('val_precision', self.precision, prog_bar=True)
        self.log('val_recall', self.recall, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.accuracy(preds, y)
        self.precision(preds, y)
        self.recall(preds, y)
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_accuracy', self.accuracy, prog_bar=True)
        self.log('test_precision', self.precision, prog_bar=True)
        self.log('test_recall', self.recall, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

# Data Module
class ImageNetDataModule(pl.LightningDataModule):
    def __init__(self, bucket_name, prefix, batch_size=32):
        super().__init__()
        self.bucket_name = bucket_name
        self.prefix = prefix
        self.batch_size = batch_size

    def setup(self, stage=None):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        full_dataset = S3ImageNetDataset(self.bucket_name, self.prefix, transform=transform)
        
        # Split dataset
        train_size = int(0.7 * len(full_dataset))
        val_size = int(0.15 * len(full_dataset))
        test_size = len(full_dataset) - train_size - val_size
        self.train_dataset, self.val_dataset, self.test_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size, test_size]
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=4)

# Main training script
def main():
    bucket_name = 'ray-poc-data'
    prefix = 'imagenet/'
    num_classes = 1000  # Adjust based on your subset

    # Initialize data module
    data_module = ImageNetDataModule(bucket_name, prefix, batch_size=32)

    # Initialize model
    model = LightningModel(num_classes=num_classes)

    # Setup MLflow logger
    mlf_logger = MLFlowLogger(experiment_name="imagenet_training", tracking_uri="http://localhost:5000")

    # Setup checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints',
        filename='imagenet-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        monitor='val_loss'
    )

    # Initialize Trainer
    trainer = pl.Trainer(
        max_epochs=10,
        gpus=1,
        logger=mlf_logger,
        callbacks=[checkpoint_callback],
        progress_bar_refresh_rate=20,
    )

    # Train the model
    trainer.fit(model, data_module)

    # Test the model
    trainer.test(model, datamodule=data_module)

if __name__ == '__main__':
    main()