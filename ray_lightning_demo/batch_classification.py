import ray
import torch
from torchvision.models import ResNet152_Weights
from torchvision import transforms
from torchvision import models
import numpy as np
from typing import Dict


weights = ResNet152_Weights.IMAGENET1K_V1

transform = transforms.Compose([transforms.ToTensor(), weights.transforms()])


def preprocess_image(row: Dict[str, np.ndarray]):
    return {
        "original_image": row["image"],
        "transformed_image": transform(row["image"]),
    }


class ResnetModel:
    def __init__(self):
        self.weights = weights
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.resnet152(weights=self.weights).to(self.device)
        self.model.eval()

    def __call__(self, batch: Dict[str, np.ndarray]):
        # Convert the numpy array of images into a PyTorch tensor.
        # Move the tensor batch to GPU if available.
        torch_batch = torch.from_numpy(batch["transformed_image"]).to(self.device)
        with torch.inference_mode():
            prediction = self.model(torch_batch)
            predicted_classes = prediction.argmax(dim=1).detach().cpu()
            predicted_labels = [
                self.weights.meta["categories"][i] for i in predicted_classes
            ]
            return {
                "predicted_label": predicted_labels,
                "original_image": batch["original_image"],
            }


def main():

    ray.init()

    s3_uri = "s3://anonymous@air-example-data-2/imagenette2/train/"

    ds = ray.data.read_images(s3_uri, mode="RGB")

    predictions = (
        ds
        .map(preprocess_image)
        .map_batches(
            ResnetModel,
            concurrency=2,  # 2 actors will process the batches
            num_gpus=1,  # Each actor will use 1 GPU
            batch_size=32,  # Use the largest batch size that can fit on our GPUs
        )
    )

    predictions.write_json("s3://ray-demo-data/batch_classification/", concurrency=8)

if __name__ == "__main__":
    main()
