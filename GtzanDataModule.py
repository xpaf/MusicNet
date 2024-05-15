import lightning
import splitfolders
import os
import torchvision.transforms.v2 as transforms
import gdown
import shutil
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


class GtzanDataModule(lightning.LightningDataModule):

    def __init__(self, split_seed: int, batch_size: int, shuffle_datasets: bool):
        super().__init__()
        self.test_dataset = None
        self.val_dataset = None
        self.train_dataset = None
        self.split_seed: int = split_seed
        self.batch_size: int = batch_size
        self.shuffle_datasets: bool = shuffle_datasets
        self.transformer = transforms.Compose([
            transforms.Resize((128, 216)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
        ])

    def prepare_data(self) -> None:
        # Download dataset .zip archive
        gdown.download(id='18apG_DhUjD_w3e5RJ57mSOA_Ow5t04C8', output='gtzan.images.zip')

        # Unzip
        shutil.unpack_archive("gtzan.images.zip", "/content/GTZAN_images_original")

        # Delete previous directory with splited dataset if exists
        if os.path.isdir("/content/GTZAN_prepared_dataset"):
            shutil.rmtree("/content/GTZAN_prepared_dataset")

        # Split raw dataset to train, val, test parts
        splitfolders.ratio("/content/GTZAN_images_original", "/content/GTZAN_prepared_dataset", seed=self.split_seed,
                           ratio=(.8, .1, .1))

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.train_dataset = ImageFolder("/content/GTZAN_prepared_dataset/train", transform=self.transformer)
            self.val_dataset = ImageFolder("/content/GTZAN_prepared_dataset/val", transform=self.transformer)
        if stage == "test":
            self.test_dataset = ImageFolder("/content/GTZAN_prepared_dataset/test", transform=self.transformer)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle_datasets
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )
