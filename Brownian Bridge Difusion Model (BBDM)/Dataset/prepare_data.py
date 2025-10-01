from torch.utils.data import DataLoader
from splitter import Edges2ShoesSplitter
from datasets import DatasetConfig, CustomAlignedDataset

# Split dataset
splitter = Edges2ShoesSplitter("train", "train_split")
splitter.split_and_save()

# Create dataset and DataLoader
cfg = DatasetConfig(dataset_path="train_split", image_size=64)
train_dataset = CustomAlignedDataset(cfg, stage='train')
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Test a batch
for img_target, img_cond in train_loader:
    print(img_target.shape, img_cond.shape)  
    break
