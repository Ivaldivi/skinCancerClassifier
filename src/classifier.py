import pandas as pd
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, models
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

if __name__ == '__main__':
    image_dir = Path("../MIDAS-dataset")  # folder where all images live
    log_dir = "runs/midas_melanoma"

    writer = SummaryWriter(log_dir=log_dir)
    def find_image(stem):
        # stem = filename without extension
        for ext in [".jpg", ".jpeg", ".png"]:
            candidate = image_dir / f"{stem}{ext}"
            if candidate.exists():
                return candidate
        return None

    def has_any_image(row):
        stem = Path(row["midas_file_name"]).stem
        for ext in [".jpg", ".jpeg", ".png"]:
            if (image_dir / f"{stem}{ext}").exists():
                return True
        return False

    # ---- 1. Load your metadata ----
    df = pd.read_excel("../release_midas.xlsx")  # your xlsx file

    # Keep only rows with valid filenames and labels
    df = df.dropna(subset=["midas_file_name", "midas_melanoma"])

    # Change labels to ints (0/no 1/yes)
    df['midas_melanoma'] = df['midas_melanoma'].map({'no':0,'yes':1})

    # Drop rows where there isn't an image
    df = df[df.apply(has_any_image, axis=1)].reset_index(drop=True)
    df['image_stem'] = df['midas_file_name'].apply(lambda x: Path(x).stem)

    print(f"Valid images: {len(df)} / {len(df_original)}")
    # ---- 2. Dataset class ----
    class MelanomaDataset(Dataset):
        def __init__(self, df, image_dir, transform=None):
            self.df = df.reset_index(drop=True)
            self.image_dir = image_dir
            self.transform = transform

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            # Use PRECOMPUTED stem (much faster than Path.stem() every time)
            stem = row["image_stem"]  
            
            # Try common extensions in order
            for ext in [".jpg", ".jpeg", ".png"]:
                img_path = self.image_dir / f"{stem}{ext}"
                if img_path.exists():
                    image = Image.open(img_path).convert("RGB")
                    if self.transform:
                        image = self.transform(image)
                    label = row["midas_melanoma"]
                    print(label, img_path)
                    return image, label
    
            raise FileNotFoundError(f"No image found for '{stem}'")

        
    # ---- 3. Transforms & splits ----
    transform = transforms.Compose([
        transforms.Resize((128, 128)), # TODO: Add more augmentations.

        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])

    dataset = MelanomaDataset(df, image_dir, transform)

    # Stratified split so melanoma/non-melanoma ratios are similar
    train_idx, val_idx = train_test_split(
        range(len(dataset)),
        test_size=0.2,
        random_state=42,
        stratify=df["midas_melanoma"]
    )

    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)

    train_loader = DataLoader(
        train_ds, 
        batch_size=32, 
        shuffle=True, 
        num_workers=0
    )
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)

    # ---- 4. Model: ResNet18 for binary classification ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 1)   # one logit for binary.
    model = torch.compile(model)

    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()   
    optimizer = optim.Adam(model.parameters(), lr=1e-4) 

    # ---- 5. Training loop ----
    for epoch in range(10): 
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)  # shape (B,1)

            optimizer.zero_grad()
            logits = model(images)                   
            loss = criterion(logits, labels)        
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_ds)

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                logits = model(images)
                probs = torch.sigmoid(logits).squeeze(1)
                preds = (probs > 0.5).long()

                correct += (preds == labels).sum().item()
                total += labels.size(0)

        # TensorBoard Metrics
        val_acc = correct / total
        # Training error rate
        writer.add_scalar("Loss/train", train_loss, epoch)

        # Validation Accurary 
        writer.add_scalar("Accuracy/val", val_acc, epoch)

        print(f"Epoch {epoch+1}, train loss {train_loss:.4f}, val acc {val_acc:.4f}")

    torch.save(model.state_dict(), 'midas_melanoma_final.pth')

    writer.close()