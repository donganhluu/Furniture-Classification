import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

def main():

    
    # CONFIG
    
    train_dir = r"D:/Furniture_classification/dataset_split/train"
    val_dir   = r"D:/Furniture_classification/dataset_split/val"
    test_dir  = r"D:/Furniture_classification/dataset_split/test"

    output_dir = r"D:/Furniture_classification/CNN"
    os.makedirs(output_dir, exist_ok=True)

    best_model_path = os.path.join(output_dir, "best_cnn.pth")
    checkpoint_path  = os.path.join(output_dir, "cnn_checkpoint.pth")

    num_classes = 4
    batch_size = 32
    epochs = 10
    lr = 1e-4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ======================
    # TRANSFORM
    # ======================
    train_tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(0.2, 0.2),
        transforms.ToTensor(),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    test_tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    train_data = datasets.ImageFolder(train_dir, transform=train_tf)
    val_data   = datasets.ImageFolder(val_dir, transform=val_tf)
    test_data  = datasets.ImageFolder(test_dir, transform=test_tf)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # ======================
    # CNN MODEL
    # ======================
    class SimpleCNN(nn.Module):
        def __init__(self, num_classes=4):
            super(SimpleCNN, self).__init__()

            self.features = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),

                nn.Conv2d(16, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),

                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )

            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64 * 32 * 32, 128),
                nn.ReLU(),
                nn.Linear(128, num_classes)
            )

        def forward(self, x):
            x = self.features(x)
            x = self.classifier(x)
            return x

    model = SimpleCNN(num_classes).to(device)

    # ======================
    # LOSS + OPTIMIZER
    # ======================
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # ======================
    # RESUME CHECKPOINT
    # ======================
    start_epoch = 0
    best_acc = 0.0

    train_loss_list = []
    val_loss_list = []
    train_acc_list = []
    val_acc_list = []

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)

        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])

        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['best_acc']

        train_loss_list = checkpoint['train_loss']
        val_loss_list = checkpoint['val_loss']
        train_acc_list = checkpoint['train_acc']
        val_acc_list = checkpoint['val_acc']

        print(f"Resume from epoch {start_epoch}")

   
    # TRAINING LOOP
   
    for epoch in range(start_epoch, epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")

        # TRAIN
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for x, y in tqdm(train_loader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x.size(0)
            preds = torch.argmax(out, 1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        train_loss /= total
        train_acc = correct / total

        # VALIDATION
        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)

                out = model(x)
                loss = criterion(out, y)

                val_loss += loss.item() * x.size(0)
                preds = torch.argmax(out, 1)

                correct += (preds == y).sum().item()
                total += y.size(0)

        val_loss /= total
        val_acc = correct / total

        # SAVE BEST MODEL
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print("Best model saved")

        # LOG
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)

        print(f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        # SAVE CHECKPOINT
        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'best_acc': best_acc,
            'train_loss': train_loss_list,
            'val_loss': val_loss_list,
            'train_acc': train_acc_list,
            'val_acc': val_acc_list
        }, checkpoint_path)

        print("Checkpoint saved")

    # TEST 
   
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)

            outputs = model(x)
            preds = torch.argmax(outputs, 1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(y.numpy())

    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')

    print("\nTEST METRICS")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-score  : {f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=test_data.classes))

   
    # PLOT
   
    epochs_range = range(1, len(train_loss_list) + 1)

    plt.figure()
    plt.plot(epochs_range, train_loss_list, label="Train Loss")
    plt.plot(epochs_range, val_loss_list, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, "loss.png"))
    plt.show()

    plt.figure()
    plt.plot(epochs_range, train_acc_list, label="Train Acc")
    plt.plot(epochs_range, val_acc_list, label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, "accuracy.png"))
    plt.show()

    print("\nBest Val Acc:", best_acc)


if __name__ == "__main__":
    main()