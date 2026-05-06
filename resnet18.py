import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

def main():

    # ======================
    # CONFIG
    # ======================
    train_dir = r"D:/Furniture_classification/dataset_split/train"
    val_dir   = r"D:/Furniture_classification/dataset_split/val"
    test_dir  = r"D:/Furniture_classification/dataset_split/test"  

    output_dir = r"D:/Furniture_classification/ResNet18_full"
    os.makedirs(output_dir, exist_ok=True)

    best_model_path = os.path.join(output_dir, "best_resnet18.pth")
    checkpoint_path = os.path.join(output_dir, "resnet18_checkpoint.pth")

    num_classes = 4
    batch_size = 32
    epochs = 10
    lr = 1e-5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

   
    # TRANSFORM
  
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # ======================
    # DATA
    # ======================
    train_data = datasets.ImageFolder(train_dir, transform=train_tf)
    val_data   = datasets.ImageFolder(val_dir, transform=val_tf)
    test_data  = datasets.ImageFolder(test_dir, transform=val_tf)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_data, batch_size=batch_size, shuffle=False)

  
    # MODEL
    
    model = models.resnet18(pretrained=True)

    for param in model.parameters():
        param.requires_grad = True

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    model = model.to(device)

 
    # LOSS + OPTIMIZER

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)


    # RESUME
    
    start_epoch = 0
    best_acc = 0.0

    train_loss_list, val_loss_list = [], []
    train_acc_list, val_acc_list = [], []

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)

        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])

        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['best_acc']

        train_loss_list = checkpoint['train_loss']
        val_loss_list   = checkpoint['val_loss']
        train_acc_list  = checkpoint['train_acc']
        val_acc_list    = checkpoint['val_acc']

        print(f"🔄 Resume from epoch {start_epoch}")

   
    # TRAIN LOOP
   
    for epoch in range(start_epoch, epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")

        # TRAIN 
        model.train()
        train_loss, correct, total = 0, 0, 0

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
        val_loss, correct, total = 0, 0, 0

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

        #  SAVE BEST 
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(" Best model saved")

        #  LOG 
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

        print(" checkpoint saved")
        
    # PLOT
    
    epochs_range = range(1, len(train_loss_list) + 1)

    plt.figure()
    plt.plot(epochs_range, train_loss_list, label="Train Loss")
    plt.plot(epochs_range, val_loss_list, label="Val Loss")
    plt.legend(); plt.grid()
    plt.savefig(os.path.join(output_dir, "loss.png"))

    plt.figure()
    plt.plot(epochs_range, train_acc_list, label="Train Acc")
    plt.plot(epochs_range, val_acc_list, label="Val Acc")
    plt.legend(); plt.grid()
    plt.savefig(os.path.join(output_dir, "accuracy.png"))

    
    # TEST
    
    print("\n Test best model...")

    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    y_true, y_pred = [], []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)

            out = model(x)
            preds = torch.argmax(out, 1).cpu().numpy()

            y_pred.extend(preds)
            y_true.extend(y.numpy())

    # METRICS

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')

    print("\n TEST METRICS")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print("\n Classification Report:")
    report = classification_report(y_true, y_pred, target_names=test_data.classes)
    print(report)

    # SAVE METRICS
 
    with open(os.path.join(output_dir, "test_metrics.txt"), "w") as f:
        f.write(f"Accuracy : {acc:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall   : {recall:.4f}\n")
        f.write(f"F1 Score : {f1:.4f}\n\n")
        f.write(report)

    print("\n Best Val Acc:", best_acc)


if __name__ == "__main__":
    main()