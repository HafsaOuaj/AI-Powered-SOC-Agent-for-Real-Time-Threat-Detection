import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import pickle
from data.NetworkLogDataset import NetworkLogDataset
import TabTransformer


def main():

    # Read data
    x_train=pd.read_csv("../../../data/starter/train_val_data/x_train.csv").drop(columns=['Unnamed: 0'])
    x_val=pd.read_csv("../../../data/starter/train_val_data/x_val.csv").drop(columns=['Unnamed: 0'])
    y_train=pd.read_csv("../../../data/starter/train_val_data/y_train.csv").drop(columns=['Unnamed: 0'])
    y_val=pd.read_csv("../../../data/starter/train_val_data/y_val.csv").drop(columns=['Unnamed: 0'])

    with open("../../../resources/data_preprocessing/label_encoders.pkl","rb") as f:
        labels_encoder=pickle.load(f)
    num_col = [col for col in x_train.columns if col not in labels_encoder.keys()]

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Cardinalities for embedding layers
    categorical_cardinates = [len(label_encoder.classes_) for label_encoder in labels_encoder.values()]

    # Dataset & DataLoader
    dataset_train = NetworkLogDataset(
        x_train[list(labels_encoder.keys())].values,
        x_train[num_col].values,
        y_train.values
    )
    dataset_val = NetworkLogDataset(
        x_val[list(labels_encoder.keys())].values,
        x_val[num_col].values,
        y_val.values
    )

    train_loader = DataLoader(dataset_train, batch_size=64, shuffle=True)
    val_loader = DataLoader(dataset_val, batch_size=64)

    # Initialize model
    model = TabTransformer(
        categorical_cardinates,
        num_numerical=x_train[num_col].shape[1]
    ).to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    losses_train=[]
    losses_val=[]
    accuracy_val=[]
    accuracy_train=[]
    correct_train = 0
    total_train = 0
    
    # Training loop
    for epoch in range(10):
        model.train()
        train_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)

        for x_cat, x_num, y_batch in loop:
            x_cat = x_cat.to(device)
            x_num = x_num.to(device)
            y_batch = y_batch.to(device).float().squeeze()

            preds = model(x_cat, x_num).squeeze()
            loss = criterion(preds, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            correct_train += (preds == y_batch).sum().item()
            total_train += y_batch.size(0)
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for x_cat_val, x_num_val, y_val in val_loader:
                x_cat_val = x_cat_val.to(device)
                x_num_val = x_num_val.to(device)
                y_val = y_val.to(device).float().squeeze()

                val_preds = model(x_cat_val, x_num_val).squeeze()
                v_loss = criterion(val_preds, y_val)
                val_loss += v_loss.item()

                predicted = (val_preds > 0.5).float()
                correct += (predicted == y_val).sum().item()
                total += y_val.size(0)

        print(f"Epoch {epoch+1} | Train Loss: {train_loss/len(train_loader):.4f} | "
            f"Val Loss: {val_loss/len(val_loader):.4f} | "
            f"Val Acc: {correct/total:.4f}")
        losses_train=losses_train+[train_loss/len(train_loader)]
        losses_val=losses_val+[val_loss/len(val_loader)]
        accuracy_val=accuracy_val+[correct/total]
        accuracy_train=accuracy_train+[correct_train/total_train]


if __name__=="__main__":
    main()