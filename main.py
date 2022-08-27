from tokenize import Double
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

from args import parser
from data import VideoDataset
from models import AvgPoolClassifier
from models import RelationModuleMultiScaleWithClassifier

import pandas as pd
import pickle as pkl
import datetime
import numpy as np


if __name__ == '__main__':


    def training_loop(n_epochs, train_loader, model, lr):

        optimizer=optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        total_step=len(train_loader)
        train_loss=[]
        loss_fn=nn.CrossEntropyLoss()

        for epoch in range(n_epochs):
            running_loss=0.0
            for i, batch_images in enumerate(train_loader):
                feat, lbl=batch_images
                # feat, lbl=feat.cuda(), lbl.cuda()

                optimizer.zero_grad()

                out=model(feat)
                loss=loss_fn(out, lbl)

                loss.backward()
                optimizer.step()
                running_loss+=loss.item()

                if ((i)%len(train_loader)==0):
                    print(f"Epoch: {epoch+1}\tLoss:{loss.item()}")
            
            train_loss.append(running_loss/total_step)

        print(f"\ntrain loss: {np.mean(train_loss):.4f}")



        return

    def validate(val_loader, model):
        correct = 0
        total = 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                outputs = model(imgs)
                _, predicted = torch.max(outputs, dim=1)
                total += labels.shape[0]
                correct += int((predicted == labels).sum())

            print(f"Accuracy: {correct / total :.4f}")
        return

    args=parser.parse_args()

    # device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device="cpu"

    train_extracted_features_path=args.source
    train_labels_path=args.train_labels

    val_extracted_feature_path=args.target_features
    val_labels_path=args.val_labels

    ### TRAINING STEP


    with open(train_extracted_features_path, "rb") as f:
        data = pkl.load(f)


    train_features=data["features"][args.modality]
    train_features=torch.from_numpy(train_features).type(torch.float32)

    with open(train_labels_path, "rb") as f:
        train_labels=pkl.load(f)

    df=pd.DataFrame(train_labels)
    train_labels=train_labels['verb_class'].astype(float)

    train_labels=train_labels.to_numpy()
    train_labels=torch.from_numpy(train_labels).type(torch.LongTensor)
    train_labels=train_labels.to(device)
     
    train_data=VideoDataset(train_features, train_labels)
    train_loader=DataLoader(train_data, batch_size=128, shuffle=True)

    num_frames = 5
    num_class = 8
    if args.backbone=='i3d':
        img_features_dim=1024
        model=AvgPoolClassifier(img_features_dim, num_class)
        model.to(device)
    else:
        img_features_dim=2048
        model=RelationModuleMultiScaleWithClassifier(img_features_dim, num_frames, num_class)
        model.to(device)

    # TRAINING
    training_loop(n_epochs=30, train_loader=train_loader, model=model, lr=0.001)

    # TEST
    with open(val_extracted_feature_path, "rb") as f:
        p=pkl.load(f)

    val_input_feat=p['features'][args.modality]
    val_input_feat=torch.from_numpy(val_input_feat).type(torch.float32)

    with open(val_labels_path, "rb") as f:
        p=pkl.load(f)

    df=pd.DataFrame(p)

    val_labels=df['verb_class'].astype(float)
    val_labels=val_labels.to_numpy()
    val_data=VideoDataset(val_input_feat, val_labels)
    val_loader=DataLoader(val_data, batch_size=64)

    # TEST
    validate(val_loader=val_loader, model=model)



