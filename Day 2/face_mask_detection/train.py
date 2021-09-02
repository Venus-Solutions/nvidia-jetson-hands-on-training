import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split, DataLoader
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from network import Net

# pip install ipywidgets
from tqdm.notebook import tqdm

from sklearn.metrics import confusion_matrix, classification_report

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_class_distribution(object_dataset, idx2class):
    dict_counter = {k:0 for k, v in object_dataset.class_to_idx.items()}

    for element in object_dataset:
        y_label = element[1]
        y_label = idx2class[y_label]
        dict_counter[y_label] += 1

    return dict_counter

def train(dataset):
    train_dataset, val_dataset = random_split(dataset, (6000, 1553))

    train_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=4, num_workers=2)
    val_loader = DataLoader(dataset=val_dataset, shuffle=False, batch_size=4, num_workers=2)

    data_iter = iter(train_loader)
    images, labels = data_iter.next()

    imshow(torchvision.utils.make_grid(images))

    classes = ("with_mask", "without_mask")
    number_of_random_pics = 4 # equal batch size
    print(' '.join('%5s' % classes[labels[i]] for i in range(number_of_random_pics)))

    net = Net()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    num_steps = 0
    min_loss = 1e+10

    epochs = 2 # for test

    for epoch in tqdm(range(1, epochs + 1), total=epochs, desc="Training"):
        running_loss = []
        
        net.train()

        for images, labels in train_loader:
            images, labels = images, labels
            num_steps += 1

            # False Positive (FP)
            outs = net(images)
            loss = criterion(outs, labels)

            # Logging the loss value
            running_loss.append(loss.item())

            # BP
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_loss = sum(running_loss) / len(running_loss)
        acc = evaluate(net, train_loader)

        print("Accuracy: ", acc)
        print("Loss: ", epoch_loss)

        # Model checkpoint
        if epoch_loss < min_loss:
            min_loss = epoch_loss
            best_model = net.state_dict()
    
    torch.save(best_model, "./{0}_{1:0.4f}.pth".format("classifier", min_loss))
    print("Training finished && model saved!")

    # Testing
    img_test, test = next(iter(val_loader))

    pred = net(img_test)
    pred_test = torch.argmax(pred, axis=1)

    print("Predicted: ", [i for i in pred_test])
    print("Actual: ", [i for i in test])

    for i in pred_test:
        if i == 0:
            print("without_mask")
        else:
            print("with_mask")
    
    print("=============================")

    for i in test:
        if i == 0:
            print("without_mask")
        else:
            print("with_mask")
    
    classes = ('With Mask', 'Without Mask')
    
    pred_test = pred_test.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    test = test.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    y_pred = pred_test.cpu().numpy()
    y_true = test.cpu().numpy()
    matrix = confusion_matrix(y_true, y_pred)
    print(matrix)

    df_cm = pd.DataFrame(matrix/np.sum(matrix) * 2, index = [i for i in classes],
                     columns = [i for i in classes])
    plt.figure(figsize = (12,7))
    sns.heatmap(df_cm, annot=True)
    plt.savefig('confusion_matrix_output.png')

    print("=============================")

    report = classification_report(y_true, y_pred, target_names=['with_mask','without_mask'])
    print(report)

    return None

# Convert output probabilities to predicted class
def images_to_probs(net, images):
    output = net(images)
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]

def evaluate(net, dataloader):
    correct, total = 0, 0
    with torch.no_grad():
        net.eval()
        for images, labels in dataloader:
            images, labels = images, labels.numpy()

            preds, probs = images_to_probs(net, images)

            total += len(labels)
            correct += (preds == labels).sum()
    return correct/total * 100    

def imshow(img):
    img = img / 2 + 0.5
    img_np = img.numpy()
    plt.imshow(np.transpose(img_np, (1, 2, 0)))
    plt.show()

def main():
    image_transforms = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    path_file = './data'

    dataset = ImageFolder(root=path_file, transform=image_transforms)

    dataset.class_to_idx = {'with_mask':1, 'without_mask': 0}

    idx2class = {v: k for k, v in dataset.class_to_idx.items()}

    print("Distribution of classes: ", get_class_distribution(dataset, idx2class))

    plt.figure(figsize=(15, 8))
    sns.barplot(data = pd.DataFrame.from_dict([get_class_distribution(dataset, idx2class)]).melt(), x= "variable", y="value", hue="variable").set_title("Face Mask Class Distribution")
    plt.show()

    return dataset


if __name__ == "__main__":
    dataset = main()
    train(dataset)