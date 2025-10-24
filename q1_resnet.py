import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
from tqdm import tqdm

RANDOM_SEED = 12345
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

class ResNetBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResNetBlock, self).__init__()
        
        # TODO: create the ResNetBlock as per the provided specification (use padding=1, bias=False for conv)
        # Hint: Lookup nn.Conv2d, nn.BatchNorm2d, and nn.ReLU
        self.downsample = downsample
        
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.final_relu = nn.ReLU()

    def forward(self, x):
        identity = x
        out = None
        
        # TODO: fill the forward based on the provided specification
        out = self.model(x)
        
        # keep this as is
        if self.downsample is not None:
            identity = self.downsample(x)
    
        out += identity
        out = self.final_relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, layers, num_classes=10):
        super(ResNet, self).__init__()
        
        # layer A
        # TODO: initialise the following correctly based on the provided specification 
        self.layerA = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
        )

        # layer B
        self.layerB = self.make_resnet_layer(in_channels=64, out_channels=64, blocks=layers[0], stride=1)
        
        # layer C
        self.layerC = self.make_resnet_layer(in_channels=64, out_channels=128, blocks=layers[1], stride=2)
        
        # layer D
        self.layerD = self.make_resnet_layer(in_channels=128, out_channels=256, blocks=layers[2], stride=2)
        
        # layer E
        self.layerE = self.make_resnet_layer(in_channels=256, out_channels=512, blocks=layers[3], stride=2)
        
        # final pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

        self.model = nn.Sequential(
            self.layerA,
            self.layerB,
            self.layerC,
            self.layerD,
            self.layerE,
            self.avgpool,
            nn.Flatten(),
            self.fc,
        )

    def make_resnet_layer(self, in_channels, out_channels, blocks, stride):
        downsample = None
        
        if stride != 1 or in_channels != out_channels:
            # i.e. the dimensions of F(x) and x are different, i.e. skip connection can't be a simple addition
            # TODO: fill the downsample with the apt convolution and batchnorm
            # Hint: lookup nn.Sequential
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        
        resnet_blocks_list = [] # list of ResNetBlock blocks that comprise this layer
        
        # consider the first block (special case because it could have stride, downsample)
        # TODO: initialise and append the correct ResNetBlock
        resnet_blocks_list.append(ResNetBlock(in_channels, out_channels, stride=stride, downsample=downsample))
        
        # TODO: initialise and append the correct ResNetBlock based on the provided specification
        for _ in range(1, blocks):
            resnet_blocks_list.append(ResNetBlock(out_channels, out_channels))
            
        return nn.Sequential(*resnet_blocks_list)

    def forward(self, x):
        # TODO: fill the forward based on the provided specification
        return self.model(x)

def resnet18(num_classes=10):
  return ResNet([2, 2, 2, 2], num_classes)

def get_metrics(model, data_loader, device, criterion):
    # set model to eval
    model.eval()
    # TODO: Implement the function to calculate accuracy, average batch loss, precision, recall, and f1 score
    # Note: Use precision_score, recall_score, f1_score functions from sklearn.metrics to calculate metrics
    # https://scikit-learn.org/1.5/modules/generated/sklearn.metrics.f1_score.html
    # set average='weighted' for the sklearn functions
    tot_num_batches = 0
    tot_data = 0
    tot_batch_loss = 0
    tot_correct = 0

    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc="eval"):   
            inputs, labels = inputs.to(device), labels.to(device)
            logits = model(inputs)
            loss = criterion(logits, labels)

            tot_batch_loss += loss.item()
            tot_num_batches += 1
            tot_data += inputs.shape[0]
            preds = torch.argmax(logits, dim=-1)
            tot_correct += torch.sum(preds == labels).item()

            y_true += labels.cpu().tolist()
            y_pred += preds.cpu().tolist()
    
    accuracy = tot_correct / tot_data
    avg_batch_loss = tot_batch_loss / tot_num_batches
    f1 = f1_score(y_true, y_pred, average="weighted")
    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")
    
    # Note: use the torch.no_grad() to create a context where grads of the operations aren't kept track of
    # Return: (accuracy, avg_batch_loss, f1_score, precision, recall)
    return accuracy, avg_batch_loss, f1, precision, recall

def train(momentum: float = 0):
    # Load the CIFAR10 dataset 
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], 
                                                                                std=[0.2023, 0.1994, 0.2010])])
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Create data loaders
    batch_size = 256
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Assign the device to be used for the compute
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.mps.is_available():
        device = "mps"
    
    # Create the model
    model = resnet18(num_classes=10).to(device)

    # Define loss function and optimizer
    # TODO: change momentum in the optimizer as asked
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=0.001, momentum=momentum)
    
    # Train the model
    num_epochs = 20
    tr_loss_trajectory = []
    tr_acc_trajectory = []
    te_loss_trajectory = []
    te_acc_trajectory = []
    
    # Loop over the epochs
    for epoch in range(num_epochs):
        # set model to train
        model.train()

        # Define variables to compute the average batch loss + accuracy for the train dataset
        tot_num_batches = 0
        tot_data = 0
        tot_batch_loss = 0
        tot_correct = 0
        
        # Iterate over the loader
        for inputs, labels in tqdm(train_loader):
            
            # Shift to the gpu
            inputs, labels = inputs.to(device), labels.to(device)
            
            # TODO: fill in missing code here to complete the training logic
            # Note: remember to update the accumulator variables
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            tot_num_batches += 1
            tot_data += inputs.shape[0]
            tot_batch_loss += loss.item()
            preds = torch.argmax(logits, dim=-1)
            tot_correct += torch.sum(preds == labels).item()
        
        # Compute the average batch loss + accuracy for the train dataset
        avg_batch_loss = tot_batch_loss / tot_num_batches
        avg_batch_acc = tot_correct / tot_data
        print(f'Epoch {epoch+1} Loss: {avg_batch_loss} Acc: {avg_batch_acc}')
        
        # Compute the average batch loss + accuracy for the test dataset
        avg_te_acc, avg_te_batch_loss, _, _, _ = get_metrics(model, test_loader, device, criterion)
        print(f'Epoch {epoch+1} Test Loss: {avg_te_batch_loss} Acc: {avg_te_acc}')    

        # TODO: Modify the trajectories
        tr_loss_trajectory.append(avg_batch_loss)
        tr_acc_trajectory.append(avg_batch_acc)
        te_loss_trajectory.append(avg_te_batch_loss)
        te_acc_trajectory.append(avg_te_acc)
    
    avg_te_acc, avg_te_batch_loss, f1, precision, recall = get_metrics(model, test_loader, device, criterion)
    print(f'Test Loss: {avg_te_batch_loss} Acc: {avg_te_acc} Precision: {precision} Recall: {recall} F1 Score: {f1}')

    # TODO: Save the trajectories (so as to be plotted later)
    return {
        "train_loss": tr_loss_trajectory,
        "train_acc": tr_acc_trajectory,
        "test_loss": te_loss_trajectory,
        "test_acc": te_acc_trajectory,
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
    }


if __name__ == "__main__":
    train()
