import torch
from model.model import LeNet5
from data.dataset import get_mnist_data
from model.loss import TrainingConfig

def train_epoch(model, train_loader, config, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        config.optimizer.zero_grad()  # Zero gradients
        outputs = model(inputs)
        loss = config.compute_loss(outputs, targets)
        loss.backward()  # Backpropagation
        config.optimizer.step()  # Update weights

        running_loss += loss.item()
        _, predicted = outputs.max(1)  # Get predictions

        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return running_loss / len(train_loader), 100. * correct / total

def validate(model, val_loader, config, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = config.compute_loss(outputs, targets)

            running_loss += loss.item()
            _, predicted = outputs.max(1)  # Get predictions
            
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return running_loss / len(val_loader), 100. * correct / total

def train():
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    model = LeNet5().to(device)
    
    # Get data loaders
    train_loader, val_loader, test_loader = get_mnist_data()
    
    # Setup training configuration
    config = TrainingConfig(model)
    
    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, config, device)
        val_loss, val_acc = validate(model, val_loader, config, device)
        
        print(f'Epoch: {epoch+1}/{num_epochs}')
        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
        print('-' * 50)

    # Save the trained model weights
    torch.save(model.state_dict(), './../weights/lenet5.pth')


