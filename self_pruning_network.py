import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple

# ==========================================
# Part 1: The "Prunable" Linear Layer
# ==========================================

class PrunableLinear(nn.Module):
    """
    A custom Linear layer that learns to prune itself during training by 
    associating each weight with a learnable gate parameter.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super(PrunableLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Standard weight parameter
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        
        # Standard bias parameter (optional)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
            
        # The learnable gate scores with the exact same shape as the weight tensor.
        # These will be transformed via sigmoid to act as gates between 0 and 1.
        self.gate_scores = nn.Parameter(torch.Tensor(out_features, in_features))
        
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize weights, biases, and gate scores."""
        # Initialize weights and biases using standard PyTorch initialization
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
            
        # Initialize gate scores. We want them to start with a positive value so that
        # the initial gate values (after sigmoid) are close to 1. This ensures that 
        # the network starts with all connections active and slowly learns to prune them.
        # A value of 1.0 gives a sigmoid output of ~0.73.
        nn.init.constant_(self.gate_scores, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass applying the dynamic pruning mechanism.
        """
        # Transform gate scores to gates in the range (0, 1)
        gates = torch.sigmoid(self.gate_scores)
        
        # Calculate pruned weights via element-wise multiplication
        pruned_weights = self.weight * gates
        
        # Perform standard linear operation using the pruned weights
        return F.linear(x, pruned_weights, self.bias)

# ==========================================
# Neural Network Architecture
# ==========================================

class SelfPruningNet(nn.Module):
    """
    A simple feed-forward neural network for CIFAR-10 classification,
    augmented with PrunableLinear layers.
    """
    def __init__(self):
        super(SelfPruningNet, self).__init__()
        # Flatten CIFAR-10 images: 3 channels * 32x32 pixels = 3072 input features
        self.flatten = nn.Flatten()
        
        # Using two prunable hidden layers
        self.fc1 = PrunableLinear(3072, 512)
        self.fc2 = PrunableLinear(512, 256)
        
        # Output layer for 10 classes
        self.fc3 = PrunableLinear(256, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x) # Output logits
        return x

# ==========================================
# Part 2: The Sparsity Regularization Loss
# ==========================================

def calculate_sparsity_loss(model: nn.Module) -> torch.Tensor:
    """
    Calculates the L1 norm of all gate values across all PrunableLinear layers.
    Since gates = sigmoid(gate_scores), all gates are positive, so the L1 norm
    is simply the sum of all gate values.
    """
    sparsity_loss = 0.0
    for module in model.modules():
        if isinstance(module, PrunableLinear):
            # Transform gate scores to actual gate values [0, 1]
            gates = torch.sigmoid(module.gate_scores)
            # Add sum of gates to the total sparsity loss
            sparsity_loss += torch.sum(gates)
    
    # Use zero if no prunable layers are found to keep types consistent
    return sparsity_loss if isinstance(sparsity_loss, torch.Tensor) else torch.tensor(0.0)

def calculate_sparsity_level(model: nn.Module, threshold: float = 1e-2) -> float:
    """
    Calculates the percentage of weights whose corresponding gate value 
    is below the given threshold.
    """
    total_weights = 0
    pruned_weights = 0
    
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, PrunableLinear):
                gates = torch.sigmoid(module.gate_scores)
                
                # Count total weights and weights where the gate is below threshold
                total_weights += gates.numel()
                pruned_weights += (gates < threshold).sum().item()
                
    if total_weights == 0:
        return 0.0
        
    return (pruned_weights / total_weights) * 100.0

# ==========================================
# Part 3: Training and Evaluation Loop
# ==========================================

def train_and_evaluate(lambda_val: float, epochs: int = 5, device: str = 'cpu') -> Tuple[float, float, List[float]]:
    """
    Trains the self-pruning network with a specific lambda value for the sparsity loss.
    Returns the final test accuracy, sparsity level, and the list of all gate values 
    for plotting.
    """
    print(f"\n--- Training with Lambda = {lambda_val} ---")
    
    # 1. Setup Data Loaders for CIFAR-10
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
    
    # 2. Initialize Model, Loss Function, and Optimizer
    model = SelfPruningNet().to(device)
    criterion = nn.CrossEntropyLoss()
    
    # We assign a higher learning rate to the gate scores. 
    # This ensures that the L1 penalty can push the gates to exactly 0 
    # within a small number of epochs, whereas a normal learning rate 
    # might be too slow to drive them down past the threshold.
    gate_params = [p for n, p in model.named_parameters() if 'gate_scores' in n]
    weight_params = [p for n, p in model.named_parameters() if 'gate_scores' not in n]
    
    optimizer = optim.Adam([
        {'params': weight_params, 'lr': 0.001},
        {'params': gate_params, 'lr': 0.05}
    ])
    
    # 3. Training Loop
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            
            # Calculate Standard Classification Loss
            classification_loss = criterion(outputs, labels)
            
            # Calculate Sparsity Regularization Loss
            sparsity_loss = calculate_sparsity_loss(model)
            
            # Total Loss formulation
            total_loss = classification_loss + lambda_val * sparsity_loss
            
            # Backward pass and optimize (updates both weights and gate_scores)
            total_loss.backward()
            optimizer.step()
            
            running_loss += total_loss.item()
            
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {running_loss / len(trainloader):.4f}")
        
    # 4. Evaluation Loop
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    accuracy = 100 * correct / total
    sparsity_level = calculate_sparsity_level(model)
    
    print(f"Final Test Accuracy: {accuracy:.2f}%")
    print(f"Final Sparsity Level: {sparsity_level:.2f}%")
    
    # 5. Collect all gate values for plotting later
    all_gates = []
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, PrunableLinear):
                gates = torch.sigmoid(module.gate_scores).cpu().numpy().flatten()
                all_gates.extend(gates)
                
    return accuracy, sparsity_level, all_gates

def main():
    # To run on GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # We will test three different lambda values to observe the trade-off 
    # between sparsity and accuracy.
    lambdas = [0.0, 0.0001, 0.001]
    
    # Store results for the table
    results = []
    best_gates = None
    best_lambda = None
    highest_sparsity_acceptable_accuracy = 0
    
    for l_val in lambdas:
        # Train for a small number of epochs for demonstration.
        # In a real scenario, this would be higher (e.g., 20-50).
        acc, sparsity, gates = train_and_evaluate(lambda_val=l_val, epochs=10, device=device)
        results.append((l_val, acc, sparsity))
        
        # Keep track of the gate distribution for the "best" pruned model
        # (the one that achieves decent sparsity)
        if sparsity > highest_sparsity_acceptable_accuracy:
            highest_sparsity_acceptable_accuracy = sparsity
            best_gates = gates
            best_lambda = l_val
            
    # Print the final summary table
    print("\n" + "="*50)
    print("FINAL RESULTS SUMMARY")
    print("="*50)
    print(f"{'Lambda':<10} | {'Test Accuracy (%)':<20} | {'Sparsity Level (%)':<20}")
    print("-" * 50)
    for l_val, acc, sparsity in results:
        print(f"{l_val:<10} | {acc:<20.2f} | {sparsity:<20.2f}")
    print("="*50)
    
    # Plot the distribution of the final gate values for the best model
    if best_gates is not None:
        plt.figure(figsize=(10, 6))
        plt.hist(best_gates, bins=50, color='skyblue', edgecolor='black')
        plt.title(rf'Distribution of Final Gate Values ($\lambda$ = {best_lambda})')
        plt.xlabel('Gate Value (Sigmoid Output)')
        plt.ylabel('Frequency (Number of Weights)')
        plt.grid(axis='y', alpha=0.75)
        
        # The expected output is a large spike near 0 (pruned weights)
        # and another cluster away from 0 (important weights).
        plt.savefig('gate_distribution.png')
        print(f"\nSaved gate distribution plot to 'gate_distribution.png'.")

if __name__ == "__main__":
    main()
