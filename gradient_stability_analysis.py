import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import sys

# Activations import
from activations import get_activation

sys.setrecursionlimit(5000)

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Veri setini yükleme
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

# Gradyan büyüklüğünü izleme sınıfı
class GradientMonitor:
    def __init__(self, log_frequency=30):
        self.gradients_history = []
        self.log_frequency = log_frequency
        self.batch_count = 0

    def monitor_gradients(self, model):
        if self.batch_count % self.log_frequency == 0:
            gradients_norms = []
            for param in model.parameters():
                if param.grad is not None:
                    gradients_norms.append(param.grad.data.norm().item())
            if gradients_norms:
                self.gradients_history.append(np.mean(gradients_norms))
        self.batch_count += 1

# Model tanımlama fonksiyonu
class DeepModel(nn.Module):
    def __init__(self, activation, num_layers):
        super(DeepModel, self).__init__()
        layers = []
        
        # İlk katman
        layers.append(nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False))
        layers.append(get_activation(activation))
        layers.append(nn.BatchNorm2d(32))

        # Gizli katmanlar
        for i in range(num_layers - 1):
            layers.append(nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False))
            layers.append(get_activation(activation))
            if i % 5 == 0 and i > 0:
                layers.append(nn.MaxPool2d(2, padding=1))
            layers.append(nn.BatchNorm2d(32))

        # Son katmanlar
        layers.append(nn.AdaptiveAvgPool2d(1))
        layers.append(nn.Flatten())
        layers.append(nn.Linear(32, 256))
        layers.append(get_activation(activation))
        layers.append(nn.Dropout(0.5))
        layers.append(nn.Linear(256, 128))
        layers.append(get_activation(activation))
        layers.append(nn.Linear(128, 10))
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def create_deep_model(activation, num_layers):
    return DeepModel(activation, num_layers)

# Katman sayıları ve aktivasyon fonksiyonları için test yapma
layers_to_test = [50]  # Manuel olarak artırılıp azaltılabilir

activation_functions = {
    'LoCLU': 'cahlu',
    'S2LU': 'sslu',
   # 'ReLU': 'relu',
   # 'Leaky ReLU': 'leaky_relu',
   # 'ELU': 'elu',
   # 'GELU': 'gelu',
    'Mish': 'mish',
   # 'Swish': 'swish',
}

results = {}

for num_layers in layers_to_test:
    print(f"\nTesting with {num_layers} layers...")
    results[num_layers] = {}

    for activation_name, activation in activation_functions.items():
        print(f"  Activation: {activation_name}")
        gradient_monitor = GradientMonitor(log_frequency=30)

        # Modeli oluşturun ve eğitin
        model = create_deep_model(activation, num_layers).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Eğitim
        model.train()
        for epoch in range(10):
            running_loss = 0.0
            correct = 0
            total = 0
            
            for batch_idx, (inputs, targets) in enumerate(trainloader):
                inputs, targets = inputs.to(device), targets.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                
                gradient_monitor.monitor_gradients(model)
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            
            if (epoch + 1) % 2 == 0:  # Her 2 epoch'ta bir yazdır
                print(f'Epoch {epoch+1}/10 - Loss: {running_loss/len(trainloader):.4f}, Acc: {100.*correct/total:.2f}%')

        # Test
        model.eval()
        test_loss = 0.0
        correct = 0 
        total = 0
        
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        # Sonuçları sakla
        final_train_acc = 100. * correct / total  # Son epoch train acc için yaklaşık
        final_val_acc = 100. * correct / total
        final_train_loss = running_loss / len(trainloader)
        final_val_loss = test_loss / len(testloader)
        
        results[num_layers][activation_name] = {
            'gradient_history': gradient_monitor.gradients_history,
            'final_train_accuracy': final_train_acc / 100.0,
            'final_val_accuracy': final_val_acc / 100.0,
            'final_train_loss': final_train_loss,
            'final_val_loss': final_val_loss
        }

    # Tek bir grafikte tüm aktivasyon fonksiyonlarını göster
    plt.figure(figsize=(10, 6))
    for activation_name in activation_functions:
        plt.plot(results[num_layers][activation_name]['gradient_history'], label=f'{activation_name}')
    plt.xlabel('Batch')
    plt.ylabel('Gradyan Normları (Ortalama)')
    plt.title(f'{num_layers} Katmanlı Yapı İçin Aktivasyon Fonksiyonlarının Gradyan Normlarının Zamanla Değişimi')
    plt.legend()
    plt.show()

# Sonuçları metin olarak yazdırma
print("\n==== Sonuçlar ====")
for num_layers in layers_to_test:
    print(f"\n{num_layers} Katmanlı Yapı:")
    for activation_name in activation_functions:
        print(f"  Aktivasyon Fonksiyonu: {activation_name}")
        print(f"    Son Eğitim Doğruluğu: {results[num_layers][activation_name]['final_train_accuracy']:.4f}")
        print(f"    Son Doğrulama Doğruluğu: {results[num_layers][activation_name]['final_val_accuracy']:.4f}")
        print(f"    Son Eğitim Kaybı: {results[num_layers][activation_name]['final_train_loss']:.4f}")
        print(f"    Son Doğrulama Kaybı: {results[num_layers][activation_name]['final_val_loss']:.4f}")
        print(f"    Ortalama Gradyan Normu: {np.mean(results[num_layers][activation_name]['gradient_history']):.4f}\n")