import torch.nn as nn
import torch.functional as F

class MyModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass

class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.act_fn = nn.Tanh()
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.act_fn(x)
        x = self.layer2(x)
        return x
    
model = SimpleClassifier(input_dim=2, hidden_dim=4, output_dim=1)
print(model)

for name, param in model.named_parameters():
    print(f"name: {name}, shape: {param.shape}")