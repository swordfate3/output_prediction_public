import torch
from models.iTransformer import CipherITransformer

model = CipherITransformer(input_dim=16, hidden_dim=64, num_layers=2,
                           num_heads=4, output_dim=16, dropout=0.1)
x1 = torch.randint(0, 2, (8, 1, 16)).float()
x2 = torch.randint(0, 2, (8, 16)).float()
y1 = model.eval()(x1)
y2 = model.eval()(x2)
print(y1.shape, y2.shape)