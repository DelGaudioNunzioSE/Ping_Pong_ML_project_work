import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from utilities.action_space import ActionSpaceArm
from utilities.arm_net import ArmModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class EarlyStopping:
    def __init__(self, monitor='val_loss', patience=20, restore_best_weights=True):
        self.monitor = monitor
        self.patience = patience
        self.restore_best_weights = restore_best_weights
        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0
        self.best_model_wts = None
        self.early_stop = False

    def __call__(self, current_val_loss, model):
        if current_val_loss < self.best_val_loss:
            self.best_val_loss = current_val_loss
            self.epochs_no_improve = 0
            if self.restore_best_weights:
                self.best_model_wts = model.state_dict()
        else:
            self.epochs_no_improve += 1

        if self.epochs_no_improve >= self.patience:
            self.early_stop = True
            if self.restore_best_weights and self.best_model_wts is not None:
                model.load_state_dict(self.best_model_wts)

class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MultiLayerPerceptron, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.relu3 = nn.ReLU()
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.layer2(x)
        x = self.relu2(x)
        x = self.layer3(x)
        x = self.relu3(x)
        x = self.output_layer(x)
        return x


df = pd.read_csv('dataset_file.csv')

# Separare le caratteristiche dagli obiettivi
Y = df.iloc[:, :-2]
X = df.iloc[:, 4:6]

seed = 14

# Suddividi il dataset in set di addestramento e di subtest
X_train, X_sub_test, Y_train, Y_sub_test = train_test_split(X, Y, test_size=0.5, random_state=seed)

# Suddividi il dataset in set di test e di test
X_test, X_val, Y_test, Y_val = train_test_split(X_sub_test, Y_sub_test, test_size=0.5, random_state=seed)

X_train_np = X_train.to_numpy()
X_test_np = X_test.to_numpy()
X_val_np = X_val.to_numpy()

Y_train_np = Y_train.to_numpy()
Y_test_np = Y_test.to_numpy()
Y_val_np = Y_val.to_numpy()

X_train = torch.tensor(X_train_np, dtype=torch.float32)
X_test = torch.tensor(X_test_np, dtype=torch.float32)
X_val = torch.tensor(X_val_np, dtype=torch.float32)

Y_train = torch.tensor(Y_train_np, dtype=torch.float32)
Y_test = torch.tensor(Y_test_np, dtype=torch.float32)
Y_val = torch.tensor(Y_val_np, dtype=torch.float32)

train_dataset = TensorDataset(X_train, Y_train)
test_dataset = TensorDataset(X_test, Y_test)
validation_dataset = TensorDataset(X_val, Y_val)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
val_loader = DataLoader(validation_dataset, batch_size=64, shuffle=False)

# input_dim = X_train.shape[1]
# hidden_dim = 256  # Numero di neuroni nei layer nascosti
# output_dim = Y_train.shape[1]

input_dim = 2
hidden_dim = (300, 300, 300)
output_dim = ActionSpaceArm()


model = ArmModel(hidden_dim, input_dim, output_dim).to(device)

lossFunction = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 1000
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for X_batch, Y_batch in train_loader:
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = lossFunction(outputs, Y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * X_batch.size(0)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_batch, Y_batch in val_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            outputs = model(X_batch)
            loss = lossFunction(outputs, Y_batch)
            val_loss += loss.item() * X_batch.size(0)

    running_loss /= len(train_loader.dataset)
    val_loss /= len(val_loader.dataset)

    print(f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {running_loss:.4f}, Validation Loss: {val_loss:.4f}')

    # Controlla se si deve applicare l'early stopping
    early_stopping(val_loss, model)

    if early_stopping.early_stop:
        print('Early stopping triggered')
        break

print('Training finished.')

torch.save(model.state_dict(), 'saved_model6.pth')