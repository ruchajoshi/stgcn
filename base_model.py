from data_utils import PeMSD7M, describe_data, load_data
from model import SpatioTemporalConv
from model_updated import STGCN

from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import torch.nn as nn
import torch
from torch.utils.tensorboard import SummaryWriter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_epochs = 200
batch_size = 12
K_t = 3
K_s = 1
c = [1, 64, 16, 64]
num_nodes = 228
num_edges = 51756
num_graphs = 12

dataset = PeMSD7M(root='./data/PeMS-M/')

train, val, test = load_data(dataset, batch_size)
print('Data Loading Complete!')

writer = SummaryWriter()
model = STGCN(K_t, K_s, c, num_nodes, num_edges, num_graphs).to(device)
# model = SpatioTemporalConv(3, 1, 1, 1).to(device)
# writer.add_graph(model, train)

criterion = nn.MSELoss(reduction='sum')
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)


for epoch in range(num_epochs):
    # training
    for batch in train:
        labels = batch.label
        labels = labels.to(device)
        batch = batch.to(device)

        # forward pass
        outputs = model(batch)
        train_loss = criterion(outputs, labels)
        writer.add_scalar('Training Loss', train_loss, epoch)
        # backward, optimize
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch+1}]/[{num_epochs}], Training Loss:{train_loss:.4f}')

    # validation
    with torch.no_grad():
        for i, batch in enumerate(val):
            labels = batch.label
            labels = labels.to(device)
            batch = batch.to(device)

            outputs = model(batch)
            val_loss = criterion(outputs, labels)
            val_loss = torch.sqrt(val_loss)
            writer.add_scalar('Validation Loss', val_loss, epoch)
            # print(outputs[100], labels[100])
        print(f'Epoch [{epoch+1}]/[{num_epochs}], Validation Loss:{val_loss:.4f}')

    scheduler.step()

print('Training Complete!')

# with torch.no_grad():
#     n_correct = 0
#     n_samples = 0
#     for batch in test:
#         labels = batch.label
#         labels = labels.to(device)
#         batch = batch.to(device)

#         # forward pass
#         outputs = model(batch)
#         n_correct += torch.eq(outputs, labels).sum()
#         n_samples += outputs.shape[0]
#     acc = 100.0 * n_correct/n_samples
#     print(f'Test Accuracy: {acc}%')

with torch.no_grad():
        for batch in test:
            labels = batch.label
            labels = labels.to(device)
            batch = batch.to(device)

            outputs = model(batch)
            test_loss = criterion(outputs, labels)
            test_loss = torch.sqrt(test_loss)
        print(f'Epoch [{epoch+1}]/[{num_epochs}], Test Loss:{test_loss:.4f}')
print('Test Finished!')
