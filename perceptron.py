#single layer perceptron with xor problem
import torch
import torch.nn as nn


device = 'cuda' if torch.cuda.is_available() else 'cpu'
#set seed for make random number
torch.manual_seed(777)

#set tensor input&output
X=torch.FloatTensor([[0,0],[1,0],[0,1],[1,1]]).to(device)
Y=torch.FloatTensor([[0],[1],[1],[0]]).to(device)

#perceptron
linear= nn.Linear(2,1,bias=True)
sigmoid=nn.Sigmoid()
#nn.Sequential: run the functions first to end in turn.
model=nn.Sequential(linear,sigmoid).to(device)

#loss function to use binary classification
criterion=nn.BCELoss().to(device)
#gredient descent(lr: how much move the differential value)
optimizer=torch.optim.SGD(model.parameters(),lr=1)

#train model
epoch=10001
for step in range(epoch):
    optimizer.zero_grad()
    hypothesis=model(X)

    #cost function
    cost=criterion(hypothesis,Y)
    cost.backward()
    optimizer.step()
    
    #print cost
    if step %100==0:
        print(step,cost.item())

#model test
with torch.no_grad():
    hypothesis=model(X)
    predicted=(hypothesis>0.5).float()
    accuracy=(predicted==Y).float().mean()
    print('Hypothesis: ', hypothesis.detach().cpu().numpy())
    print('Predicted: ', predicted.detach().cpu().numpy())
    print('Real value: ', Y.cpu().numpy())
    print('Accuracy: ', accuracy.item())