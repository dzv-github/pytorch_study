import torch
import torch.nn as nn
from sklearn import datasets
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

onehot=OneHotEncoder()

device='cuda' if torch.cuda.is_available() else 'cpu'
iris = datasets.load_iris()

data1 = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['target'])

temp_data=pd.DataFrame()
temp_data['ordinal']=data1['target']
temp_data=pd.DataFrame(onehot.fit_transform(temp_data).toarray())

d=pd.get_dummies(temp_data)
X=torch.FloatTensor(data1.iloc[:, 0:4].values).to(device)

y=torch.FloatTensor(d.iloc[:, 0:4].values).to(device)

softmax=nn.Softmax()
criterion=nn.CrossEntropyLoss().to(device)
epoch=10001

model=nn.Sequential(
    nn.Linear(4,20,bias=True),
    nn.ReLU(),
    nn.Linear(20,20,bias=True),
    nn.ReLU(),
    nn.Linear(20,20,bias=True),
    nn.ReLU(),
    nn.Linear(20,20,bias=True),
    nn.ReLU(),
    nn.Linear(20,20,bias=True),
    nn.ReLU(),
    nn.Linear(20,20,bias=True),
    nn.ReLU(),
    nn.Linear(20,20,bias=True),
    nn.ReLU(),
    nn.Linear(20,3,bias=True),
    softmax,
).to(device)

optimizer=torch.optim.SGD(model.parameters(),lr=0.01)
for step in range(epoch):
    optimizer.zero_grad()
    hypothesis=model(X)

    cost=criterion(hypothesis,y)
    cost.backward()
    optimizer.step()

    if step%1000==0:
        print(step,cost.item())


with torch.no_grad():
    hypothesis=model(X)
    predicted=(torch.argmax(hypothesis,dim=1)).float()
    accuracy=(predicted==torch.argmax(y,dim=1)).float().mean()
    print('Hypothesis: ', hypothesis)
    print('Predicted: ', predicted)
    print('Real value: ', torch.argmax(y,dim=1).cpu().numpy())
    print('Accuracy: ', accuracy.item()*100,'%')
