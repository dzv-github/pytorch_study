import torch
import torch.nn as nn
from sklearn import datasets
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

onehot=OneHotEncoder()

device='cuda' if torch.cuda.is_available() else 'cpu'
iris = datasets.load_iris()

#iris dataset load
data1 = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['target'])

#iris dataset to dataframe
temp_data=pd.DataFrame()
temp_data['ordinal']=data1['target']

#one hot on result
temp_data=pd.DataFrame(onehot.fit_transform(temp_data).toarray())
d=pd.get_dummies(temp_data)

#set X,y
X=torch.FloatTensor(data1.iloc[:, 0:4].values).to(device)
y=torch.FloatTensor(d.iloc[:, 0:4].values).to(device)

#set loss func
criterion=nn.CrossEntropyLoss().to(device)

#epoch parameter
epoch=10001

#set model architecture
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
    nn.Softmax(),
).to(device)

#set optimizer
optimizer=torch.optim.SGD(model.parameters(),lr=0.01)

#learning
for step in range(epoch):
    optimizer.zero_grad()
    hypothesis=model(X)

    cost=criterion(hypothesis,y)
    cost.backward()
    optimizer.step()

    if step%1000==0:
        print(step,cost.item())

#result output
with torch.no_grad():
    hypothesis=model(X)
    predicted=(torch.argmax(hypothesis,dim=1)).float()
    accuracy=(predicted==torch.argmax(y,dim=1)).float().mean()
    print('Hypothesis: ', hypothesis)
    print('Predicted: ', predicted)
    print('Real value: ', torch.argmax(y,dim=1).cpu().numpy())
    print('Accuracy: ', accuracy.item()*100,'%')
