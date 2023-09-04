
# Performing the required imports

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns

np.random.seed(9)

device="cuda" if torch.cuda.is_available() else "cpu"
print("The device is {0}".format(device))

# Extracting the data
training_set=torchvision.datasets.CIFAR10(root="./data",train=True,download=True,
                                          transform=transforms.ToTensor())
testing_set=torchvision.datasets.CIFAR10(root="./data",train=False,download=True,
                                          transform=transforms.ToTensor())

# Loading the data (Data Loader)

train_loader=torch.utils.data.DataLoader(training_set,batch_size=70,shuffle=True
                                         ,num_workers=2,pin_memory=True,
                                         drop_last=True)
test_loader=torch.utils.data.DataLoader(testing_set,batch_size=1,shuffle=False)

num_of_pixels=32*32*3
dropout=nn.Dropout(0.3)

# Defining a modular neural network

class Neural_Net(nn.Module):
  def __init__(self):
    super(Neural_Net,self).__init__()
    self.hidden1=nn.Linear(num_of_pixels,256)
    self.hidden2=nn.Linear(256,128)
    self.output=nn.Linear(128,10)

  def forward(self,x):
    x=F.relu(self.hidden1(x))
    x=dropout(x)
    x=F.relu(self.hidden2(x))
    x=dropout(x)
    x=F.softmax(self.output(x))
    return x

model=Neural_Net()

loss_function=nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(model.parameters(),lr=0.01,weight_decay=0.0001,
                          momentum=0.9)

# Verifying the model

print(model)

# Training the model

num_of_epochs=400
count=0
for epoch in range(num_of_epochs):
  correct=0
  for images, labels in train_loader:
    count+=1
    input_image=images.view(-1,num_of_pixels)
    outputs=model(input_image)
    loss=loss_function(outputs,labels)
    # Back prop
    optimizer.zero_grad()
    loss.backward()
    # Update weights (optimize)
    optimizer.step()
    # Evaluating the performance
    predictions=torch.max(outputs,1)[1]
    correct+=(predictions==labels).sum().numpy()

  if((epoch)%20==0):
    print("Epoch is: {0}, Loss is {1} and Accuracy is: {2}".format(epoch+1,loss.data,100*correct/len(train_loader.dataset)))

print("Finished Training")

# Checking the performance on test set

iteration_list = []
accuracy_list = []
prediction_list = []
label_list = []

with torch.no_grad():
  total = 0
  correct = 0
  for images, labels in test_loader:
    label_list.append(labels.numpy()[0])
    model.eval()

    test = images.view(-1, num_of_pixels)
    outputs = model(test)
    predictions = torch.max(outputs, 1)[1]
    prediction_list.append(predictions.numpy()[0])
    correct += (predictions == labels).sum().numpy()
    total += len(labels)

  iteration_list.append(count)
  print(f'Iteration: {count:5d}, Loss:{loss.data:.4f}, Final Accuracy:{correct*100/total:.3f}%')

# Generating the confusion matrix

confusion_matrix=metrics.confusion_matrix(label_list,prediction_list)
plt.figure(figsize=(12,10))
sns.heatmap(confusion_matrix,annot=True)

# ##### For class 0, it is most likely to be confused with class 8
# ##### For class 1, it is most likely to be confused with class 9
# ##### For class 2, it is most likely to be confused with class 4
# ##### For class 3, it is most likely to be confused with class 5
# ##### For class 4, it is most likely to be confused with class 2
# ##### For class 5, it is most likely to be confused with class 3
# ##### For class 6, it is most likely to be confused with class 4
# #####For class 7, it is most likely to be confused with class 3
# ##### For class 8, it is most likely to be confused with class 0
# ##### For class 9, it is most likely to be confused with class 1

# Class 0 and 8 are most likely to be confused overall


