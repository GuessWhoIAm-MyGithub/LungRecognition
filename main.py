import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms,models,utils
from torchsummary import summary
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt
import os
import seaborn as sns
import pandas as pd
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from PIL import Image

image_transform = {
    'train':transforms.Compose([
        transforms.RandomResizedCrop(size=300,scale=(0.8,1.1)),#随机长宽比剪裁原始图片，参数表示随机crop得图片会在0.8到1.1倍之间
        transforms.RandomRotation(degrees=10),#旋转一定角度，-10到10度
        transforms.ColorJitter(0.4,0.4,0.4),#修改亮度，对比度，饱和度
        transforms.RandomHorizontalFlip(),#水平翻转
        transforms.CenterCrop(size=256),#根据给定得size从中心裁剪
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225])#标准化，三个通道，每个通道对应标准化

    ]),
    'val': transforms.Compose([
        transforms.Resize(300),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(size=300),
        transforms.CenterCrop(size=256),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225])
    ])
}

data_dir='./chest_xray/'
train_dir=data_dir+'train/'
val_dir = data_dir+'val/'
test_dir = data_dir+'test/'

#读取数据
datasets={
    'train': datasets.ImageFolder(train_dir,transform=image_transform['train']),
    'val':datasets.ImageFolder(val_dir, transform=image_transform['val']),
    'test':datasets.ImageFolder(test_dir, transform=image_transform['test'])
}


BATCH_SIZE=128
#创建iterator，分批读取数据
dataloaders={
    'train':DataLoader(datasets['train'],batch_size=BATCH_SIZE,shuffle=True),
    'val':DataLoader(datasets["val"], batch_size=BATCH_SIZE, shuffle=True),
    'test':DataLoader(datasets["test"], batch_size=BATCH_SIZE, shuffle=True)
}



#创建label的键值对
LABEL=dict((v,k) for k,v in datasets['train'].class_to_idx.items())#class_to_idx获取名称并创建对应索引,item获取dict中的的元素
# print(LABEL)

# print(dataloader['train'].dataset.classes)#类别
# print(dataloader['train'].dataset.root)#路径

files_normal=os.listdir(os.path.join(str(dataloaders['train'].dataset.root),'NORMAL'))#获得所有图片的名称
#print(files_normal)

file_infected=os.listdir(os.path.join(str(dataloaders["train"].dataset.root),'PNEUMONIA'))
#print('----------------\n',file_infected)
# print(dataloaders['train'].dataset.root)
# print(dataloaders['val'].dataset)
# print(dataloaders['test'].dataset)

from torch.utils.tensorboard import SummaryWriter
#向事件文件写入事件和摘要

log_dir='logdir/'

#获取tensorboard writer
def tb_writer():
    timestr=time.strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(log_dir+timestr)#写入日志
    return writer


writer=tb_writer()

images,labels = next(iter(dataloaders['train']))
#获取数据，转换为可迭代数据

#显示部分图片
'''
def imshow(img):
    img=img / 2+0.5#逆正则化
    np_img=img.numpy()
    plt.imshow(np.transpose(np_img, (1,2,0)))#改变通道顺序
    plt.show()

grid=utils.make_grid(images)#把图片若干拼成一幅图像在网格中展示
imshow(grid)

writer.add_image('Xray grid ',grid,0)

writer.flush()

def showImage(img):
    plt.figure(figsize=(16,16))
    plt.imshow(img)
    plt.axis("off")
    plt.show()
oneimg=Image.open(dataloaders['train'].dataset.root+'Normal/IM-0117-0001.jpeg')

showImage(oneimg)

'''

def mistakeimgs(pred,writer,target,images,output,epoch,count=10):
    mistake=(pred!=target.data)
    for index,img_tensor in enumerate(images[mistake][:count]):
        img_name='Epoch{}-->Pred-->{}-->Actual-->{}'.format(epoch,LABEL[pred[mistake].tolist()[index]],LABEL[target.data[mistake].tolist()[index]])

        writer.add_image(img_name,img_tensor,epoch)#写入日志


class AdaptiveConcat2d(nn.Module):
    def __init__(self,size=None):
        super(AdaptiveConcat2d, self).__init__()
        size = size or (1,1)#kernel大小
        self.avgpooling=nn.AdaptiveAvgPool2d(size)#平均池化
        self.maxpooling = nn.AdaptiveMaxPool2d(size)#最大池化

    def forward(self, x):
        return torch.cat([self.avgpooling(x), self.maxpooling(x)],dim=1)


#迁移学习
def getmodel():
    model=models.resnet50(pretrained=True)
    #冻结模型参数
    for para in model.parameters():
        para.requires_grad = False
    model.avgpool = AdaptiveConcat2d()

    #修改fc层
    model.fc=nn.Sequential(
        nn.Flatten(),
        nn.BatchNorm1d(4096),#加速收敛速度
        nn.Dropout2d(0.5),
        nn.Linear(4096, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(0.5),
        nn.Linear(512,2),
        nn.LogSoftmax(dim=1)  #损失函数，把input转换为概率分布的形式
    )

    return model  #返回我们修改过的model



def trainAndVal(model,device,trainLoader,valLoader,optimizer,criterion,epoch,writer):
    model.train()
    total_loss=0.0
    val_loss=0.0
    acc=0
    for batch_idx,(images,labels) in enumerate(trainLoader):
        images,labels=images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()*images.size(0)
    trainloss=total_loss / len(trainLoader.dataset)
    writer.add_scalar('Training LOSS',trainloss, epoch)
    writer.flush()

    model.eval()
    with torch.no_grad():
        for images, labels in valLoader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()*images.size(0)
            _,pred=torch.max(outputs, dim=1)

            correct = pred.eq(labels.view_as(pred))
            accuracy=torch.mean(correct.type(torch.FloatTensor))
            acc+=accuracy.item()*images.size(0)
        val_loss /= len(valLoader.dataset)
        acc /= len(valLoader.dataset)
    return trainloss,val_loss,acc

def test(model,device,testLoader,criterion, epoch,writer):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (images, labels)in enumerate(testLoader):
            images,labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _,pred=torch.max(outputs, dim=1)

            correct += pred.eq(labels.view_as(pred)).sum().item()

            mistakeimgs(pred,writer,labels,images,outputs,epoch)
        avgloss=total_loss / len(testLoader.dataset)
        acc=100*correct / len(testLoader.dataset)

        writer.add_scalar('Test Loss',total_loss, epoch)
        writer.add_scalar('Acc.:',acc,epoch)
        writer.flush()
        return total_loss, acc

device=torch.device('cuda' if torch.cuda.is_available() else "cpu")

model=getmodel().to(device)
criterion = nn.NLLLoss()
optimizer=optim.SGD(model.parameters(), lr=1e-3)

def trainEpoch(model,device,dataloaders,criterion, optimizer, epochs,writer):
    print('{0:>15} | {1:>15} | {2:>15} | {3:>15} | {4:>15} | {5:>15}'.format('Epochs','TrainLoss','val_loss','val_acc','testloss','testacc'))
    bestloss=np.inf
    for epoch in range(0,epochs+ 1):
        train_loss,val_loss,val_acc=trainAndVal(model, device,dataloaders['train'], dataloaders['val'],optimizer, criterion, epoch, writer)

        testloss,testacc=test(model, device,dataloaders["test"],criterion, epoch, writer)

        if testloss<bestloss:
            bestloss = testloss
            torch.save(model.state_dict(),'Model.pth')
        print('{0:>15} | {1:>15} | {2:>15} | {3:>15} | {4:>15} | {5:>15}'.format(epoch,train_loss, val_loss, val_acc,testloss,testacc))
        writer.flush()

epochs=10
model.cuda()
trainEpoch(model, device, dataloaders,criterion,optimizer, epochs, writer)
writer.close()


def plot_confusion(cm):
    plt.figure()
    plot_confusion_matrix(cm,figsize=(24,16),cmap=plt.cm.Blues)#绘制混淆矩阵
    plt.xticks(range(2),['Normal','Pneumonia'],fontsize=14)
    plt.yticks(range(2), ['Normal', 'Pneumonia'], fontsize=14)
    plt.xlabel('pred label',fontsize=14)
    plt.xlabel('pred label',fontdict=14)
    plt.show()


def accuracy(output,labels):
    _,pred=torch.max(output, dim=1)
    correct = torch.tensor(torch.sum((pred == labels).item())/len(pred))
    return correct

def metrics(output, labels):
    _,pred = torch.max(output, dim=1)

    cm=confusion_matrix(labels.cpu().numpy(), pred.cpu().numpy())

    plot_confusion(cm)
    tn,fp,fn,tp=cm.ravel()

    precision=tp/(tp+fp)

    recall=tp/(tp+fn)

    f1=2*((precision*recall)/(precision + recall))

    return precision, recall, f1

precisions=[]
recall=[]
f1s=[]
acc=[]

with torch.no_grad():
    model.eval()
    for datas,labels in dataloaders['test']:
        datas,labels=datas.to(device),labels.to(device)

        outputs=model(datas)
        precisions,recall, f1=metrics(outputs, labels)
        acc=accuracy(outputs, labels)
        precisions.append(precisions)
        recall.append(recall)
        f1s.append(f1)
        acc.append(acc.item())
