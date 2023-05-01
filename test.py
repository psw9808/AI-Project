from __future__ import unicode_literals
import dataset
import feature
import model
import train
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import torch.optim as optim
from torchsummary import summary
import time
import pandas as pd
import pickle
from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt
import torch, gc
import sound
import youtube_dl
import ffmpeg

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('학습을 진행하는 기기:', device)
print('cuda index:', torch.cuda.current_device())
print('gpu 개수:', torch.cuda.device_count())
print("graphic name:", torch.cuda.get_device_name())

# GPU 메모리 캐시 삭제
# gc.collect()
# torch.cuda.empty_cache()

# 데이터셋 추출
# dataset.d_extract(re_extract=False, audio_len=5)

# 데이터프레임 생성
# train_df = dataset.d_create_df(audio_len=5)

# 데이터셋 생성
# dataset.d_create_npy('./MAESTRA_dataset/','test', False)

# 특성추출 테스트
# 10초 가로 크기 : 431, 5초 가로 크기 : 216
# 주파수 해상도 4배 세로 크기 : 336, 주파수 해상도 2배 세로 크기 : 168
# 84x431 -> hop_length 512->1024 ->> 168x216
# feature.compare_features('./MAESTRA_dataset/1020-02.wav')
# feature.show_cqt('./MAESTRA_dataset/1468-005.wav')

# with open('MAESTRA_data.pkl', "rb") as file:
#     data = pickle.load(file)
# split_idx = data.index[data['audio_file'] == '1028-00.wav']
# print(split_idx)
# labels = data.loc[:split_idx[0] - 1, ['audio_file']].values.tolist()
# print(labels)
# print(len(labels))
# print(data)

# display(data.iloc[0])
# display(data.loc[:, 'cello'])

# label_lst = ['cello', 'piano']
# labels = data[label_lst].values.tolist()
# print(labels[0])


# feature.show_cqt('./MAESTRA_dataset/1031-002.wav')
"""
data = np.array([])
label_lst = ['cello', 'clarinet', 'drum', 'flute', 'piano', 'viola', 'violin']
with open('./MAESTRA_data.pkl', "rb") as f:
    pkl_data = pickle.load(f)
split_idx = pkl_data.index[pkl_data['audio_file'] == '1130-000.wav']
labels = np.array(pkl_data.loc[:split_idx[0] - 1, label_lst].values.tolist())
audio_path = pkl_data.loc[:split_idx[0] - 1, 'audio_file'].values.tolist()

data = np.append(data, feature.f_cqt('./MAESTRA_dataset/' + audio_path[0]))
print(split_idx)
print(audio_path)
print(audio_path[0])
print(data)
print(labels[0])
data = data.reshape((336, 431, 1))
print(data.shape)
"""


batch_size = 64

test_dataset = dataset.maestraDataset(root='./MAESTRA_dataset/', purpose='test', transform=None)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                          shuffle=False, num_workers=0)

# dataiter = iter(train_loader)
# images, labels = dataiter.next()
# print(images.size())
# print(labels)
inst = 'clarinet'
epoch = 10
modelpath = './model/4/MAESTRA_' + inst 
n_model = model.MAESTRA2(1,1,True)

#n_model.load_state_dict(torch.load(modelpath))
#n_model = n_model.to(device)

# n_model = model.ResNet(base_dim=32).to(device)
#summary(n_model, input_size=(1, 168, 216))

dataloader_dict = {'test': test_loader}
# criterion = nn.BCEWithLogitsLoss()
# criterion = nn.MultiLabelSoftMarginLoss()
criterion = nn.BCEWithLogitsLoss().to(device)

for i in range (epoch):
    modelpath_n = modelpath + '_' + str(i+1) + '.pt'
    print(modelpath_n)
    n_model.load_state_dict(torch.load(modelpath_n))
    n_model = n_model.to(device)
    t_model = train.t_test(inst, n_model, dataloader_dict, criterion, device)

"""
torch.save(model.state_dict(), "./MAESTRAmodel.pt")

correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images = images.to(device).float()
        labels = labels.type(torch.LongTensor)
        labels = labels.to(device)
        outputs = model(images)
        print(outputs)
        print(labels.data)
"""

"""
training_epochs = 72
num_batch_size = 128

learning_rate = 0.001
opt = keras.optimizers.Adam(learning_rate=learning_rate)

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

history = model.fit(x_train, y_train, batch_size=num_batch_size, epochs=training_epochs)

def plot_history(history):
    key_value = list(set([i.split("val_")[-1] for i in list(history.history.keys())]))
    plt.figure(figsize=(12, 4))
    for idx, key in enumerate(key_value):
        plt.subplot(1, len(key_value), idx + 1)
        vis(history, key)
    plt.tight_layout()
    plt.show()


plot_history(history)
"""
