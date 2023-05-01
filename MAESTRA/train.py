from __future__ import unicode_literals
import dataset
import feature
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import torch.optim as optim
from torchsummary import summary
from time import time
import pandas as pd
import pickle
from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display


def t_train(instrument, model, dataloader_dict, criterion, optimizer, num_epoch, device):
    
    train_loss_list = []
    val_loss_list = []
    accuracy_list = []
    start = time()
    inst_dic = {'cello': 0, 'clarinet': 1, 'drum': 2, 'flute': 3,
                'piano': 4, 'viola': 5, 'violin': 6}
    target_lbl = inst_dic[instrument]

    print("Begin training...")
    # 훈련 루프
    for epoch in range(num_epoch):
        accuracy = 0.0
        best_acc = 0.0
        n = 0
        running_acc = 0.0
        epoch_loss = 0.0
        val_epoch_loss = 0.0
        total = 0

        print('Epoch {}/{}'.format(epoch + 1, num_epoch))
        print('-' * 20)

        for inputs, targets in dataloader_dict['train']:
            inputs = inputs.to(device).float()
            targets = targets[:,target_lbl].unsqueeze(1).float().to(device)
        
            optimizer.zero_grad()
            output = model(inputs)

            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            n += len(output)
            if n % 1000 == 0:
                print(f"Try : {n:4d}, Loss : {epoch_loss/n:.6f}")
                now = time() - start
                print('훈련 진행 중.. {:.0f}m {:.0f}s'.format(now // 60, now % 60))

        epoch_loss = epoch_loss / len(dataloader_dict['train'].dataset)
        print('Train Loss: {:.6f}'.format(epoch_loss))
        
        # 검증 루프
        with torch.no_grad():
            model.eval()

            for inputs, targets in dataloader_dict['val']:
                inputs = inputs.to(device).float()
                targets = targets[:,target_lbl].unsqueeze(1).float().to(device)

                output = model(inputs)
                val_loss = criterion(output, targets)
                
                predicted = nn.Sigmoid()(output) >= torch.FloatTensor([0.5]).to(device)
                val_epoch_loss += val_loss.item()
                total += targets.size(0)
                running_acc += (predicted == targets).sum().item()

        val_epoch_loss = val_epoch_loss / len(dataloader_dict['val'].dataset)
        print('Val Loss: {:.6f}'.format(val_epoch_loss))

        accuracy = (100 * running_acc / total)

        if accuracy > best_acc:
            best_acc = accuracy
            #torch.save(model.state_dict(), './model/3/MAESTRA_' + instrument + '.pt')

        print('Epoch: ', epoch + 1, ', Training Loss is: %.5f' % epoch_loss,
              ', Validation Loss is: %.5f' % val_epoch_loss, ', Accuracy is %.2f %%' % best_acc)
        train_loss_list.append(round(epoch_loss,5))
        val_loss_list.append(round(val_epoch_loss,5))
        accuracy_list.append(round(accuracy,5))

        torch.save(model.state_dict(), './model/4/MAESTRA_' + instrument + '_' + str(epoch+1) + '.pt')
        print()

    time_elapsed = time() - start
    print('Training completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Train_loss_list')
    print(train_loss_list)
    print('val_loss_list')
    print(val_loss_list)
    print('accuracy_list')
    print(accuracy_list)
    
   # torch.save(model.state_dict(), './model/2_batch64/MAESTRA_' + instrument + '.pth')

def t_test(instrument, model, dataloader_dict, criterion, device):
    accuracy = 0.0
    running_acc = 0.0
    val_epoch_loss = 0.0
    total = 0

    inst_dic = {'cello': 0, 'clarinet': 1, 'drum': 2, 'flute': 3,
                'piano': 4, 'viola': 5, 'violin': 6}
    target_lbl = inst_dic[instrument]
    
    with torch.no_grad():
        model.eval()

        for inputs, targets in dataloader_dict['test']:
            inputs = inputs.to(device).float()
            targets = targets[:,target_lbl].unsqueeze(1).float().to(device)

            output = model(inputs)
            val_loss = criterion(output, targets)

            predicted = output >= torch.FloatTensor([0.5]).to(device)
            val_epoch_loss += val_loss.item()
            total += targets.size(0)
            running_acc += (predicted == targets).sum().item()

        val_epoch_loss = val_epoch_loss / len(dataloader_dict['test'].dataset)
        #print('Test Loss: {:.6f}'.format(val_epoch_loss))

        accuracy = (100 * running_acc / total)

        print('Test Loss is: %.5f' % val_epoch_loss, ', Test Accuracy is %.2f %%' % accuracy)


