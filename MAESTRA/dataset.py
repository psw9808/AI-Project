from __future__ import unicode_literals
#import youtube_dl
#import ffmpeg
#import sound
import pandas as pd
import os
from sklearn.preprocessing import MultiLabelBinarizer
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
#import feature


# 데이터셋을 만들기 위한 파일 (d:dataset)
class maestraDataset(torch.utils.data.Dataset):
    def __init__(self, root, purpose='train', transform=None):
        """
        :param root: train/val/test wav파일이 있는 폴더 경로 (ex:./MAESTRA_dataset/)
        :param purpose: train->훈련용 데이터셋, val->검증용 데이터셋, test->테스트 데이터셋
        :param transform: Data augmentation 커스텀
        """
        self.root = root
        self.purpose = purpose
        self.transform = transform
        self.file = './MAESTRA_' + self.purpose + '.pkl'
        self.data = np.array([])

        # 라벨 리스트 정의
        label_lst = ['cello', 'clarinet', 'drum', 'flute', 'piano', 'viola', 'violin']

        # 데이터 프레임 불러오기
        with open(self.file, "rb") as f:
            pkl_data = pickle.load(f)

        # train, val, test 여부에 따라 파일명(ex:1000-000.wav), 라벨 목록(ex:[0,1,0,0])을 순서대로 리스트에 할당
        self.labels = np.array(pkl_data.loc[:, label_lst].values.tolist())
        self.audio_path = pkl_data.loc[:, 'audio_file'].values.tolist()

        self.length = len(pkl_data)
        print(purpose + ' 데이터 수: ' + str(self.length))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # 변환된 데이터를 불러오기
        self.data = np.load('./' + self.purpose + '/' + self.audio_path[idx][0:8] + '.npy')
        data = self.data.reshape((1, 168, 216))

        if self.transform is not None:
            data = self.transform(data)

        label = self.labels[idx]

        return data, label


def d_extract(re_extract=False, audio_len=10):
    """
    :param re_extract: 모든 데이터셋 초기화 후 재추출 여부
    :param audio_len: 데이터셋 하나 당 길이
    """
    if re_extract:
        # Dataset 보관 폴더 존재 여부 확인
        if os.path.exists('./MAESTRA_dataset'):
            # 폴더 내 모든 Dataset 파일 제거
            for file in os.scandir('./MAESTRA_dataset'):
                os.remove(file.path)
            print('Removed all dataset file')
        else:
            print('Cannot find dataset directory')
            return

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': 'output.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
        }],
    }

    metadata = pd.read_excel('./metadata.xlsx', sheet_name=0)
    print(metadata)

    last_url=''
    for index, row in metadata.iterrows():
        # 비어있는 url 셀이 있을 때 까지 읽어들이며 Dataset을 생성
        if str(row["url"]) == 'nan':
            break
        # Dataset 중복 생성 방지
        if os.path.isfile('./MAESTRA_dataset/' + str(row["id"]) + '-000.wav'):
            continue

        url, id, start, end = str(row["url"]), str(row["id"]), int(row["start"]), int(row["end"])
        # 데이터 추출
        if url == last_url:
            # 같은 영상에서 연속으로 추출을 한다면 불필요한 download를 하지 않고 기존 file을 사용
            # audio_len 초씩 나눈 Dataset 저장
            sound.s_split('./output.wav', './MAESTRA_dataset', id, audio_len, start, end)
        else:
            with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
                stream = ffmpeg.input('./output.m4a')
                stream = ffmpeg.output(stream, './output.wav')
            # audio_len 초씩 나눈 Dataset 저장
            sound.s_split('./output.wav', './MAESTRA_dataset', id, audio_len, start, end)
        last_url = url

    print('Dataset extraction complete')
    # ERROR: unable to download video data: HTTP Error 403: Forbidden 에러가 떴을 시
    # youtube-dl --rm-cache-dir 입력해서 해결


def d_create_df(audio_len=10):

    metadata = pd.read_excel('./metadata.xlsx', sheet_name=0)
    train_data_lst = []
    val_data_lst = []
    test_data_lst = []
    total_data_lst = []
    mlb = MultiLabelBinarizer()

    for index, row in metadata.iterrows():
        # 비어있는 url 셀이 있을 때 까지 읽어들이며 Data list를 생성
        if str(row["url"]) == 'nan':
            break

        ensemble, id, start, end, val_test = \
            str(row["ensemble"]), str(row["id"]), int(row["start"]), int(row["end"]), str(row["val/test"])

        n = int((end - start) / audio_len)

        # 각 데이터셋의 [file name, label]을 리스트에 저장
        labels = ensemble.split(' ')

        for num in range(0, n):
            audio_file = id + '-' + str(num).rjust(3, '0') + '.wav'

            # train/validation/test용 데이터리스트를 각각 저장
            if val_test == 'val':
                val_data_lst.append([audio_file, labels])
            elif val_test == 'test':
                test_data_lst.append([audio_file, labels])
            else:
                train_data_lst.append([audio_file, labels])
            # for statistical purpose
            total_data_lst.append([audio_file, labels])

    # file name과 labels를 열로 하는 데이터 프레임을 생성
    train_data_df = pd.DataFrame(train_data_lst, columns=['audio_file', 'labels'])
    val_data_df = pd.DataFrame(val_data_lst, columns=['audio_file', 'labels'])
    test_data_df = pd.DataFrame(test_data_lst, columns=['audio_file', 'labels'])
    total_data_df = pd.DataFrame(total_data_lst, columns=['audio_file', 'labels'])

    # 모든 label 성분을 자동으로 분류하고, 각 label을 열로 하는 새 데이터 프레임 생성
    labels = mlb.fit_transform(train_data_df['labels'].values)
    new_train_data_df = pd.DataFrame(columns=mlb.classes_, data=labels)
    new_train_data_df.insert(0, 'audio_file', train_data_df['audio_file'])

    labels = mlb.fit_transform(val_data_df['labels'].values)
    new_val_data_df = pd.DataFrame(columns=mlb.classes_, data=labels)
    new_val_data_df.insert(0, 'audio_file', val_data_df['audio_file'])

    labels = mlb.fit_transform(test_data_df['labels'].values)
    new_test_data_df = pd.DataFrame(columns=mlb.classes_, data=labels)
    new_test_data_df.insert(0, 'audio_file', test_data_df['audio_file'])

    labels = mlb.fit_transform(total_data_df['labels'].values)
    new_total_data_df = pd.DataFrame(columns=mlb.classes_, data=labels)
    new_total_data_df.insert(0, 'audio_file', total_data_df['audio_file'])

    # 새 데이터 프레임을 csv파일과 pickle파일 형태로 저장
    new_train_data_df.to_csv('./MAESTRA_train.csv')
    new_train_data_df.to_pickle('./MAESTRA_train.pkl')
    print(len(new_train_data_df), "개의 파일들의 Train 데이터프레임이 완성되었습니다.")

    new_val_data_df.to_csv('./MAESTRA_val.csv')
    new_val_data_df.to_pickle('./MAESTRA_val.pkl')
    print(len(new_val_data_df), "개의 파일들의 Validation 데이터프레임이 완성되었습니다.")

    new_test_data_df.to_csv('./MAESTRA_test.csv')
    new_test_data_df.to_pickle('./MAESTRA_test.pkl')
    print(len(new_test_data_df), "개의 파일들의 Test 데이터프레임이 완성되었습니다.")

    new_total_data_df.to_csv('./MAESTRA_total.csv')
    print(len(new_total_data_df), "개의 파일들의 통계용 Total 데이터프레임이 완성되었습니다.")

    return new_train_data_df, new_val_data_df, new_test_data_df, new_total_data_df


def d_create_npy(root, purpose, re_create=False):
    data = np.array([])
    file = './MAESTRA_' + purpose + '.pkl'

    if re_create:
        # Dataset 보관 폴더 존재 여부 확인
        if os.path.exists('./data/cqt_' + purpose):
            # 폴더 내 모든 Dataset 파일 제거
            for dsfile in os.scandir('./data/cqt_' + purpose):
                os.remove(dsfile.path)
            print('Removed all dataset file')
        else:
            print('Cannot find dataset directory')
            return

    with open(file, "rb") as f:
        pkl_data = pickle.load(f)

    audio_path = pkl_data.loc[:, 'audio_file'].values.tolist()

    for idx in audio_path:
        # Dataset 중복 생성 방지
        if os.path.isfile('./data/cqt_' + purpose + '/' + idx + '.npy'):
            continue

        data = np.array(feature.f_cqt(root + idx))
        data = data.reshape((1, 168, 216))
        np.save('./data/cqt_' + purpose + '/' + idx, data)
        print(idx + ' completed')
