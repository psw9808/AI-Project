import pandas as pd
import os
import feature as ft
import moviepy.editor as mp
from time import time


# This file is to handle audio files (s:sound)
def s_split(input_path, output_path, audio_id, length=60, start_t=0, end_t=None):
    """
    :param input_path: 작업할 오디오 데이터 경로
    :param output_path: 저장할 오디오 데이터 경로
    :param audio_id: 저장할 오디오 파일 인덱스
    :param length: 분할할 데이터셋 하나당 길이
    :param start_t: 오디오 분할을 시작할 시각
    :param end_t: 오디오 분할이 끝나는 시각
    """
    audio_clip = mp.AudioFileClip(input_path)
    # end_t를 입력 받지 않거나 전체 오디오 길이보다 긴 경우, 전체 오디오 길이까지 분할
    if end_t is None or end_t > audio_clip.duration:
        end_t = audio_clip.duration
    dur = start_t
    num = 0
    # length 초씩 오디오를 분할하고 저장
    while dur + length <= end_t:
        clip = audio_clip.subclip(dur, dur + length)
        clip.write_audiofile(output_path + '\\' + audio_id + '-' + str(num).rjust(3, '0') + '.wav')
        dur += length
        num += 1
    # 남은 끝부분 데이터까지 사용
    # clip = audio_clip.subclip(dur, int(end_t) - 1)
    # clip.write_audiofile(output_path + '\\' + audio_id + '-' + str(num).rjust(2, '0') + '.wav')