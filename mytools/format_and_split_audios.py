#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Wang Zifan

"""切分和识别文件夹下的音频"""
import argparse
import os
import queue
import time
import wave
import threading

import numpy as np
import webrtcvad
from scipy.io import wavfile
from ffmpy import FFmpeg
import shutil


class FormatSingleAudioThread(threading.Thread):
    """ 格式化音频，只支持采样大小为2的音频

        Author: Wang Zifan
        Date: 2022/12/07

        Attributes:
            audio_path (str): 音频路径
            output_path(str):输出路径
            num_channels (int): 转换后的通道数
            resample_rate (int): 转换后的采样率
            ffmpeg_executable_path (str): ffmpeg可执行文件的路径
    """

    def __init__(self, thread_id: str, audio_path: str, output_path: str, num_channels: int = 1,
                 resample_rate: int = 16000,
                 ffmpeg_executable_path: str = '/usr/bin/ffmpeg'):
        super(FormatSingleAudioThread, self).__init__()
        self.thread_id = thread_id
        self.audio_path = audio_path
        self.output_path = output_path
        self.num_channels = num_channels
        self.resample_rate = resample_rate
        self.ffmpeg_executable_path = ffmpeg_executable_path

    def run(self):
        audio_type = self.audio_path.split('.')[-1]
        output_dir = os.path.dirname(self.output_path)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        if audio_type != 'wav':
            mpy_obj = FFmpeg(
                executable=self.ffmpeg_executable_path,
                inputs={
                    self.audio_path: None
                },
                outputs={
                    self.output_path: '-y -f wav -ac ' + str(self.num_channels) + ' -ar ' + str(
                        self.resample_rate) + ' -acodec ' + 'pcm_s16le'
                }
            )
            mpy_obj.run()
        else:
            # 读取更改后的音频文件
            rf = wave.open(self.audio_path, "rb")
            # 获取音频参数
            channels = rf.getnchannels()
            rate = rf.getframerate()
            samp_width = rf.getsampwidth()
            if channels == self.num_channels and rate == self.resample_rate and samp_width == 2:
                if os.path.isfile(self.output_path):
                    os.remove(self.output_path)
                shutil.copy(self.audio_path, self.output_path)
            else:
                mpy_obj = FFmpeg(
                    executable=self.ffmpeg_executable_path,
                    inputs={
                        self.audio_path: None
                    },
                    outputs={
                        self.output_path: '-y -f wav -ac ' + str(self.num_channels) + ' -ar ' + str(
                            self.resample_rate) + ' -acodec ' + 'pcm_s16le'
                    }
                )
                mpy_obj.run()


class SplitAndSave1ChannelWavThread(threading.Thread):
    """ 切分并保存单通道音频

        Author: Wang Zifan
        Date: 2022/09/22

        Attributes:
            thread_id (str): 线程id
            wav_path (str): 单通道音频路径
            splited_dir (str): 切分结果音频存放目录
            role (str): 角色  左通道/右通道,客户端/服务端 等
            level (tuple): 切分严格程度等级划分,如(2.4, 5, 10, 16, 20, 30)表示0-2.4秒不切分,
                           2.4-5秒15帧里有少于1帧是人声才切分
                           5-10秒15帧里有少于2帧是人声则切分
                           10-16秒10帧里有少于2帧是人声就切分
                           16-20秒5帧里有少于2帧是人声就切分
                           20-30秒3帧里有少于2帧是人声就切分
                           30秒强制切分
            remove_origin_wav (bool): 是否删除原始音频
    """

    def __init__(self, thread_id: str, wav_path: str, splited_dir: str, role: str = '',
                 level: tuple = (2.4, 5, 10, 16, 20, 30), remove_origin_wav: bool = False):
        super(SplitAndSave1ChannelWavThread, self).__init__()
        self.thread_id = thread_id
        self.wav_path = wav_path
        self.splited_dir = splited_dir
        self.role = role
        self.level = level
        self.remove_origin_wav = remove_origin_wav
        self.wav_path_dir = os.path.dirname(wav_path)
        self.wav_path_basename = os.path.basename(wav_path)

    def run(self):
        self.split_single_1channel_wav(self.wav_path, self.splited_dir, self.role, self.level)
        if self.remove_origin_wav:
            os.remove(self.wav_path)

    def split_single_1channel_wav(self, wav_path: str, splited_dir: str, role: str = '',
                                  level: tuple = (2.4, 5, 10, 16, 20, 30)):
        """ 切分并保存单个单通道音频

            Args:
                wav_path (str): 单通道音频路径
                splited_dir (str): 切分结果音频存放目录
                role (str): 角色(example: left, right, server, client)
                level (tuple): 切分严格程度等级划分,如(2.4, 5, 10, 16, 20, 30)表示0-2.4秒不切分,
                               2.4-5秒15帧里有少于1帧是人声才切分
                               5-10秒15帧里有少于2帧是人声则切分
                               10-16秒10帧里有少于2帧是人声就切分
                               16-20秒5帧里有少于2帧是人声就切分
                               20-30秒3帧里有少于2帧是人声就切分
                               30秒强制切分

            Returns:
                None
        """
        if not os.path.isdir(splited_dir):
            os.makedirs(splited_dir)
        if role == '':
            role_str = ''
        else:
            role_str = '_' + role
        # 读取更改后的音频文件
        rf = wave.open(wav_path, "rb")
        # 获取音频参数
        # 通道数，vad只支持单通道
        channels = rf.getnchannels()
        # 每个通道的采样率，vad只支持8k、16k、32k和48k
        rate = rf.getframerate()
        # 采样大小（每次采样数据的字节数），vad只支持2
        samp_width = rf.getsampwidth()
        # print("channels:", channels)
        # print("rate:", rate)
        # print("samp_width:", samp_width)
        # 单通道切分
        assert channels == 1
        # vad采样大小只支持2
        assert samp_width == 2

        # 定义帧长，vad只支持10、20和30ms
        frame_duration_ms = 20
        # 读取数据块大小（每一帧每个通道上的采样次数）（除以通道数和乘以采样大小分之2来进行勉强的纠正）
        chunk_size = int(rate * frame_duration_ms / 1000 / channels * 2 / samp_width)
        # 判断人声的敏感程度（一般为1~3,3为最敏感）
        vad = webrtcvad.Vad(3)

        buffer_data = b''
        temp = []
        file_num = 1

        # 上一段是否为人声
        pre_is_speech = False
        # 当前已读的frames数
        num_frames_readed = 0
        while True:
            # 读取一帧的数据
            data = rf.readframes(chunk_size)
            num_frames_readed += chunk_size
            len_data = len(data)
            # 如果这一帧是最后一帧
            if len_data < chunk_size * channels * samp_width:
                buffer_data += data
                # 如果上一段有人声，则将数据写入文件
                if pre_is_speech:
                    temp_start_time = (num_frames_readed - len(buffer_data) / channels / samp_width) / rate
                    temp_end_time = num_frames_readed / rate
                    temp_wav_path = os.path.join(splited_dir,
                                                 str('{:.2f}'.format(temp_start_time)).zfill(8) + '_' + str(
                                                     '{:.2f}'.format(temp_end_time)).zfill(8) + role_str + '.wav')
                    # print(os.path.basename(temp_wav_path))
                    wf = wave.open(temp_wav_path, "wb")
                    wf.setnchannels(channels)
                    wf.setsampwidth(samp_width)
                    wf.setframerate(rate)
                    wf.writeframes(buffer_data)
                    wf.close()
                buffer_data = b''
                temp = []
                # 结束循环
                break
            # 否则
            else:
                is_speech = vad.is_speech(data, rate)
            # 将这一帧的数据和判断结果放入缓存
            buffer_data += data
            temp += [is_speech]

            # buffer_data里面包含数据的帧数
            numframes_buffer_data = int(len(buffer_data) / chunk_size / 2)
            # cache_size帧里面有大于等于threshold帧是人声，就认为这一段(cache_size帧)是人声
            # 根据buffer_data里面包含数据的帧数动态调整切分窗口
            # 小于3秒，什么都不做。如果连续1秒没人声，则删除这1秒
            if numframes_buffer_data <= int(level[0] * 1000 / frame_duration_ms):
                cache_size = 0
                threshold = 0
                # 每30帧判断一次，判断当前buffer_data里面的数据是否为人声
                if numframes_buffer_data % 30 == 0:
                    # 不是人声，则清空缓存
                    if sum(temp) < numframes_buffer_data / 15:
                        pre_is_speech = False
                        buffer_data = b''
                        temp = []
                    # 是人声。本阶段最后一次判断时，清空temp,以便后续判断
                    else:
                        pre_is_speech = True
                        if numframes_buffer_data == int(level[0] * 1000 / frame_duration_ms):
                            temp = []
            # 正常切分
            elif numframes_buffer_data < int(level[1] * 1000 / frame_duration_ms):
                cache_size = 15
                threshold = 1
            # 按较小的窗口切分
            elif numframes_buffer_data < int(level[2] * 1000 / frame_duration_ms):
                cache_size = 15
                threshold = 2
            # 按更小的窗口切分
            elif numframes_buffer_data < int(level[3] * 1000 / frame_duration_ms):
                cache_size = 10
                threshold = 2
            # 按更小的窗口切分
            elif numframes_buffer_data < int(level[4] * 1000 / frame_duration_ms):
                cache_size = 5
                threshold = 2
            # 按更小的窗口切分
            elif numframes_buffer_data < int(level[5] * 1000 / frame_duration_ms):
                cache_size = 3
                threshold = 2
            # 达到最大长度，强制切分，将buffer_data里面的数据写入文件并语音识别，然后清空缓存，回归初始状态
            else:
                cache_size = 0
                threshold = 0
                pre_is_speech = False
                temp_start_time = (num_frames_readed - len(buffer_data) / channels / samp_width) / rate
                temp_end_time = num_frames_readed / rate
                temp_wav_path = os.path.join(splited_dir,
                                             str('{:.2f}'.format(temp_start_time)).zfill(8) + '_' + str(
                                                 '{:.2f}'.format(temp_end_time)).zfill(8) + role_str + '.wav')
                # print(os.path.basename(temp_wav_path))
                wf = wave.open(temp_wav_path, "wb")
                file_num += 1
                wf.setnchannels(channels)
                wf.setsampwidth(samp_width)
                wf.setframerate(rate)
                wf.writeframes(buffer_data)
                wf.close()
                buffer_data = b''
                temp = []

            # 如果缓存满了
            if len(temp) >= cache_size > 0:
                num_is_speech = sum(temp)
                # cache_size帧里面有大于等于threshold帧是人声，就认为这一段(cache_size帧)是人声
                if num_is_speech >= threshold:
                    # 如果是人声，清空判断结果缓存
                    pre_is_speech = True
                    temp = []
                # 如果这一段不是人声且上一段是人声，就将缓存中的所有数据写入文件并清空缓存
                elif pre_is_speech == True:
                    pre_is_speech = False
                    temp_start_time = (num_frames_readed - len(buffer_data) / channels / samp_width) / rate
                    temp_end_time = num_frames_readed / rate
                    temp_wav_path = os.path.join(splited_dir,
                                                 str('{:.2f}'.format(temp_start_time)).zfill(8) + '_' + str(
                                                     '{:.2f}'.format(temp_end_time)).zfill(8) + role_str + '.wav')
                    # print(os.path.basename(temp_wav_path))
                    wf = wave.open(temp_wav_path, "wb")
                    file_num += 1
                    wf.setnchannels(channels)
                    wf.setsampwidth(samp_width)
                    wf.setframerate(rate)
                    wf.writeframes(buffer_data)
                    wf.close()
                    buffer_data = b''
                    temp = []
                # 如果这一段不是人声且上一段也不是人声
                else:
                    # 如果后(threshold-1)帧为人声，则认为这一段是人声
                    if 0 < num_is_speech == sum(temp[cache_size - threshold + 1:cache_size]):
                        pre_is_speech = True
                        temp = []
                    # 否则清空缓存
                    else:
                        buffer_data = b''
                        temp = []
        rf.close()


class SplitAndSave2ChannelWavThread(threading.Thread):
    """ 切分并保存单通道音频

        Author: Wang Zifan
        Date: 2022/09/22

        Attributes:
            thread_id (str): 线程id
            wav_path (str): 单通道音频路径
            splited_dir (str): 切分结果音频存放目录
            role_name_left (str): 左通道角色名称
            role_name_right (str): 右通道角色名称
            level (tuple): 切分严格程度等级划分,如(2.4, 5, 10, 16, 20, 30)表示0-2.4秒不切分,
                           2.4-5秒15帧里有少于1帧是人声才切分
                           5-10秒15帧里有少于2帧是人声则切分
                           10-16秒10帧里有少于2帧是人声就切分
                           16-20秒5帧里有少于2帧是人声就切分
                           20-30秒3帧里有少于2帧是人声就切分
                           30秒强制切分
            remove_origin_wav (bool): 是否删除原始音频
    """

    def __init__(self, thread_id: str, wav_path: str, splited_dir: str,
                 role_name_left: str = '', role_name_right: str = '',
                 level: tuple = (2.4, 5, 10, 16, 20, 30), remove_origin_wav: bool = False):
        super(SplitAndSave2ChannelWavThread, self).__init__()
        self.thread_id = thread_id
        self.wav_path = wav_path
        self.splited_dir = splited_dir
        self.wav_path_dir = os.path.dirname(wav_path)
        self.wav_path_basename = os.path.basename(wav_path)
        self.left_wav_path = os.path.join(self.wav_path_dir,
                                          ''.join(self.wav_path_basename.split('.')[0:-1]) + '_left.wav')
        self.right_wav_path = os.path.join(self.wav_path_dir,
                                           ''.join(self.wav_path_basename.split('.')[0:-1]) + '_right.wav')
        self.role_name_left = role_name_left
        self.role_name_right = role_name_right
        self.level = level
        self.remove_origin_wav = remove_origin_wav

    def run(self):
        self.split_left_right(self.wav_path, self.left_wav_path, self.right_wav_path)
        if not os.path.isdir(self.splited_dir):
            os.makedirs(self.splited_dir)
        left_split_thread = SplitAndSave1ChannelWavThread(self.thread_id + 'left', self.left_wav_path,
                                                          self.splited_dir, self.role_name_left, level=self.level)
        right_split_thread = SplitAndSave1ChannelWavThread(self.thread_id + 'right', self.right_wav_path,
                                                           self.splited_dir, self.role_name_right, level=self.level)
        left_split_thread.start()
        right_split_thread.start()
        left_split_thread.join()
        right_split_thread.join()
        os.remove(self.left_wav_path)
        os.remove(self.right_wav_path)
        if self.remove_origin_wav:
            os.remove(self.wav_path)

    def split_left_right(self, wav_path, left_wav_path, right_wav_path):
        """ 通道分离

            Args:
                wav_path (str): wav音频的路径
                left_wav_path (str): 左声道的wav音频路径
                right_wav_path (str): 右声道的wav音频路径

            Returns:
                None
        """
        sample_rate, wav_data = wavfile.read(wav_path)
        left = []
        right = []
        for item in wav_data:
            left.append(item[0])
            right.append(item[1])
        wavfile.write(left_wav_path, sample_rate, np.array(left))
        wavfile.write(right_wav_path, sample_rate, np.array(right))


def generate_scp(input_dir: str, scp_path: str = '',
                 type_list: tuple = ('flac', 'mp3', 'm4a', 'ogg', 'opus', 'wav', 'wma')):
    """ 产生scp文件

        Args:
            input_dir (str): 所有文件存放的目录
            scp_path (str): 产生的scp文件路径及名称
            type_list (str): 忽略type_list中文件类型以外的文件，只针对type_list中的文件类型产生scp

        Returns:
            None
    """
    input_dir = os.path.abspath(input_dir)
    type_set = set(type_list)
    id_set = set()
    if scp_path == '':
        scp_path = os.path.join(input_dir, 'wav.scp')
    with open(scp_path, 'w', encoding='utf8') as fw:
        for root, dirs, files in os.walk(input_dir):
            for name in files:
                file_path = os.path.join(root, name)
                # use file_dir/basename as wav_id
                id = '.'.join(os.path.relpath(file_path, input_dir).split('.')[0:-1])
                # file type
                file_type = file_path.split('.')[-1]
                if file_type in type_set:
                    if id not in id_set:
                        fw.write(id + ' ' + file_path + '\n')
                        id_set.add(id)
                    else:
                        print("id", id, "already exsits")


def format_and_split_audios(audio_dir: str, output_dir: str, audio_scp_path: str = '',
                            num_channels: int = 1, resample_rate: int = 16000,
                            split: bool = False,
                            split_level: tuple = (2.4, 5, 10, 16, 20, 30),
                            num_thread: int = 4,
                            ffmpeg_executable_path: str = '/usr/bin/ffmpeg',
                            type_list: tuple = ('flac', 'mp3', 'm4a', 'ogg', 'opus', 'wav', 'wma'),
                            keep_temp_file: bool = False):
    """ 格式化、切分音频并产生wav.scp

        Args:
            audio_dir (str): 音频目录
            output_dir(str): 输出目录
            audio_scp_path (str): 原音频的scp文件路径，空字符串表示没有
            num_channels (int): 转换后的通道数
            resample_rate (int): 转换后的采样率
            split (bool): 是否切分音频
            split_level (tuple): 切分严格程度等级划分,如(2.4, 5, 10, 16, 20, 30)表示0-2.4秒不切分,
                           2.4-5秒15帧里有少于1帧是人声才切分
                           5-10秒15帧里有少于2帧是人声则切分
                           10-16秒10帧里有少于2帧是人声就切分
                           16-20秒5帧里有少于2帧是人声就切分
                           20-30秒3帧里有少于2帧是人声就切分
                           30秒强制切分
            num_thread (int): 线程数
            ffmpeg_executable_path (str): ffmpeg可执行文件的路径
            type_list (tuple): 所有的待被处理的音频文件类型
            keep_temp_file (bool): 保留临时文件，包括临时的scp文件和被各式化但未被切分的音频

        Returns:
            None
    """
    # 如果没有wav.scp，则自动产生scp文件
    if audio_scp_path == '':
        audio_scp_path = os.path.join(audio_dir, 'temp.scp')
        generate_scp(audio_dir, audio_scp_path, type_list)

    audio_id_list = []
    audio_path_list = []
    with open(audio_scp_path, 'r', encoding='utf8') as fr:
        for line in fr.readlines():
            line_list = line.strip().split(' ')
            audio_id_list.append(line_list[0])
            audio_path_list.append(line_list[1])
    if not keep_temp_file and audio_scp_path == '':
        os.remove(os.path.join(audio_dir, 'temp.scp'))

    # 被格式化和被切分好的音频存放路径
    formatted_audio_path_list = ['.'.join(os.path.relpath(path, audio_dir).split('.')[0:-1]) + '.wav' for path in
                                 audio_path_list]
    formatted_audio_path_list = [os.path.join(output_dir, path) for path in formatted_audio_path_list]
    splitted_audio_dir_list = ['.'.join(path.split('.')[0:-1]) for path in formatted_audio_path_list]
    num_audios = len(audio_path_list)
    assert num_audios == len(formatted_audio_path_list)
    assert num_audios == len(splitted_audio_dir_list)

    # format audio
    print('formatting audios')
    format_audio_thread_queue = queue.Queue(num_audios)
    temp_thread_list = []
    for i in range(num_audios):
        format_audio_thread_queue.put(
            FormatSingleAudioThread(str(i + 1), audio_path_list[i], formatted_audio_path_list[i],
                                    num_channels, resample_rate,
                                    ffmpeg_executable_path))
    while not (format_audio_thread_queue.empty() and len(temp_thread_list) == 0):
        temp_thread_list = [x for x in temp_thread_list if x.is_alive()]
        # 将线程队列中的线程放入列表运行
        if len(temp_thread_list) < num_thread and not format_audio_thread_queue.empty():
            temp_thread_list.append(format_audio_thread_queue.get())
            temp_thread_list[-1].start()
            print('formatting audio', temp_thread_list[-1].thread_id + '/' + str(num_audios))
        time.sleep(0.1)
    print('all audios have been formatted')

    # split audio
    if split:
        print('splitting audios')
        split_audio_thread_queue = queue.Queue(num_audios)
        temp_thread_list = []
        if num_channels == 2:
            for i in range(num_audios):
                split_audio_thread_queue.put(SplitAndSave2ChannelWavThread(str(i + 1), formatted_audio_path_list[i],
                                                                           splitted_audio_dir_list[i],
                                                                           'left', 'right', level=split_level,
                                                                           remove_origin_wav=~keep_temp_file))
        else:
            for i in range(num_audios):
                split_audio_thread_queue.put(SplitAndSave1ChannelWavThread(str(i + 1), formatted_audio_path_list[i],
                                                                           splitted_audio_dir_list[i], role='',
                                                                           level=split_level,
                                                                           remove_origin_wav=~keep_temp_file))
        while not (split_audio_thread_queue.empty() and len(temp_thread_list) == 0):
            temp_thread_list = [x for x in temp_thread_list if x.is_alive()]
            # 将线程队列中的线程放入列表运行
            if len(temp_thread_list) < num_thread and not split_audio_thread_queue.empty():
                temp_thread_list.append(split_audio_thread_queue.get())
                temp_thread_list[-1].start()
                print('splitting audio', temp_thread_list[-1].thread_id + '/' + str(num_audios))
            time.sleep(0.1)
        print('all audios have been splitted')

    # generate wav.scp
    if not split:
        with open(os.path.join(output_dir, 'wav.scp'), 'w', encoding='utf8') as fw:
            for i in range(num_audios):
                fw.write(audio_id_list[i] + ' ' + formatted_audio_path_list[i] + '\n')
    else:
        with open(os.path.join(output_dir, 'wav.scp'), 'w', encoding='utf8') as fw:
            for i in range(num_audios):
                files = os.listdir(splitted_audio_dir_list[i])
                files.sort()
                files_basename = ['.'.join(name.split('.')[0:-1]) for name in files]
                files_path = [os.path.abspath(os.path.join(splitted_audio_dir_list[i], name)) for name in files]
                for j in range(len(files_basename)):
                    fw.write(audio_id_list[i] + '-' + files_basename[j] + ' ' + files_path[j] + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--audio_dir', default='audio_dir', help='dir of audios')
    parser.add_argument('--output_dir', default='output_dir', help='dir of formatted and splitted audios')
    parser.add_argument('--audio_scp_path', default='', help='origin audio scp path')
    parser.add_argument('--num_channels', default=1, help='channels of formatted audios')
    parser.add_argument('--resample_rate', default=16000, help='resample rate of formatted audios')
    parser.add_argument('--split', default=False, help='whether to split audio', action='store_true')
    parser.add_argument('--split_level', default="(2.4, 5, 10, 16, 20, 30)", help='split level')
    parser.add_argument('--num_thread', default=4, help='number of threads')
    parser.add_argument('--ffmpeg_executable_path', default='/usr/bin/ffmpeg', help='ffmpeg executable path')
    parser.add_argument('--type_list', default="('flac', 'mp3', 'm4a', 'ogg', 'opus', 'wav', 'wma')",
                        help='type of files selected')
    parser.add_argument('--keep_temp_file', default=False, help='whether to keep temp file', action='store_true')
    args = parser.parse_args()
    args.split_level = eval(args.split_level)
    args.type_list = eval(args.type_list)

    # args for test
    # args.audio_dir = 'wav_dir'
    # args.output_dir = 'result'
    # args.audio_scp_path = 'wav_dir/wav.scp'
    # args.ffmpeg_executable_path = './ffmpeg.exe'
    # args.num_channels = 2
    # args.split = True
    # args.keep_temp_file = True

    format_and_split_audios(args.audio_dir, args.output_dir, args.audio_scp_path,
                            args.num_channels, args.resample_rate,
                            args.split, args.split_level, args.num_thread,
                            args.ffmpeg_executable_path, args.type_list, args.keep_temp_file)
