#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Wang Zifan

"""分离视频和音频"""
import argparse

from moviepy.editor import VideoFileClip


def split_video_and_audio(video_path: str, video_output_path: str, audio_output_path: str, num_thread: int = 8,
                          write_logfile: bool = False):
    """ 分离视频和音频

       Author: Wang Zifan
       Date: 2022/04/30

       Attributes:
           video_path (str): 输入的视频路径
           video_output_path (str): 输出的不带声音的视频路径
           audio_output_path (str): 输出的音频路径
   """
    # 提取音频
    video = VideoFileClip(video_path)
    audio = video.audio
    audio.write_audiofile(audio_output_path, write_logfile=write_logfile)
    # 获取无声的视频
    video_without_audio = video.without_audio()
    video_without_audio.write_videofile(video_output_path, write_logfile=write_logfile, threads=num_thread)
    video.close()


parser = argparse.ArgumentParser(description='')
parser.add_argument('--video_path', default='test.mp4', help='video path')
parser.add_argument('--video_output_path', default='video_silence', help='video output path')
parser.add_argument('--audio_output_path', default='test.wav', help='audiovoutput path')
parser.add_argument('--nj', help='num threads')
args = parser.parse_args()

split_video_and_audio(args.video_path, args.video_output_path, args.audio_output_path, args.nj, False)
