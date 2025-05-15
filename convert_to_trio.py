#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import mido
import shutil
from tqdm import tqdm

def convert_to_trio(midi_file_path, output_dir=None, verbose=True):
    """
    将MIDI文件转换为三轨道格式，轨道0为鼓，轨道1为主旋律，轨道2为贝斯，仅修改轨道信息，保留所有MIDI事件
    
    Args:
        midi_file_path: 输入MIDI文件路径
        output_dir: 输出目录，默认为原文件同目录下的trio_midis子目录
        verbose: 是否显示详细信息
        
    Returns:
        输出文件路径
    """
    try:
        # 加载MIDI文件
        midi_data = mido.MidiFile(midi_file_path)
        track_count = len(midi_data.tracks)
        if verbose:
            print(f"\n处理MIDI文件: {os.path.basename(midi_file_path)}")
            print(f"原始轨道数量: {track_count}")
        if track_count != 3:
            if verbose:
                print(f"轨道数量不是3，无法处理: {track_count}")
            return None
        # 新建MIDI文件
        new_midi = mido.MidiFile(ticks_per_beat=midi_data.ticks_per_beat)
        # 复制三轨并重命名
        track_names = ["Drums", "Melody", "Bass"]
        for i, track in enumerate(midi_data.tracks):
            new_track = mido.MidiTrack()
            # 标记是否已写入track_name
            name_written = False
            for msg in track:
                if msg.type == 'track_name':
                    # 用新名字替换
                    new_track.append(mido.MetaMessage('track_name', name=track_names[i], time=msg.time))
                    name_written = True
                else:
                    # 如果是第0轨，且msg有channel属性且msg.channel==9，保留is_drum语义（mido本身不支持is_drum，但Magenta会按通道9识别鼓）
                    if i == 0 and hasattr(msg, 'channel'):
                        msg = msg.copy(channel=9)
                    new_track.append(msg)
            # 如果没有track_name事件，补一个
            if not name_written:
                new_track.insert(0, mido.MetaMessage('track_name', name=track_names[i], time=0))
            new_midi.tracks.append(new_track)
            if verbose:
                print(f"轨道 {i}: 设为 {track_names[i]}")
        # 输出目录
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(midi_file_path), 'trio_midis')
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.basename(midi_file_path)
        output_path = os.path.join(output_dir, filename)
        new_midi.save(output_path)
        if verbose:
            print(f"已转换为三轨MIDI文件并保存: {output_path}")
        return output_path
    except Exception as e:
        if verbose:
            print(f"处理文件 {midi_file_path} 时出错: {str(e)}")
        return None

def batch_convert_to_trio(input_dir, output_dir=None, copy_originals=False):
    """
    批量转换目录下所有MIDI文件为三轨道格式
    
    Args:
        input_dir: 输入目录，包含MIDI文件
        output_dir: 输出目录，默认在输入目录下创建trio_midis子目录
        copy_originals: 是否复制不需要转换的三轨文件到输出目录
    """
    # 如果未指定输出目录，则在输入目录下创建trio_midis子目录
    if output_dir is None:
        output_dir = os.path.join(input_dir, 'trio_midis')
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有MIDI文件
    midi_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.mid', '.midi')):
                midi_files.append(os.path.join(root, file))
    
    if not midi_files:
        print(f"在 {input_dir} 中没有找到MIDI文件")
        return
    
    print(f"找到 {len(midi_files)} 个MIDI文件")
    
    # 转换统计
    converted_count = 0
    skipped_count = 0
    error_count = 0
    copied_count = 0
    
    # 使用tqdm显示进度条
    for midi_file in tqdm(midi_files, desc="转换MIDI文件"):
        # 检查文件轨道数量
        try:
            midi_data = mido.MidiFile(midi_file)
            track_count = len(midi_data.tracks)
            # 只要轨道数为3，直接重命名重排；轨道数大于3的仍可按需处理（此处只处理3轨）
            if track_count == 3:
                result = convert_to_trio(midi_file, output_dir, verbose=False)
                if result:
                    converted_count += 1
                else:
                    error_count += 1
            elif track_count > 3:
                # 可选：如需处理多轨，按原有逻辑
                skipped_count += 1
            else:
                skipped_count += 1
        except Exception as e:
            print(f"处理文件 {midi_file} 时出错: {str(e)}")
            error_count += 1
    
    # 输出转换结果
    print("\n===== 转换完成 =====")
    print(f"总计MIDI文件: {len(midi_files)}")
    print(f"成功转换: {converted_count}")
    if copy_originals:
        print(f"已复制的三轨文件: {copied_count}")
    else:
        print(f"跳过的三轨文件: {skipped_count}")
    print(f"错误文件: {error_count}")
    print(f"输出目录: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='将MIDI文件转换为三轨格式，删除第一个轨道')
    parser.add_argument('input', help='输入的MIDI文件或包含MIDI文件的目录')
    parser.add_argument('-o', '--output', help='输出目录（默认在输入目录下创建trio_midis子目录）')
    parser.add_argument('-c', '--copy', action='store_true', help='是否复制已经是三轨的文件到输出目录')
    
    args = parser.parse_args()
    
    if os.path.isdir(args.input):
        batch_convert_to_trio(args.input, args.output, args.copy)
    elif os.path.isfile(args.input) and args.input.lower().endswith(('.mid', '.midi')):
        convert_to_trio(args.input, args.output)
    else:
        print(f"错误: 输入 '{args.input}' 不是有效的MIDI文件或目录")
        sys.exit(1)

if __name__ == "__main__":
    main() 