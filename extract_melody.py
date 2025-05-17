#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
提取MIDI文件中的旋律轨道
"""
import os
import sys
import shutil
import pretty_midi
import numpy as np
from collections import defaultdict

def ensure_dir(directory):
    """确保目录存在，如果不存在则创建"""
    if not os.path.exists(directory):
        print(f"创建目录: {directory}")
        os.makedirs(directory)

def identify_melody_track(midi_data):
    """
    识别MIDI文件中最可能是旋律的轨道
    
    识别策略:
    1. 优先考虑名称包含"melody"、"旋律"等关键词的轨道
    2. 按音符密度排序
    3. 优先选择中高音区域的轨道
    4. 排除明显是打击乐的轨道
    """
    # 收集每个轨道的信息
    track_info = []
    melody_keywords = ["melody", "旋律", "主题", "主奏", "solo", "lead", "main"]
    
    for i, instrument in enumerate(midi_data.instruments):
        # 跳过空轨道
        if len(instrument.notes) == 0:
            continue
            
        # 跳过打击乐轨道
        if instrument.is_drum:
            continue
            
        # 计算音符密度
        total_duration = midi_data.get_end_time()
        note_density = len(instrument.notes) / total_duration if total_duration > 0 else 0
        
        # 计算平均音高
        pitches = [note.pitch for note in instrument.notes]
        avg_pitch = sum(pitches) / len(pitches) if pitches else 0
        
        # 计算音高变化率
        pitch_changes = [abs(pitches[i] - pitches[i-1]) for i in range(1, len(pitches))]
        avg_pitch_change = sum(pitch_changes) / len(pitch_changes) if pitch_changes else 0
        
        # 检查轨道名称是否包含旋律关键词
        name_score = 0
        name_lower = instrument.name.lower() if instrument.name else ""
        for keyword in melody_keywords:
            if keyword.lower() in name_lower:
                name_score = 10  # 给予很高的初始分数
                break
        
        # 收集轨道信息
        track_info.append({
            'index': i,
            'name': instrument.name,
            'program': instrument.program,
            'is_drum': instrument.is_drum,
            'num_notes': len(instrument.notes),
            'note_density': note_density,
            'avg_pitch': avg_pitch,
            'avg_pitch_change': avg_pitch_change,
            'name_score': name_score,
            'instrument': instrument
        })
    
    if not track_info:
        return None
        
    # 分数计算：为每个特征赋权重
    max_density = max(t['note_density'] for t in track_info) if track_info else 1
    
    # 修复：避免NumPy数组条件判断
    has_pitch_change = any(t.get('avg_pitch_change', 0) > 0 for t in track_info)
    if has_pitch_change:
        max_pitch_change = max(t.get('avg_pitch_change', 0) for t in track_info if t.get('avg_pitch_change', 0) > 0)
    else:
        max_pitch_change = 1
    
    for track in track_info:
        # 名称分数已经在上面计算
        density_score = 0.4 * track['note_density'] / max_density if max_density > 0 else 0
        
        # 修复：使用明确的条件判断
        pitch_score = 0.3 * (track['avg_pitch'] - 40) / 40 if track['avg_pitch'] > 40 else 0
        
        # 修复：检查避免除以零
        change_score = 0.3 * (track['avg_pitch_change'] / max_pitch_change) if max_pitch_change > 0 and track.get('avg_pitch_change', 0) > 0 else 0
        
        # 计算总分，名称匹配有很高的权重
        melody_score = track['name_score'] + density_score + pitch_score + change_score
        track['melody_score'] = melody_score
        
        print(f"轨道 {track['index']}: {track['name']} - 分数: {melody_score:.2f} (名称:{track['name_score']:.1f}, 密度:{density_score:.2f}, 音高:{pitch_score:.2f}, 变化:{change_score:.2f})")
    
    # 按旋律可能性评分排序
    track_info.sort(key=lambda x: x['melody_score'], reverse=True)
    
    # 返回最有可能是旋律的轨道
    return track_info[0]['instrument'] if track_info else None

def extract_melody(input_file, output_file):
    """提取MIDI文件中的旋律轨道并保存为新文件"""
    try:
        # 加载MIDI文件
        midi_data = pretty_midi.PrettyMIDI(input_file)
        
        # 识别旋律轨道
        melody_track = identify_melody_track(midi_data)
        
        if melody_track is None:
            print(f"警告：在文件 {input_file} 中没有找到有效的旋律轨道")
            return False
            
        # 创建新的MIDI文件，仅包含旋律轨道
        tempo_changes = midi_data.get_tempo_changes()
        # 修复：检查避免索引错误
        initial_tempo = 120.0  # 默认值
        if len(tempo_changes) > 1 and len(tempo_changes[1]) > 0:
            initial_tempo = tempo_changes[1][0]
            
        new_midi = pretty_midi.PrettyMIDI(
            resolution=midi_data.resolution, 
            initial_tempo=initial_tempo
        )
        
        # 复制旋律轨道
        new_instrument = pretty_midi.Instrument(
            program=melody_track.program,
            is_drum=False,
            name="Melody"
        )
        
        # 复制音符
        for note in melody_track.notes:
            new_instrument.notes.append(pretty_midi.Note(
                velocity=note.velocity,
                pitch=note.pitch,
                start=note.start,
                end=note.end
            ))
            
        # 添加到新MIDI
        new_midi.instruments.append(new_instrument)
        
        # 保存新文件
        new_midi.write(output_file)
        return True
        
    except Exception as e:
        print(f"处理文件 {input_file} 时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def process_directory(input_dir, output_dir):
    """处理文件夹中的所有MIDI文件"""
    # 确保输出目录存在
    ensure_dir(output_dir)
    
    # 统计信息
    stats = {
        'total': 0,
        'success': 0,
        'failed': 0
    }
    
    # 遍历输入目录
    for root, _, files in os.walk(input_dir):
        for file in files:
            # 检查是否是MIDI文件
            if file.lower().endswith(('.mid', '.midi')):
                stats['total'] += 1
                
                # 构建输入路径
                input_path = os.path.join(root, file)
                
                # 构建输出路径，保持目录结构
                rel_path = os.path.relpath(root, input_dir)
                output_subdir = os.path.join(output_dir, rel_path)
                ensure_dir(output_subdir)
                output_path = os.path.join(output_subdir, file)
                
                # 提取旋律轨道
                print(f"处理: {input_path}")
                if extract_melody(input_path, output_path):
                    stats['success'] += 1
                else:
                    stats['failed'] += 1
                    
    # 打印统计信息
    print("\n处理完成!")
    print(f"总文件数: {stats['total']}")
    print(f"成功处理: {stats['success']}")
    print(f"处理失败: {stats['failed']}")

def analyze_sequence(sequence):
    """分析NoteSequence的详细信息"""
    print(f"\n分析序列:")
    print(f"总时长: {sequence.total_time} 秒")
    print(f"拍号: {sequence.time_signatures[0].numerator}/{sequence.time_signatures[0].denominator}" if sequence.time_signatures else "未指定拍号")
    print(f"速度: {sequence.tempos[0].qpm} BPM" if sequence.tempos else "未指定速度")
    
    # 计算小节数
    if sequence.time_signatures and sequence.tempos:
        beats_per_bar = sequence.time_signatures[0].numerator
        qpm = sequence.tempos[0].qpm
        total_bars = (sequence.total_time * qpm / 60) / beats_per_bar
        print(f"估计小节数: {total_bars:.2f}")
    
    # 分析轨道信息
    instruments = {}
    for note in sequence.notes:
        if note.instrument not in instruments:
            instruments[note.instrument] = []
        instruments[note.instrument].append(note)
    
    print(f"包含 {len(instruments)} 个独特的乐器轨道:")
    for instrument, notes in instruments.items():
        print(f"  乐器 {instrument}: {len(notes)} 个音符")
        if notes:
            # 安全地计算最小/最大值，避免NumPy数组的问题
            start_times = [note.start_time for note in notes]
            end_times = [note.end_time for note in notes]
            pitches = [note.pitch for note in notes]
            
            min_start = min(start_times) if start_times else 0
            max_end = max(end_times) if end_times else 0
            min_pitch = min(pitches) if pitches else 0
            max_pitch = max(pitches) if pitches else 0
            
            print(f"    时间范围: {min_start:.2f}s - {max_end:.2f}s")
            print(f"    音高范围: {min_pitch}-{max_pitch}")
            print(f"    是否为鼓: {notes[0].is_drum}")
            print(f"    乐器类型: {notes[0].program}")

if __name__ == "__main__":
    # 设置默认目录
    input_dir = "./data"
    output_dir = "./data_mel"
    
    # 如果提供了命令行参数，则使用
    if len(sys.argv) > 1:
        input_dir = sys.argv[1]
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
        
    # 确保输入目录存在
    if not os.path.exists(input_dir):
        print(f"错误: 输入目录 '{input_dir}' 不存在")
        sys.exit(1)
        
    # 处理目录
    process_directory(input_dir, output_dir) 