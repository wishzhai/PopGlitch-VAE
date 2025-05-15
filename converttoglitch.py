import os
import random
import pretty_midi
import numpy as np
import argparse
import sys

def add_extreme_pitch_bends(inst, note, base_time, duration):
    """在音符周围添加完全随机的弯音效果"""
    # 在音符持续时间内随机添加5-15个弯音事件
    num_bends = random.randint(5, 15)
    for _ in range(num_bends):
        # 完全随机的时间点
        t = random.uniform(base_time, base_time + duration)
        # 创建剧烈的弯音变化，使用更多的极端值组合
        pitch = random.choice([-8192, -6000, -4096, -2048, 0, 2048, 4096, 6000, 8191])
        inst.pitch_bends.append(pretty_midi.PitchBend(pitch=pitch, time=t))

def add_extreme_controls(inst, start_time, end_time):
    """添加剧烈和随机的控制器变化"""
    # 定义更多种类的控制器及其特殊效果范围
    controllers = [
        (1, [0, 127]),     # 调制轮 - 极端值
        (7, [0, 127]),     # 音量 - 极端值
        (10, [0, 127]),    # 声相 - 左右声道快速切换
        (11, [0, 127]),    # 表情控制 - 极端值
        (71, [0, 127]),    # 谐波含量 - 音色变化
        (74, [0, 127]),    # 亮度 - 音色变化
        (91, [0, 127]),    # 混响深度 - 空间效果
        (93, [0, 127]),    # 合唱效果 - 调制效果
        (94, [0, 127]),    # 延迟效果 - 回声效果
    ]
    
    # 随机选择要使用的控制器数量
    num_controllers = random.randint(3, len(controllers))
    selected_controllers = random.sample(controllers, num_controllers)

    # 为每个选中的控制器添加多个随机变化
    for ctrl_num, value_range in selected_controllers:
        # 每个控制器添加2-8个随机变化
        num_changes = random.randint(2, 8)
        for _ in range(num_changes):
            t = random.uniform(start_time, end_time)
            # 有50%的概率使用极端值，50%的概率使用随机值
            if random.random() < 0.5:
                value = random.choice(value_range)
            else:
                value = random.randint(min(value_range), max(value_range))
            inst.control_changes.append(
                pretty_midi.ControlChange(number=ctrl_num, value=value, time=t)
            )

def add_random_effects(inst, note):
    """为单个音符添加随机效果"""
    # 获取音符的时间范围
    start_time = note.start
    end_time = note.end
    duration = end_time - start_time
    
    # 随机决定是否添加每种效果
    effects = []
    if random.random() < 0.3:  # 30%概率添加弯音
        effects.append(lambda: add_extreme_pitch_bends(inst, note, start_time, duration))
    if random.random() < 0.3:  # 30%概率添加控制器变化
        effects.append(lambda: add_extreme_controls(inst, start_time, end_time))
    
    # 随机打乱效果的执行顺序
    random.shuffle(effects)
    for effect in effects:
        effect()

def glitch_note(note):
    """对单个音符进行glitch处理"""
    # 随机改变音符长度（通过修改end时间）
    if random.random() < 0.4:
        duration = note.end - note.start
        new_duration = duration * random.choice([0.25, 0.5, 1.5, 2.0])
        note.end = note.start + new_duration
    
    # 随机改变起始时间
    if random.random() < 0.3:
        shift = random.uniform(-0.1, 0.1)
        note.start += shift
        note.end += shift
    
    # 随机改变力度
    if random.random() < 0.5:
        note.velocity = random.choice([0, 127, random.randint(30, 120)])

def process_midi_file(input_path, output_path):
    """处理单个MIDI文件"""
    try:
    # 加载MIDI文件
    pm = pretty_midi.PrettyMIDI(input_path)
        print(f"成功加载MIDI文件，包含 {len(pm.instruments)} 个音轨")
    
    # 对每个音轨进行处理
    for inst in pm.instruments:
            # 1. 随机删除音符（更高的删除概率）
            delete_prob = 0.6  # 提高删除概率
        kept_notes = []
        for note in inst.notes:
            if random.random() > delete_prob:
                kept_notes.append(note)
        inst.notes = kept_notes
        
            # 2. 处理每个保留的音符
        for note in inst.notes:
                # 对音符进行glitch处理
                glitch_note(note)
                # 添加随机效果
                add_random_effects(inst, note)
            
            # 3. 在关键位置插入全部音符关闭事件
            if inst.notes:
                track_start = min(n.start for n in inst.notes)
                track_end = max(n.end for n in inst.notes)
                
                # 随机插入2-5次全部音符关闭
                for _ in range(random.randint(2, 5)):
                    t = random.uniform(track_start, track_end)
                    for cc in range(120, 127):
                        inst.control_changes.append(
                            pretty_midi.ControlChange(number=cc, value=0, time=t)
                        )
    
        # 保存处理后的MIDI
    pm.write(output_path)
    print(f"已生成故障风格MIDI: {output_path}")
        return True
    except Exception as e:
        print(f"处理文件时发生错误: {e}")
        return False

def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='将MIDI文件转换为Glitch风格')
    parser.add_argument('--input', '-i', help='输入的MIDI文件路径')
    parser.add_argument('--output', '-o', help='输出的MIDI文件路径')
    parser.add_argument('--batch', '-b', action='store_true', help='批量处理POP909数据集')
    args = parser.parse_args()

    if args.batch:
        # 批量处理模式
        input_dir = "./data/POP909"
        output_dir = "./data/glitch_midis"
        os.makedirs(output_dir, exist_ok=True)
        
        midi_files = []
        for item in os.listdir(input_dir):
            if item.isdigit():
                folder_path = os.path.join(input_dir, item)
                if os.path.isdir(folder_path):
                    main_midi = os.path.join(folder_path, f"{item}.mid")
                    if os.path.exists(main_midi):
                        midi_files.append(main_midi)
        
        print(f"在输入目录中找到 {len(midi_files)} 个主MIDI文件")
        for input_path in midi_files:
            filename = os.path.basename(input_path)
            output_path = os.path.join(output_dir, filename.replace(".mid", "_glitch.mid"))
            process_midi_file(input_path, output_path)
    
    elif args.input:
        # 单文件处理模式
        if not os.path.exists(args.input):
            print(f"错误：输入文件 {args.input} 不存在")
            sys.exit(1)
        
        # 如果没有指定输出路径，在输入文件名后添加_glitch
        if not args.output:
            args.output = args.input.replace(".mid", "_glitch.mid")
        
        # 处理单个文件
        if process_midi_file(args.input, args.output):
            print("处理完成！")
        else:
            print("处理失败！")
    
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
