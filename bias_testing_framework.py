#!/usr/bin/env python3


import os
import numpy as np
import pretty_midi
from magenta.music import midi_io
from magenta.models.music_vae import configs
from magenta.models.music_vae.trained_model import TrainedModel
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

class TrainingDiagnostics:
    def __init__(self):
        self.config = configs.CONFIG_MAP['cat-mel_2bar_big']
        
    def analyze_training_data_balance(self):
        """Analyze training data balance"""
        print("🔍 Training Data Balance Analysis")
        print("=" * 50)
        
        clean_dir = "data_mel/good_midis"
        glitch_dir = "data_mel/glitch_midis"
        
        # Count files
        clean_files = len([f for f in os.listdir(clean_dir) if f.endswith('.mid')]) if os.path.exists(clean_dir) else 0
        glitch_files = len([f for f in os.listdir(glitch_dir) if f.endswith('.mid')]) if os.path.exists(glitch_dir) else 0
        
        print(f"📊 Data Counts:")
        print(f"   Clean files: {clean_files}")
        print(f"   Glitch files: {glitch_files}")
        print(f"   Ratio: {clean_files}:{glitch_files}")
        
        if clean_files > 0 and glitch_files > 0:
            ratio = clean_files / glitch_files
            print(f"   Clean/Glitch ratio: {ratio:.2f}")
            
            if ratio > 2.0:
                print("⚠️ Severe data imbalance! Clean data dominates")
                print("   Suggestion: Increase glitch data or reduce clean data")
            elif ratio < 0.5:
                print("⚠️ Too much glitch data, may damage basic musical understanding")
        
        return clean_files, glitch_files
    
    def analyze_feature_differences(self):
        """Analyze learning difficulty of clean vs glitch feature differences"""
        print(f"\n🧠 Feature Difference Learning Difficulty Analysis")
        print("=" * 50)
        
        # Simulated analysis (based on our known glitch transformations)
        transformations = {
            "Note Deletion": {
                "Change Magnitude": "High (60% deletion)",
                "Learning Difficulty": "Difficult",
                "Reason": "Sparsification breaks sequence continuity"
            },
            "Pitch Bend": {
                "Change Magnitude": "Extreme (±8192)",
                "Learning Difficulty": "Extremely Difficult", 
                "Reason": "MusicVAE does not process controller information"
            },
            "Controller Chaos": {
                "Change Magnitude": "Extreme (9 CC types)",
                "Learning Difficulty": "Impossible",
                "Reason": "Model architecture does not support CC data"
            },
            "Temporal Displacement": {
                "Change Magnitude": "Medium (±0.1s)",
                "Learning Difficulty": "Medium",
                "Reason": "Can be handled through quantization"
            }
        }
        
        print("🎯 Glitch Feature Analysis:")
        for feature, info in transformations.items():
            print(f"   {feature}:")
            print(f"     Change Magnitude: {info['Change Magnitude']}")
            print(f"     Learning Difficulty: {info['Learning Difficulty']}")
            print(f"     Reason: {info['Reason']}")
        
        # Key findings
        print(f"\n💡 Key Findings:")
        print(f"   ❌ MusicVAE only processes note sequences, not controllers")
        print(f"   ❌ Most glitch features are lost at model input stage")
        print(f"   ❌ Even with perfect training, controller features cannot be reproduced")
    
    def test_model_capacity_for_chaos(self):
        """Test model's capacity for understanding chaotic features"""
        print(f"\n🎪 Model Chaos Understanding Capacity Test")
        print("=" * 50)
        
        try:
            # Load fine-tuned model
            model = TrainedModel(
                self.config,
                batch_size=1,
                checkpoint_dir_or_path="./10000ckpt/model.ckpt-10000"
            )
            
            print("✅ Model loaded successfully")
            
            # Create test sequences
            import note_seq
            
            # Normal sequence
            normal_seq = note_seq.NoteSequence()
            normal_seq.tempos.add().qpm = 120
            time_sig = normal_seq.time_signatures.add()
            time_sig.numerator = 4
            time_sig.denominator = 4
            time_sig.time = 0
            
            # Add regular notes
            for i in range(8):
                note = normal_seq.notes.add()
                note.pitch = 60 + (i % 4)
                note.velocity = 80
                note.start_time = i * 0.25
                note.end_time = (i + 1) * 0.25
                note.instrument = 0
            
            normal_seq.total_time = 2.0
            
            # Chaotic sequence (simulating glitch)
            chaos_seq = note_seq.NoteSequence()
            chaos_seq.CopyFrom(normal_seq)
            
            # Randomly delete 50% of notes
            import random
            notes_to_keep = random.sample(list(chaos_seq.notes), len(chaos_seq.notes)//2)
            del chaos_seq.notes[:]
            for note in notes_to_keep:
                chaos_seq.notes.append(note)
            
            # Add random notes
            for i in range(3):
                note = chaos_seq.notes.add()
                note.pitch = random.randint(40, 80)
                note.velocity = random.randint(20, 127)
                note.start_time = random.uniform(0, 2.0)
                note.end_time = note.start_time + random.uniform(0.1, 0.5)
                note.instrument = 0
            
            print(f"📊 Test Sequences:")
            print(f"   Normal sequence: {len(normal_seq.notes)} notes")
            print(f"   Chaotic sequence: {len(chaos_seq.notes)} notes")
            
            # Test encoding capability
            try:
                quantized_normal = note_seq.quantize_note_sequence(normal_seq, steps_per_quarter=4)
                quantized_chaos = note_seq.quantize_note_sequence(chaos_seq, steps_per_quarter=4)
                
                # Attempt encoding
                _, mu_normal, _ = model.encode([quantized_normal])
                _, mu_chaos, _ = model.encode([quantized_chaos])
                
                # Calculate latent space distance
                distance = np.linalg.norm(mu_normal - mu_chaos)
                print(f"   Latent space distance: {distance:.4f}")
                
                if distance < 0.1:
                    print("❌ Model cannot distinguish between normal and chaotic sequences")
                elif distance < 0.5:
                    print("⚠️ Model has weak discrimination ability")
                else:
                    print("✅ Model can distinguish different sequences")
                
                # Test interpolation
                interpolated_z = 0.5 * mu_normal + 0.5 * mu_chaos
                result = model.decode(interpolated_z, length=32)
                
                if result and len(result) > 0:
                    print(f"✅ Interpolation successful, generated {len(result[0].notes)} notes")
                    
                    # Analyze interpolation result features
                    result_notes = len(result[0].notes)
                    if result_notes < 4:
                        print("🎯 Interpolation result is sparse, may have learned some chaotic features")
                    elif result_notes > 6:
                        print("⚠️ Interpolation result too regular, biased toward normal sequences")
                    else:
                        print("✅ Interpolation result balanced")
                else:
                    print("❌ Interpolation failed")
                    
            except Exception as e:
                print(f"❌ Encoding/decoding test failed: {e}")
                
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
    
    def analyze_loss_function_bias(self):
        """Analyze loss function bias against glitch features"""
        print(f"\n⚖️ Loss Function Bias Analysis")
        print("=" * 50)
        
        print(f"🔍 MusicVAE Loss Function Components:")
        print(f"   1. Reconstruction Loss")
        print(f"      - Objective: Minimize input-output differences")
        print(f"      - Bias: Tends to generate 'normal' sequences")
        print(f"      - Impact on glitch: Penalizes sparse and irregular features")
        
        print(f"   2. KL Divergence Loss")
        print(f"      - Objective: Regularize latent space")
        print(f"      - Bias: Pushes latent vectors toward standard distribution")
        print(f"      - Impact on glitch: Suppresses extreme latent representations")
        
        print(f"💡 Problem Diagnosis:")
        print(f"   ❌ Loss functions naturally bias toward coherence")
        print(f"   ❌ No dedicated reward term for 'chaotic' features")
        print(f"   ❌ Glitch features treated as 'errors' and penalized during training")
        
        print(f"\n🔧 Possible Technical Solutions (Not Implemented):")
        print(f"   1. Add glitch reward term to loss function")
        print(f"   2. Use adversarial training to distinguish clean/glitch")
        print(f"   3. Use different loss weights for clean/glitch samples")
        print(f"\n📝 Note: This study focuses on bias detection, not solution implementation.")
        print(f"   The goal is to expose systematic exclusion mechanisms, not to fix them.")
        print(f"   Technical solutions would address symptoms, not root philosophical issues.")
    
    def comprehensive_diagnosis(self):
        """Comprehensive diagnosis report"""
        print("🔬 MusicVAE Glitch Learning Failure Comprehensive Diagnosis")
        print("=" * 80)
        
        # Run all analyses
        clean_count, glitch_count = self.analyze_training_data_balance()
        self.analyze_feature_differences()
        self.test_model_capacity_for_chaos()
        self.analyze_loss_function_bias()
        
        # Comprehensive conclusion
        print(f"\n🎯 Comprehensive Diagnosis Conclusion")
        print("=" * 50)
        
        print(f"🔴 Main Problems:")
        print(f"   1. Architecture limitation: MusicVAE does not process MIDI controllers")
        print(f"   2. Feature loss: Most glitch features disappear at input stage")
        print(f"   3. Loss bias: Loss functions penalize irregular features")
        print(f"   4. Data imbalance: Clean data may dominate")
        
        print(f"\n✅ Solution Recommendations:")
        print(f"   1. Modify architecture to support controller data")
        print(f"   2. Design glitch-aware loss functions")
        print(f"   3. Use adversarial training")
        print(f"   4. Balance training data")
        print(f"   5. Consider post-processing to add glitch features")

def main():
    print("🔬 Training Issue Diagnosis Tool")
    print("=" * 50)
    print("Objective: Analyze why fine-tuned model lacks glitch features")
    print("=" * 50)
    
    diagnostics = TrainingDiagnostics()
    diagnostics.comprehensive_diagnosis()

if __name__ == "__main__":
    main() 