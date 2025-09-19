# PopGlitch-VAE

A research project that fine-tunes Google's Magenta MusicVAE model using glitched POP909 dataset to explore the model's capacity for learning experimental music characteristics.

## 🎵 Project Overview

This project investigates whether a state-of-the-art music generation model (MusicVAE) can learn and reproduce "glitch" music aesthetics through fine-tuning. We transform the clean POP909 dataset into a chaotic, glitched version and apply full-parameter fine-tuning to analyze the model's adaptation capabilities.

### Key Research Questions
- Can MusicVAE learn irregular, sparse, and chaotic musical patterns?
- What are the architectural limitations when processing experimental music?
- How do loss functions bias against unconventional musical features?

## 🔬 Methodology

### 1. Data Pipeline
```
POP909 Dataset → Melody Extraction → Glitch Transformation → MusicVAE Training
     (909 songs)      (single track)      (chaos effects)      (fine-tuning)
```

### 2. Glitch Transformations
Our `converttoglitch.py` applies several chaotic transformations:
- **Note Deletion**: Randomly removes 60% of notes to create sparsity
- **Extreme Pitch Bends**: Adds random pitch bend events (-8192 to +8191)
- **Controller Chaos**: Injects random MIDI controller changes (9 types)
- **Temporal Displacement**: Shifts note timing by ±0.1 seconds
- **Velocity Randomization**: Creates extreme velocity contrasts

### 3. Training Strategy
- **Model**: MusicVAE `cat-mel_2bar_big` checkpoint
- **Strategy**: Full-parameter fine-tuning (all 57 variables trainable)
- **Rationale**: Maximize model's capacity to learn glitch characteristics
- **Data**: Balanced clean vs. glitched melody tracks

## 📁 Project Structure

```
PopGlitch-VAE/
├── README.md                          # This file
├── musicvae_glitch_full_finetune.py   # Main training script with full-parameter strategy
├── extract_melody.py                  # Extracts melody tracks from POP909 multi-track files
├── converttoglitch.py                 # Transforms clean MIDI to glitch style
├── bias_testing_framework.py          # Comprehensive analysis of training biases
├── experimental_pipeline.svg          # Visual workflow diagram
├── data_mel/                          # Processed melody data
│   ├── good_midis/                    # Clean melody tracks (909 files)
│   └── glitch_midis/                  # Glitched melody tracks (909 files)
└── 10000ckpt/                        # Fine-tuned model checkpoint
    ├── model.ckpt-10000.data-00000-of-00001
    ├── model.ckpt-10000.index
    └── model.ckpt-10000.meta
```

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8 (tested and recommended)
- TensorFlow 2.x compatible environment

### Dependencies
```bash
pip install magenta==2.4.1
pip install pretty_midi>=0.2.9
pip install numpy>=1.20.0
pip install tensorflow>=2.0.0
```

### Data Preparation
1. **Obtain POP909 Dataset**: Download from [POP909 official repository](https://github.com/music-x-lab/POP909-Dataset)
2. **Extract Melody Tracks**:
   ```bash
   python extract_melody.py ./path/to/POP909 ./data_mel/good_midis
   ```
3. **Generate Glitched Data**:
   ```bash
   python converttoglitch.py --batch
   ```

## 🚀 Usage

### Step 1: Data Processing
Extract melody tracks from your POP909 dataset:
```bash
python extract_melody.py ./data/POP909 ./data_mel
```

### Step 2: Generate Glitched Versions
Create chaotic transformations of the clean data:
```bash
python converttoglitch.py --batch
```

This will process all files in `./data_mel/good_midis/` and create glitched versions in `./data_mel/glitch_midis/`.

You can also process single files:
```bash
python converttoglitch.py --input song.mid --output song_glitch.mid
```

### Step 3: Prepare Training Data
Convert MIDI files to TFRecord format for MusicVAE training:
```bash
# You'll need to use Magenta's data conversion tools
# Example command (adjust paths as needed):
convert_dir_to_note_sequences \
  --input_dir=./data_mel \
  --output_file=./mel.tfrecord \
  --recursive
```

### Step 4: Fine-tune MusicVAE
Run full-parameter fine-tuning:
```bash
python musicvae_glitch_full_finetune.py \
  --config=cat-mel_2bar_big \
  --run_dir=./runs/cat-mel \
  --examples_path=./mel.tfrecord \
  --num_steps=10000 \
  --mode=train \
  --hparams=batch_size=32,learning_rate=0.0005
```

### Step 5: Analyze Results
Run comprehensive bias analysis:
```bash
python bias_testing_framework.py
```

This will analyze:
- Training data balance between clean and glitched samples
- Feature learning difficulty across different glitch transformations
- Model's capacity for understanding chaotic patterns
- Loss function biases against irregular features

## 📊 Key Findings

Our analysis reveals several systematic limitations:

### Architectural Limitations
- **MIDI Controller Loss**: MusicVAE only processes note sequences, discarding controller data where most glitch effects reside
- **Feature Incompatibility**: Pitch bends, CC messages, and other glitch elements are lost at the input stage

### Training Biases
- **Reconstruction Loss Bias**: Penalizes sparse and irregular patterns, favoring coherent sequences
- **KL Divergence Pressure**: Pushes latent representations toward standard distributions, suppressing extreme patterns

### Data Processing Issues
- **Information Bottleneck**: Critical glitch features are filtered out before reaching the neural network
- **Quantization Effects**: Temporal irregularities are normalized during preprocessing

## 🎯 Research Implications

This project demonstrates that current music AI architectures have fundamental limitations when handling experimental music:

1. **Input Representation Matters**: Models can only learn what they can perceive
2. **Loss Function Design**: Standard objectives inherently bias against chaos and irregularity  
3. **Architecture Specificity**: Models designed for conventional music struggle with experimental aesthetics

## 🔬 Technical Details

### Full-Parameter Training Strategy
Unlike typical fine-tuning that freezes certain layers, our approach makes all 57 model variables trainable:
- **Encoder BiLSTM**: Re-learns sparse note pattern encoding
- **Latent Space**: Remaps clean↔glitch distribution boundaries  
- **Decoder LSTM**: Adapts to irregular temporal generation patterns
- **Output Layers**: Learns extreme note combinations

### Glitch Feature Analysis
Our transformations target different aspects of musical structure:
- **Sparsification** (60% deletion): Tests model's handling of discontinuous sequences
- **Extreme Controllers**: Challenges input processing capabilities  
- **Temporal Chaos**: Evaluates quantization robustness
- **Pitch Extremes**: Explores harmonic boundary conditions

## 📈 Usage Tips

- **Model Loading**: Use the provided checkpoint at `./10000ckpt/model.ckpt-10000`
- **Generation**: Standard MusicVAE generation techniques apply
- **Interpolation**: Test clean↔glitch interpolations to evaluate learning
- **Analysis**: Run bias testing framework to understand model behavior

## 🤝 Contributing

This is a research project exploring AI music generation limitations. Contributions welcome for:
- Alternative glitch transformation strategies
- Different loss function designs  
- Architecture modifications for experimental music
- Additional bias analysis methods



## 🙏 Acknowledgments

- Google Magenta team for MusicVAE architecture
- POP909 dataset creators
- Pretty_MIDI library maintainers

---

*This project demonstrates that current AI music models have systematic biases against experimental music, revealing important limitations in how we design and evaluate creative AI systems.*