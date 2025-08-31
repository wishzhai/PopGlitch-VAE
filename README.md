# PopGlitch-VAE
fine-turning Magenta's MusicVAE by glitched-POP909dataset
Because POP909 is a three‑track dataset, I initially planned to use the hierdec-trio_16bar checkpoint—since it ostensibly best matched POP909’s format—but I faced many challenges adapting the data. The hierdec-trio_16bar model requires exactly three inputs (drums, melody, and bass) to exploit its hierarchical decoder design. However, POP909’s three tracks are MELODY, BRIDGE, and PIANO.Even after renaming, hierdec-trio_16bar still enforces a strict drum‑track requirement, so instead we switched to the cat-mel_2bar_big checkpoint and fine‑tuned using only POP909’s separated melody track.
#  fine-tuning Strategy
In this project, we implemented a full-parameter fine-tuning strategy to maximize MusicVAE's capacity to learn glitch music characteristics.
# Requirememts
pip install magenta==2.4.1 package (tested only on Python == 3.8)
pretty_midi>=0.2.9
numpy>=1.20.0
# Training
python musicvae_glitch_full_finetune.py \
  --config=cat-mel_2bar_big \
  --run_dir=./runs/cat-mel \
  --examples_path=./mel.tfrecord \
  --num_steps=10000 \
  --mode=train \
  --hparams=batch_size=32,learning_rate=0.0005
# Result
Run `python bias_testing_framework.py` to execute comprehensive bias analysis across data representation, architecture, training objectives, and output generation stages. 
