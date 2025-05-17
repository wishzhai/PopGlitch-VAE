# PopGlitch-VAE
fine-turning Magenta's MusicVAE by glitched-POP909dataset
Because POP909 is a three‑track dataset, I initially planned to use the hierdec-trio_16bar checkpoint—since it ostensibly best matched POP909’s format—but I faced many challenges adapting the data. The hierdec-trio_16bar model requires exactly three inputs (drums, melody, and bass) to exploit its hierarchical decoder design. However, POP909’s three tracks are MELODY, BRIDGE, and PIANO.Even after renaming, hierdec-trio_16bar still enforces a strict drum‑track requirement, so instead we switched to the cat-mel_2bar_big checkpoint and fine‑tuned using only POP909’s separated melody track.
#  Freezing Strategy
In this project, I implemented an innovative layer freezing strategy to optimize the MusicVAE model for digital score generation. I applied transfer learning techniques by selectively freezing the first BiLSTM/RNN layer of the encoder and the fundamental layers of the decoder, while keeping the latent space fully trainable. This approach preserves the pre-trained model's advantages in low-level feature extraction while allowing targeted optimization in high-level feature representation and musical structure generation. Notably, this freezing strategy significantly improves the model's interpolation capabilities between good/bad samples, enabling the generated music to maintain structural coherence while expressing richer musical semantics and emotional characteristics.
# Requirememts
pip install magenta==2.4.1 package (tested only on Python == 3.8)
pretty_midi>=0.2.9
numpy>=1.20.0

