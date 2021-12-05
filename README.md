# Ling Ling

Ling Ling v1 is a generative deep learning model that involves transformer encoders (Feature Learning) and VQ-VAE (Generative).
Transformer Encoders are trained using DINO Self-Supervised Learning method to learn features in the given dataset. 

## Status

DINO is not build for generation task (or it is, but my data is not enough), and vq vae seems required more data than I have in order to learn and produce meaningful results. In general, I suspect the poor performance of current state might caused by limitation of data although there exists ~12 hours of music file, which is quite a lot for any human to learn and analyze. I will be looking for other more data-efficient model.

## TODO

- [x] Data Augmentation (for ablation study)
  - [x] Local Global View
  - [ ] Note masking
- [x] More efficient data Loader
- [x] Learning Rate scheduler
- [x] Loss scheduler
- [x] Momentum scheduler
- [ ] Argparser
- [x] TensorBoard Visualization
- [x] Attention Visualizer
