# Ling Ling

Ling Ling is a project aiming to build a deep generative model to generate nice (very subjective, I know) Piano music.

## Status

6 Dec 2021

DINO is not build for generation task (or it is, but my data is not enough), and vq vae seems required more data than I have in order to learn and produce meaningful results.
In general, I suspect the poor performance of current state might caused by limitation of data although there exists ~12 hours of music file, which is quite a lot for any human to learn and analyze.
I will be looking for other more data-efficient model.

19 Dec 2021

PianoBERT seems working well (listen to [midi files](samples/)).
Using MSE Loss and quantize the outputs to the initial embedding produces really bad results, playing same note over and over.
However, using different heads after the last layer to predict each properties of a note dramatically improves the result.
The model is able to reconstruct (encode and decode) the song and produce some interesting melody when certain notes are masked. [Example](samples/Good-3.mid)
Looks like attention and specific property output head are what I need (maybe), this might explains why the previous models perform badly.
The use of Absolute Positional Encoding product slightly worse model.
It is unclear regarding the generative ability of PianoBERT at the current stage.
I tried mask all the notes, ended up receive garbage.

## TODO

- [x] More efficient data Loader
- [x] Learning Rate scheduler
- [x] Loss scheduler
- [x] Momentum scheduler
- [x] ~~Argparser~~ Config file (I dont like argparser, using a config file is probably easier)
- [x] TensorBoard Visualization
- [x] Attention Visualizer
- [x] Refactoring
- [x] Unit Test (Maybe I need more of this)
