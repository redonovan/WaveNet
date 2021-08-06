# WaveNet
<b>Implementation</b>

Implemented in TensorFlow 2.3.0 directly from the WaveNet paper (van den Oord, et al. 2016) with a few more pointers from the Deep Voice (Arik et al. 2017) and Tacotron 2 (Shen et al. 2018) papers.  My version was trained for 5 epochs on the train_clean-100 split of the <a href=https://www.openslr.org/12>Librispeech corpus</a>.  It was conditioned on speaker-id only since I don't have liguistic / duration / f0 data.  


