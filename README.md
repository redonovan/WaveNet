# WaveNet
<b>Implementation</b>

Implemented in TensorFlow 2.3.0 directly from the WaveNet paper (<a href=https://arxiv.org/abs/1609.03499>van den Oord, et al. 2016</a>) with a few more pointers from the Deep Voice (<a href=https://arxiv.org/abs/1702.07825>Arik et al. 2017</a>) and Tacotron 2 (<a href=https://arxiv.org/abs/1712.05884>Shen et al. 2018</a>) papers.  My version was trained for 5 epochs on the train-clean-100 split of the <a href=https://www.openslr.org/12>LibriSpeech corpus</a>.  It was conditioned on speaker-id only since I don't have liguistic / duration / f0 data.  

This is an example of the speech generated by my implementation (for speaker-id 19, which is in the training data):<BR>
The audio is available <a href=https://github.com/redonovan/WaveNet/blob/main/Speech_19.wav>here</a>.

![generated speech waveform picture](https://github.com/redonovan/WaveNet/blob/main/Speech_19.png)

