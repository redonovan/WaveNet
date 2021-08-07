# WaveNet : implemented from paper.
# Given that I don't have phoneme/duration/f0 data, this version will just babble based on speaker id.
# Written 29,30 July & 5,6 Aug 2021.


import os
import time
import math
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
import matplotlib.pyplot as plt

# Sampling can most easily be achieved using TensorFlow Probability.
# TensorFlow Probability 0.11.1 is compatible with tensorflow 2.3.0
# pip3 --no-cache-dir install tensorflow-probability==0.11.1
# That worked!

import tensorflow_probability as tfp

# GPU memory hack
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

# set random seed
tf.random.set_seed(1)


# Hyperparameters
# model 
ulaw_dim = 256       # Number of levels in the ulaw quantisation; 256 is pretty much a standard AFAIK
nlayers = 30         # From Tacotron 2; should be a multiple of 10 whatever it is (hardcoded).
model_dim = 64       # 64 from Deep Voice; looks like Parallel Wavenet uses 512.
skip_dim = 256       # From Deep Voice
max_sid = 8975       # The maximum speaker_id found in LibriSpeech train_clean100
# training
batch_size = 1       # Probably staying at 1 because of GPU memory limitations
num_epochs = 5       # For now
max_samples = 16000  # The number of waveform samples to train on at once, limited by GPU memory
# generation
sample_rate = 16000  # From LibriSpeech
nsamples = 16000     # Number of samples to generate (it generates roughly 68 per second on my system)
gensid = 19          # speaker-id to use during generation
ofn = 'speech.wav'   # output filename for generated speech

# NB : If you've already trained a system and saved checkpoints you can skip the training loop.


# I prepared librispeechpartial as a lower-diskspace version of the LibriSpeech tfds dataset.
# Load the train_clean100 split into train_ds.

import tensorflow_datasets as tfds

builder  = tfds.builder("librispeechpartial")
info     = builder.info
# dev_ds   = builder.as_dataset(split="dev_clean")
# test_ds  = builder.as_dataset(split="test_clean")
train_ds = builder.as_dataset(split="train_clean100")


# Shuffle the training dataset.
# Buffer size 2048 adds about 4GB to the resident memory size (non GPU) taking it to 9GB.
# This slightly hurts training set accuracy at 1000 steps, but I'm leaving it in for now.
train_ds = train_ds.shuffle(2048, reshuffle_each_iteration=True, seed=1)


# transform the dataset to contain only ulaw encoded speech and speaker id

def ulaw_encode(wt): # takes waveform tensor range [-32768, +32767], returns ulaw tensor range [0,255]
    mu = 255.0  # ulaw encoding parameter standard in North America and Japan
    wf = tf.cast(wt, dtype=tf.float32)
    wf = wf / 32768.0                       # range [-1.0, +1.0)
    num = tf.math.log(1.0 + (mu * tf.math.abs(wf)))
    den = tf.math.log(1.0 + mu)
    sgn = tf.math.sign(wf)
    fwf = sgn * num / den                   # range [-1.0, +1.0)
    pqt = fwf * (ulaw_dim // 2)             # range [-128.0, +128.0)
    flr = tf.math.floor(pqt)                # range [-128.0, +127.0]
    qnt = tf.cast(flr, dtype=tf.int32)      # range [-128, +127] making 256 possible values in total
    pos = tf.math.add(qnt, ulaw_dim // 2)   # range [0, 255] with shape (wav_samples,)
    return pos


def ulaw_extract(wi):      # waveform int tensor, range [-32768, +32767]
    pos = ulaw_encode(wi)  # ulaw int tensor, range [0,255]
    # I now randomly choose a max_samples chunk of speech from within wav_samples for training.
    start_positions = tf.math.maximum(0, tf.shape(pos)[0] - max_samples)
    start = tf.random.uniform(shape=(), minval=0, maxval=start_positions+1, dtype=tf.int32, seed=1)
    rnd = pos[start:start+max_samples]      # range [0, 255] with shape (max_samples,) or less
    oh  = tf.one_hot(rnd, ulaw_dim)         # shape (max_samples, 256) or fewer samples
    return oh


@tf.autograph.experimental.do_not_convert
def transform(d):
    wi = d['speech']         # -> (wav_samples)
    wq = ulaw_extract(wi)    # -> (max_samples, ulaw_dim) or possibly fewer samples
    o  = {'ulaw' : wq, 'speaker_id' : d['speaker_id']}
    return o


# transform into the training ulaw dataset
tul_ds = train_ds.map(transform)

# filter out waveforms shorter than max_samples

def filter_lengths(d):
    oh = d['ulaw']           # one-hot ulaw shape (max_samples, 256) or fewer samples
    l  = tf.shape(oh)[0]
    return l >= max_samples

tfi_ds = tul_ds.filter(filter_lengths) # training filtered dataset
trb_ds = tfi_ds.batch(batch_size)      # training batched dataset

# TEST : get one waveform of ulaws as a 1-hot batch shape (1, max_samples, ulaw_dim)
# one = tfi_ds.take(1)
# l = list(one.as_numpy_iterator())
# ohb = tf.convert_to_tensor(l[0]['ulaw'][None,:,:])


# Model

class WavLayer(keras.layers.Layer):
    def __init__(self, dilation, model_dim, skip_dim, **kwargs):
        super(WavLayer, self).__init__(**kwargs)
        self.d = dilation
        self.r = model_dim
        self.s = skip_dim
        # The dilated convolution layer W
        self.dilated_conv = tf.keras.layers.Conv1D(filters=2*self.r,
                                                   kernel_size=2,
                                                   dilation_rate=self.d,
                                                   padding='causal',
                                                   input_shape=(None,self.r),
                                                   use_bias=True)
        # The 1x1 convs can be implemented as dense layers
        self.onebyone = tf.keras.layers.Dense(self.r)
        self.parskip  = tf.keras.layers.Dense(self.s, use_bias=False)
        # The speaker_id projection matrix V can be implemented as a dense layer
        self.project  = tf.keras.layers.Dense(2*self.r, use_bias=False)
    #
    def call(self, x, i):
        # x is embedded ulaw                       (batch, timesteps, r)
        # i is embedded speaker_id                 (batch, r)
        Wx = self.dilated_conv(x)             # -> (batch, timesteps, 2r)
        Vh = self.project(i)                  # -> (batch, 2r)
        sm = Wx + Vh[:,None,:]                # -> (batch, timesteps, 2r)
        t  = tf.math.tanh(sm[:,:,:self.r])    # -> (batch, timesteps, r)
        g  = tf.math.sigmoid(sm[:,:,self.r:]) # -> (batch, timesteps, r)
        z  = tf.math.multiply(t, g)           # -> (batch, timesteps, r)
        z2 = self.onebyone(z)                 # -> (batch, timesteps, r)
        o  = tf.math.add(z2, x)               # -> (batch, timesteps, r)
        so = self.parskip(z)                  # -> (batch, timesteps, s)
        return o, so

    
class WaveNet(tf.keras.Model):
    def __init__(self, nlayers, model_dim, skip_dim, ulaw_dim, max_sid, **kwargs):
        super(WaveNet, self).__init__(**kwargs)
        self.r = model_dim
        self.s = skip_dim
        self.u = ulaw_dim
        self.n = nlayers
        self.m = max_sid
        # create n layers, with dilations 1,2,4,...512,1,2,4,...512 and so on
        self.lays = [WavLayer(dilation=2**(k%10), model_dim=self.r, skip_dim=self.s) for k in range(self.n)]
        # create an initial 'embedding' 2x1 convolution
        self.ulaw_embed = tf.keras.layers.Conv1D(filters=self.r,
                                                 kernel_size=2,
                                                 dilation_rate=1,
                                                 padding='causal',
                                                 activation='tanh',
                                                 input_shape=(None,self.u),
                                                 use_bias=True)
        # create an embedding for the speaker_id
        self.sid_embed = tf.keras.layers.Embedding(self.m + 1, self.r, input_length=1)
        # create the biases required for the parameterized skip connections
        b_init  = tf.zeros_initializer()
        self.sb = tf.Variable(initial_value=b_init(shape=(self.s,), dtype="float32"), trainable=True)
        # create the two 1x1 (aka fully connected) layers applied to the skip outputs
        self.d1 = tf.keras.layers.Dense(self.u, activation='relu')
        self.d2 = tf.keras.layers.Dense(self.u)
        #
    def call(self, oh, sid): # input shapes (batch, timesteps, ulaw_dim), (batch,)
        # 'embed' the one-hot ulaw input
        x = self.ulaw_embed(oh) # -> (batch, timesteps, r)
        # embed the speaker id
        i = self.sid_embed(sid) # -> (batch, r)
        # put the results through all n dilated convolutional layers
        sl = []
        for k in range(self.n):
            x, sx = self.lays[k](x, i)
            sl.append(sx)
        # add the outputs of the parameterized skip connections
        st = tf.stack(sl, -1)
        ss = tf.reduce_sum(st, -1)
        # add the biases and apply the activation function
        so = tf.nn.relu(tf.math.add(ss,self.sb))
        # apply the two 1x1 convolutions to produce the softmax logits
        s1 = self.d1(so)
        s2 = self.d2(s1)
        # return the softmax logits
        return s2



# instantiate the model
wavenet = WaveNet(nlayers, model_dim, skip_dim, ulaw_dim, max_sid)

# optimizer
# Experiments have shown that a learning rate of 1e-3 is optimal at the start, but that 1e-4 is
# optimal after 5 epochs (~140k batches) of training with 1e-3.  It seems likely that that system
# could have benefited from a lower learning rate sooner.  So, here I use exponential decay, with
# a drop to 1e-4 after 100k batches (~3.5 epochs), by staircasing to 0.794 every 10,000 batches.

initial_learning_rate = 1e-3

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=10000,
    decay_rate=0.794,
    staircase=True)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

# loss function
loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

def loss_function(real, pred):
    # don't apply the loss until we've seen the whole receptive field
    recep_field = (2 ** 10) * (nlayers // 10)
    loss = loss_object(real[:, recep_field:, :], pred[:, recep_field:, :])
    return loss


# accuracy function
acc_object = tf.keras.metrics.CategoricalAccuracy()

def acc_function(real, pred):
    # don't compute accuracy until we've seen the whole receptive field
    recep_field = (2 ** 10) * (nlayers // 10)
    acc = acc_object(real[:, recep_field:, :], pred[:, recep_field:, :])
    return acc


# accumulators
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_acc  = tf.keras.metrics.Mean(name='train_accuracy')

val_loss   = tf.keras.metrics.Mean(name='val_loss')
val_acc    = tf.keras.metrics.Mean(name='val_accuracy')

# Adapted from https://www.tensorflow.org/text/tutorials/transformer:
# The @tf.function trace-compiles train_step into a TF graph for faster
# execution. The function specializes to the precise shape of the argument
# tensors. To avoid re-tracing due to the variable sequence lengths or variable
# batch sizes (the last batch is smaller), use input_signature to specify
# more generic shapes.  As it turned out, since I use batch_size=1 and I 
# filter out short speech segments I don't need the signature anymore.

train_step_signature = [
    tf.TensorSpec(shape=(None, None, ulaw_dim), dtype=tf.float32),
    tf.TensorSpec(shape=(None,), dtype=tf.int64)
]


@tf.function(input_signature=train_step_signature)
def train_step(ulaw, sid): # (batch, timesteps, ulaw_dim), (batch,)
    inp = ulaw[:, :-1, :]
    tar = ulaw[:,  1:, :]
    #
    with tf.GradientTape() as tape:
        pred = wavenet(inp, sid)
        #
        # compute the categorical crossentropy loss
        loss = loss_function(tar, pred)
    #
    # compute and apply gradients
    gradients = tape.gradient(loss, wavenet.trainable_variables)
    optimizer.apply_gradients(zip(gradients, wavenet.trainable_variables))
    #
    train_loss(loss)
    train_acc(acc_function(tar, pred))

    
@tf.function(input_signature=train_step_signature)
def val_step(ulaw, sid): # (batch, timesteps, ulaw_dim), (batch,)
    inp  = ulaw[:, :-1, :]
    tar  = ulaw[:,  1:, :]
    pred = wavenet(inp, sid)
    val_loss(loss_function(tar, pred))
    val_acc(acc_function(tar, pred))


# I'll use the dev set as validation data, validating on 1024 utterances
dev_ds    = builder.as_dataset(split="dev_clean")
dul_ds    = dev_ds.map(transform)
dfi_ds    = dul_ds.filter(filter_lengths)
deb_ds    = dfi_ds.batch(batch_size)
val_steps = 1024 // batch_size


# Create a Checkpoint that will manage objects with trackable state,
# one I name "optimizer" and the other I name "model".
checkpoint           = tf.train.Checkpoint(optimizer=optimizer, model=wavenet)
checkpoint_directory = './checkpoints'
checkpoint_prefix    = os.path.join(checkpoint_directory, "ckpt")


# Training loop.
# This can be skipped if checkpoints exist on disk for the same system defined above.
for epoch in range(num_epochs):
    start = time.time()
    #
    train_loss.reset_states()
    train_acc.reset_states()
    val_loss.reset_states()
    val_acc.reset_states()
    #
    for (b, d) in enumerate(trb_ds):
        train_step(d['ulaw'], d['speaker_id'])
        if b % 50 == 0:
            print(f'Epoch {epoch + 1} Batch {b} Train Loss {train_loss.result():.4f}',
                  f' Train Acc {train_acc.result():.4f}')
    #
    print(f'Epoch {epoch + 1} Train Loss {train_loss.result():.4f} Train Acc {train_acc.result():.4f}')
    cp = checkpoint.save(file_prefix=checkpoint_prefix)
    print('Saving checkpoint to', cp)
    #
    print('Validating', end='', flush=True)
    for (b, d) in enumerate(deb_ds):
        if b == val_steps:
            break
        val_step(d['ulaw'], d['speaker_id'])
        if b % 50 == 0:
            print('.', end='', flush=True)
    print(f'\nEpoch {epoch + 1} Val Loss {val_loss.result():.4f} Val Acc {val_acc.result():.4f}')
    #
    print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')



    
# Inference


# Load the latest checkpoint from disk
status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_directory))
status.assert_consumed()


# the buffer fed to the model must always contain enough receptive field
recep_field = (2 ** 10) * (nlayers // 10)
# initially, it can be silence
pcm = tf.zeros((recep_field,), dtype=tf.float32) # (timesteps,)
# or noise
# pcm = tf.random.uniform((recep_field,), minval=-1024, maxval=1024, dtype=tf.float32) # (timesteps,)
# or a sine wave
# frequency = 150.0     # halfway between average male pitch and average female pitch
# sw    = tf.math.sin(tf.range(recep_field, dtype=tf.float32) * 2.0 * math.pi * frequency / sample_rate)
# pcm   = sw * 1024.0   # pcm sine wave, range [-1024.0, +1024.0], (timesteps,)

# However it was generated, the initial receptive field pcm must be converted to one-hot ulaw.
ulbuf = ulaw_encode(pcm)                     # ulaw range [0,255], (timesteps,)
ohbuf = tf.one_hot(ulbuf[None, :], ulaw_dim) # one-hot ulaw, float32s, (batch, timesteps, ulaw_dim)


# prepare tensor speaker id
tensid = tf.convert_to_tensor([gensid], dtype=tf.int64)


@tf.function(input_signature=train_step_signature)
def inf_step(ulaw, sid): # (batch, timesteps, ulaw_dim), (batch,)
    pred = wavenet(ulaw, sid)
    return pred

# generated speech samples
speech = []

print('Generating speech', end='', flush=True)
# inference loop
for i in range(nsamples):
    #
    pred   = inf_step(ohbuf, tensid)  # (batch, timesteps, ulaw_dim)
    logits = pred[0, -1, :]           # (ulaw_dim,)
    #
    # To obtain the next sample it is best (I read somewhere!) to sample from this distribution
    dist   = tfp.distributions.Categorical(logits=logits)
    sample = dist.sample()
    # Append that sample to the generated speech and to the one-hot buffer (discarding 0th sample)
    speech.append(sample.numpy())
    ohbuf = tf.concat([ohbuf[:, 1:, :], tf.one_hot(sample, ulaw_dim)[None, None, :]], axis=1)
    if i % 50 == 0:
        print('.', end='', flush=True)


# The speech generated by the loop above is in ulaw format.
# It must be inverse ulawed to obtain PCM.

mu = 255.0  # ulaw encoding parameter standard in North America and Japan
ua  = tf.convert_to_tensor(speech, dtype=tf.float32)      # ulaw [0,255]
fa  = (ua - ulaw_dim // 2) / (ulaw_dim // 2)              # ulaw [-1.0, 1.0)
sgn = tf.math.sign(fa)
lom = tf.math.log(1.0 + mu)
eal = tf.math.exp(tf.math.abs(fa) * lom)
xa  = (eal - 1.0) / mu
x   = xa * sgn                                            # pcm [-1.0, 1.0)

# encode the speech into a string ready for writing to disk

at = x[:, None]
rt = tf.convert_to_tensor(sample_rate)
ew = tf.audio.encode_wav(at, rt)

# save to disk
with open(ofn, 'wb') as f:
    f.write(ew.numpy())

# and plot
plt.plot(x*32768)
plt.show()

    
    
# By hand:
# for d in trb_ds:
#     break
#
# ulaw = d['ulaw']
# inp = ulaw[:, :-1, :]
# sid = d['speaker_id']
# 
# with tf.GradientTape() as tape:
#     pred = wavenet(inp, sid)


