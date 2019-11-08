import tensorflow as tf

import Utils
from Utils import LeakyReLU
import numpy as np
import functools
from tensorflow.contrib.signal.python.ops import window_ops
from modules import multihead_attention, ff, tf_multihead_attention, cnn, positional_encoding 

class UnetSpectrogramSeparator:
    '''
    U-Net separator network for singing voice separation.
    Takes in the mixture magnitude spectrogram and return estimates of the accompaniment and voice magnitude spectrograms.
    Uses "same" convolutions like in original paper
    '''

    def __init__(self, model_config):
        '''
        Initialize U-net
        :param num_layers: Number of down- and upscaling layers in the network
        '''
        self.num_layers = model_config["num_layers"]
        self.num_initial_filters = model_config["num_initial_filters"]
        self.mono = model_config["mono_downmix"]
        self.source_names = model_config["source_names"]

        assert(len(self.source_names) == 2) # Only use for acc/voice separation for now, since model gets too big otherwise
        assert(self.mono) # Only mono

        # Spectrogram settings
        self.frame_len = 1024
        self.hop = 768

    def get_padding(self, shape):
        '''
        Calculates the required amounts of padding along each axis of the input and output, so that the Unet works and has the given shape as output shape
        :param shape: Desired output shape
        :return: Padding along each axis (total): (Input frequency, input time)
        '''

        return [shape[0], shape[1], 1], [shape[0], shape[1], 1]

    def get_output(self, input, training, return_spectrogram=False, reuse=True):
        '''
        Creates symbolic computation graph of the U-Net for a given input batch
        :param input: Input batch of mixtures, 3D tensor [batch_size, num_samples, 1], mono raw audio
        :param reuse: Whether to create new parameter variables or reuse existing ones
        :Param return_spectrogram: Whether to output the spectrogram estimate or convert it to raw audio and return that
        :return: U-Net output: If return_spectrogram: Accompaniment and voice magnitudes as length-two list with two 4D tensors. Otherwise Two 3D tensors containing the raw audio estimates
        '''
        # Setup STFT computation
        window = functools.partial(window_ops.hann_window, periodic=True)
        inv_window = tf.contrib.signal.inverse_stft_window_fn(self.hop, forward_window_fn=window)
        with tf.variable_scope("separator", reuse=reuse):
            enc_outputs = list()

            # Compute spectrogram
            assert(input.get_shape().as_list()[2] == 1) # Model works ONLY on mono
            stfts = tf.contrib.signal.stft(tf.squeeze(input, 2), frame_length=self.frame_len, frame_step=self.hop, fft_length=self.frame_len, window_fn=window)
            mix_mag = tf.abs(stfts)
            mix_angle = tf.angle(stfts)

            # Input for network
            mix_mag_norm = tf.log1p(tf.expand_dims(mix_mag, 3))
            mix_mag_norm = mix_mag_norm[:,:,:-1,:] # Cut off last frequency bin to make number of frequency bins divisible by 2

            mags = dict()
            for name in self.source_names:
                current_layer = mix_mag_norm
                current_layer = tf.layers.conv2d(current_layer, 128, [3, 3], strides=[2,2], activation=None, padding='same')
                current_layer = tf.contrib.layers.batch_norm(current_layer, activation_fn=LeakyReLU, is_training=training)
                
                # Position Embedding
                current_shape = current_layer.get_shape().as_list()
                maxlen = current_shape[1] * current_shape[2]
                pos_inputs = tf.reshape(current_layer, [current_shape[0], maxlen, -1])
                pos_layer = positional_encoding(pos_inputs, maxlen, masking=False)
                pos_layer = tf.reshape(pos_layer, current_shape)
                current_layer += pos_layer
                
                # Down-convolution: Repeat pool-conv
                for i in range(self.num_layers):
                    assert(current_layer.get_shape().as_list()[1] % 2 == 0 and current_layer.get_shape().as_list()[2] % 2 == 0)
                    # block
                    current_layer = tf_multihead_attention(queries=current_layer,
                                                          keys=current_layer,
                                                          values=current_layer,
                                                          num_heads=8,
                                                          dropout_rate=0.1,
                                                          training=training,
                                                          causality=False)
                    current_layer = cnn(current_layer, training=training)

                # Compute mask
                mask = tf.layers.conv2d_transpose(current_layer, 1, [3,3], strides=[2,2], activation=tf.nn.sigmoid, padding="same")
                mask =  tf.pad(mask, [(0,0), (0,0), (0, 1), (0,0)], mode="CONSTANT", constant_values=0.5) # Pad last frequency bin of mask that is missing since we removed it in the input
                mask = tf.squeeze(mask, 3)

                # Compute source magnitudes
                source_mag = tf.multiply(mix_mag, mask)
                mags[name] = source_mag

            if return_spectrogram:
                return mags
            else:
                audio_out = dict()
                # Reconstruct audio
                for source_name in mags.keys():
                    stft = tf.multiply(tf.complex(mags[source_name], 0.0), tf.exp(tf.complex(0.0, mix_angle)))
                    audio = tf.contrib.signal.inverse_stft(stft, self.frame_len, self.hop, self.frame_len, window_fn=inv_window)

                    # Reshape to [batch_size, samples, 1]
                    audio = tf.expand_dims(audio, 2)

                    audio_out[source_name] = audio

                return audio_out
