import scipy
from scipy.io import wavfile
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st


### Top area ###

st.title("Spectrogram Visualizer for Wave Audio Files")
st.write("Brought to you by Sergiu Craioveanu and Paraschiva Mihai")


st.markdown("## Try out our example samples!")
option = st.selectbox("Pick your sample!", options=["Piano Music", "Coffin Dance"])

# st.markdown("### Or upload your own samples!")
# """TODO"""

if option == "Piano Music":
    # Read the clean .wav file
    sample_rate, data = wavfile.read('samples/piano_short.wav')

    # Read the noisy wav file
    sample_rate_noisy, data_noisy = wavfile.read('samples/piano_short_noisy.wav')

    st.markdown('## Check out the first sample!')
    audio_file = open("samples/piano_short.wav", 'rb')
    audio_bytes = audio_file.read()
    st.audio(audio_bytes, format='audio/ogg')

    st.markdown('## Check out the second sample!')
    audio_file = open("samples/piano_short_noisy.wav", 'rb')
    audio_bytes = audio_file.read()
    st.audio(audio_bytes, format='audio/ogg')

# Read channels for first sample
first_channel = data[:, 0]
second_channel = data[:, 1]

# Spectrogram of First sample, First channel
sample_freq_fs_fc, segment_time_fs_fc, spec_data_fs_fc = scipy.signal.spectrogram(first_channel, sample_rate)

# Spectrogram of First sample, Second Channel
sample_freq_fs_sc, segment_time_fs_sc, spec_data_fs_sc = scipy.signal.spectrogram(second_channel, sample_rate)
# Note sample_rate and sampling frequency values are same but theoretically they are different measures

# Read channels for second sample
first_channel_noisy = data_noisy[:, 0]
second_channel_noisy = data_noisy[:, 1]

## Plots First Sample ##
#####################################################################
st.markdown('# First Sample Spectogram and Frequency Analysis')
fig, ax = plt.subplots(2,2, figsize=(15, 10))
fig.tight_layout(pad=6.0)

fig.suptitle('First Sample', size = 17)
ax[0, 0].plot(first_channel)
ax[0, 0].set_xlabel(xlabel='Sample', size=12)
ax[0, 0].set_ylabel(ylabel='Amplitude', size=12)
ax[0, 0].set_title("First Channel", fontsize=15)
ax[0, 0].tick_params(axis='both', which='both', labelsize=12)


ax[1, 0].specgram(first_channel, Fs=sample_rate)
ax[1, 0].set_xlabel(xlabel='Time [sec]', size=12)
ax[1, 0].set_ylabel(ylabel='Frequency [Hz]', size=12)
ax[1, 0].tick_params(axis='both', which='both', labelsize=12)

ax[0, 1].plot(second_channel)
ax[0, 1].set_xlabel(xlabel='Sample', size=12)
ax[0, 1].set_ylabel(ylabel='Amplitude', size=12)
ax[0, 1].set_title("Second Channel", fontsize=15)
ax[0, 1].tick_params(axis='both', which='both', labelsize=12)

ax[1, 1].specgram(second_channel, Fs=sample_rate)
ax[1, 1].set_xlabel(xlabel='Time [sec]', size=12)
ax[1, 1].set_ylabel(ylabel='Frequency [Hz]', size=12)
ax[1, 1].tick_params(axis='both', which='both', labelsize=12)
st.pyplot()

## Plot Second Sample ##
################################################################
st.markdown('# Second Sample Spectogram and Frequency Analysis')
fig, ax = plt.subplots(2,2, figsize=(15, 10))
fig.tight_layout(pad=6.0)

fig.suptitle('Second Sample', size = 17)
ax[0, 0].plot(first_channel_noisy)
ax[0, 0].set_xlabel(xlabel='Sample', size=12)
ax[0, 0].set_ylabel(ylabel='Amplitude', size=12)
ax[0, 0].set_title("First Channel", fontsize=15)
ax[0, 0].tick_params(axis='both', which='both', labelsize=12)

ax[1, 0].specgram(first_channel_noisy, Fs=sample_rate)
ax[1, 0].set_xlabel(xlabel='Time [sec]', size=12)
ax[1, 0].set_ylabel(ylabel='Frequency [Hz]', size=12)
ax[1, 0].tick_params(axis='both', which='both', labelsize=12)

ax[0, 1].plot(second_channel_noisy)
ax[0, 1].set_xlabel(xlabel='Sample', size=12)
ax[0, 1].set_ylabel(ylabel='Amplitude', size=12)
ax[0, 1].set_title("Second Channel", fontsize=15)
ax[0, 1].tick_params(axis='both', which='both', labelsize=12)

ax[1, 1].specgram(second_channel_noisy, Fs=sample_rate)
ax[1, 1].set_xlabel(xlabel='Time [sec]', size=12)
ax[1, 1].set_ylabel(ylabel='Frequency [Hz]', size=12)
ax[1, 1].tick_params(axis='both', which='both', labelsize=12)
st.pyplot()

################################################################