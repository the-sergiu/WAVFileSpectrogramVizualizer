import scipy
from scipy.io import wavfile
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy.signal import butter, lfilter, freqz, filtfilt

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000

import numpy as np
import streamlit as st
import os
import shutil

st.set_option('deprecation.showPyplotGlobalUse', False)

# Stergem tot din folder-ul de cache
folder = 'cache/'
for filename in os.listdir(folder):
    file_path = os.path.join(folder, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))

### Top area ###

st.title("Spectrogram Visualizer for Wave Audio Files")
st.write("Brought to you by Sergiu Craioveanu and Paraschiva Mihai")

st.markdown("## Try out our example samples or upload your own!")
option = st.selectbox("Pick your sample!", options=["Take your pick", "Piano Music", "Coffin Dance", "Upload your own!"])

if option == "Take your pick":
    st.stop()

elif option == "Piano Music":
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

elif option == "Upload your own!":
    st.markdown("## Number of samples")
    no_samples = st.selectbox("Pick your sample!",
                          options=["2", "1"])

    st.markdown("# Select your first sample")
    uploaded_file1 = st.file_uploader("Pick a wave file!", type='wav', key="sample1")

    if uploaded_file1 is None:
        st.info("Please upload a wave file.")
        st.stop()

    st.markdown("# Select your second sample")
    uploaded_file2 = st.file_uploader("Pick a wave file!", type='wav', key="sample2")

    if uploaded_file2 is None:
        st.info("Please upload a wave file.")
        st.stop()

    # We generate byte data from the uploaded sample 1 to store locally
    bytes_data = uploaded_file1.getvalue()

    if os.path.exists('cache/sample1.wav'):
        pass
    else:
        with open('cache/sample1.wav', mode='bx') as f:
            f.write(bytes_data)

    # We generate byte data from the uploaded sample 2 to store locally
    bytes_data = uploaded_file2.getvalue()

    if os.path.exists('cache/sample2.wav'):
        pass
    else:
        with open('cache/sample2.wav', mode='bx') as f:
            f.write(bytes_data)

    # Read the clean .wav file
    sample_rate, data = wavfile.read('cache/sample1.wav')

    # Read the noisy wav file
    sample_rate_noisy, data_noisy = wavfile.read('cache/sample2.wav')

    st.markdown('## Check out the first sample!')
    audio_file = open("cache/sample1.wav", 'rb')
    audio_bytes = audio_file.read()
    st.audio(audio_bytes, format='audio/ogg')

    st.markdown('## Check out the second sample!')
    audio_file = open("cache/sample1.wav", 'rb')
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

# xticks for first sample and second sample
xticks_1 = [i * sample_rate for i in range((len(data) // sample_rate) + 1)]
xticks_2 = [i * sample_rate_noisy for i in range((len(data_noisy) // sample_rate_noisy) + 1)]

# yticks for spectrograms
helper = [0, 2500, 5000, 7500, 10000, 12500, 15000, 17500, 20000]
spec_yticks = [6.28 * i for i in helper]

## Plots First Sample ##
#####################################################################
st.markdown('# First Sample Analysis for Both Channels')


st.markdown('## Amplitude-Time ')
at1 = st.checkbox("Show", key=1)

fig, ax = plt.subplots(1, 2, figsize=(30, 15))
fig.tight_layout(pad=10.0)
ax[0].plot(20*np.log10(np.abs(first_channel)))
ax[0].set_xticks(xticks_1)
ax[0].set_xticklabels(np.arange(0, len(xticks_1), 1))
ax[0].set_xlabel(xlabel='Time [sec]', size=25)
ax[0].set_ylabel(ylabel='Amplitude [dB]', size=25)
ax[0].set_ylim(60, 87)
ax[0].set_title("First Channel", fontsize=30)
ax[0].tick_params(axis='both', which='both', labelsize=15)

ax[1].plot(20*np.log10(np.abs(second_channel)))
ax[1].set_xticks(xticks_1)
ax[1].set_xticklabels(np.arange(0, len(xticks_1), 1))
ax[1].set_xlabel(xlabel='Time [sec]', size=25)
ax[1].set_ylabel(ylabel='Amplitude [dB]', size=25)
ax[1].set_ylim(60, 87)
ax[1].set_title("Second Channel", fontsize=30)
ax[1].tick_params(axis='both', which='both', labelsize=15)
if at1:
    st.pyplot(fig)

## -------------------------------------------------------------
st.markdown('## Spectrogram')
spec1 = st.checkbox("Show", key=2)

fig2, ax = plt.subplots(1, 2, figsize=(30, 15))
fig.tight_layout(pad=10.0)
ax[0].specgram(first_channel, Fs=sample_rate)
ax[0].set_xlabel(xlabel='Time [sec]', size=25)
ax[0].set_ylabel(ylabel='Frequency Amplitude [rad/s]', size=25)
ax[0].set_yticks(helper)
ax[0].set_yticklabels(spec_yticks)
ax[0].set_title("First Channel", fontsize=30)
ax[0].tick_params(axis='both', which='both', labelsize=18)

ax[1].specgram(second_channel, Fs=sample_rate)
ax[1].set_xlabel(xlabel='Time [sec]', size=25)
ax[1].set_ylabel(ylabel='Frequency Amplitude [rad/s]', size=25)
ax[1].set_yticks(helper)
ax[1].set_yticklabels(spec_yticks)
ax[1].set_title("Second Channel", fontsize=30)
ax[1].tick_params(axis='both', which='both', labelsize=18)
if spec1:
    st.pyplot(fig2)

## ---------------------------------------------------------------

st.markdown('## Power Spectral Density')
psd1 = st.checkbox("Show", key=3)

fig3, ax = plt.subplots(1, 2, figsize=(30, 15))
fig.tight_layout(pad=10.0)
# freqs, psd = signal.welch(first_channel)
# ax[0].semilogx(6.28*freqs, psd)
ax[0].psd(first_channel)
ax[0].set_xlabel('Frequency [rad/s]', size=25)
ax[0].set_ylabel('Power Spectral Denisty [db]', size=25)
ax[0].set_title("First Channel", fontsize=30)
ax[0].tick_params(axis='both', which='both', labelsize=25)

# freqs, psd = signal.welch(second_channel)
# ax[1].semilogx(6.28*freqs, psd)
ax[1].psd(second_channel)
ax[1].set_xlabel('Frequency [rad/s]', size=25)
ax[1].set_ylabel('Power Spectral Denisty [db]', size=25)
ax[1].set_title("Second Channel", fontsize=30)
ax[1].tick_params(axis='both', which='both', labelsize=25)
if psd1:
    st.pyplot(fig3)

## ---------------------------------------------------------------

st.markdown('## Fast Fourier Transform')
fft1 = st.checkbox("Show", key=4)

fast1 = fft(first_channel)
fast2 = fft(second_channel)
xf1 = fftfreq(len(first_channel), 44100)
xf2 = fftfreq(len(second_channel), 44100)

fig4, ax = plt.subplots(1, 2, figsize=(30, 15))
fig.tight_layout(pad=10.0)
ax[0].set_xlabel('Frequency [rad/s]', size=25)
ax[0].set_ylabel('Amplitude', size=25)
ax[0].set_title("First Channel", fontsize=30)
ax[0].tick_params(axis='both', which='both', labelsize=25)
ax[0].plot(6.278*xf1, np.abs(fast1))

ax[1].set_xlabel('Frequency [rad/s]', size=25)
ax[1].set_ylabel('Amplitude', size=25)
ax[1].set_title("Second Channel", fontsize=30)
ax[1].tick_params(axis='both', which='both', labelsize=25)
ax[1].plot(6.278 * xf2, np.abs(fast2))
if fft1:
    st.pyplot(fig4)

## Plot Second Sample ##
################################################################
st.markdown('# Second Sample Analysis for Both Channels')

st.markdown('## Amplitude-Time ')
at2 = st.checkbox("Show", key=5)

fig5, ax = plt.subplots(1, 2, figsize=(30, 15))
fig.tight_layout(pad=10.0)
ax[0].plot(20*np.log10(np.abs(first_channel_noisy)))
ax[0].set_xticks(xticks_2)
ax[0].set_xticklabels(np.arange(0, len(xticks_2), 1))
ax[0].set_xlabel(xlabel='Time [sec]', size=25)
ax[0].set_ylabel(ylabel='Amplitude [dB]', size=25)
ax[0].set_ylim(60, 87)
ax[0].set_title("First Channel", fontsize=30)
ax[0].tick_params(axis='both', which='both', labelsize=15)

ax[1].plot(20*np.log10(np.abs(second_channel_noisy)))
ax[1].set_xticks(xticks_2)
ax[1].set_xticklabels(np.arange(0, len(xticks_2), 1))
ax[1].set_xlabel(xlabel='Time [sec]', size=25)
ax[1].set_ylabel(ylabel='Amplitude [dB]', size=25)
ax[1].set_ylim(60, 87)
ax[1].set_title("Second Channel", fontsize=30)
ax[1].tick_params(axis='both', which='both', labelsize=15)
if at2:
    st.pyplot(fig5)

## -------------------------------------------------------------
st.markdown('### Spectrogram')
spec2 = st.checkbox("Show", key=6)

fig6, ax = plt.subplots(1, 2, figsize=(30, 15))
fig.tight_layout(pad=10.0)
ax[0].specgram(first_channel_noisy, Fs=sample_rate_noisy)
ax[0].set_xlabel(xlabel='Time [sec]', size=25)
ax[0].set_ylabel(ylabel='Frequency Amplitude [rad/s]', size=25)
ax[0].set_yticks(helper)
ax[0].set_yticklabels(spec_yticks)
ax[0].set_title("First Channel", fontsize=30)
ax[0].tick_params(axis='both', which='both', labelsize=18)

ax[1].specgram(second_channel_noisy, Fs=sample_rate_noisy)
ax[1].set_xlabel(xlabel='Time [sec]', size=25)
ax[1].set_ylabel(ylabel='Frequency Amplitude [rad/s]', size=25)
ax[1].set_yticks(helper)
ax[1].set_yticklabels(spec_yticks)
ax[1].set_title("Second Channel", fontsize=30)
ax[1].tick_params(axis='both', which='both', labelsize=18)
if spec2:
    st.pyplot(fig6)

## -------------------------------------------------------------
st.markdown('### Power Spectral Density')
psd2 = st.checkbox("Show", key=7)

fig7, ax = plt.subplots(1, 2, figsize=(30, 15))
fig.tight_layout(pad=10.0)
# freqs, psd = signal.welch(first_channel_noisy)
# ax[0].semilogx(6.28*freqs, psd)
ax[0].psd(first_channel_noisy)
ax[0].set_xlabel('Frequency [rad/s]', size=25)
ax[0].set_ylabel('Power Spectral Denisty [db]', size=25)
ax[0].set_title("First Channel", fontsize=30)
ax[0].tick_params(axis='both', which='both', labelsize=25)

# freqs, psd = signal.welch(second_channel_noisy)
# ax[1].semilogx(6.28*freqs, psd)
ax[1].psd(second_channel_noisy)
ax[1].set_xlabel('Frequency [rad/s]', size=25)
ax[1].set_ylabel('Power Spectral Denisty [db]', size=25)
ax[1].set_title("Second Channel", fontsize=30)
ax[1].tick_params(axis='both', which='both', labelsize=25)
if psd2:
    st.pyplot(fig7)

## ---------------------------------------------------------------
st.markdown('### Fast Fourier Transform')
fft2 = st.checkbox("Show", key=8)

fast1 = fft(first_channel_noisy)
fast2 = fft(second_channel_noisy)
xf1n = fftfreq(len(first_channel_noisy), 44100)
xf2n = fftfreq(len(second_channel_noisy), 44100)

fig8, ax = plt.subplots(1, 2, figsize=(30, 15))
fig.tight_layout(pad=10.0)
ax[0].set_xlabel('Frequency [rad/s]', size=25)
ax[0].set_ylabel('Amplitude', size=25)
ax[0].set_title("First Channel", fontsize=30)
ax[0].tick_params(axis='both', which='both', labelsize=25)
ax[0].plot(6.278*xf1n, np.abs(fast1))

ax[1].set_xlabel('Frequency [rad/s]', size=25)
ax[1].set_ylabel('Amplitude', size=25)
ax[1].set_title("Second Channel", fontsize=30)
ax[1].tick_params(axis='both', which='both', labelsize=25)
ax[1].plot(6.278 * xf2n, np.abs(fast2))
if fft2:
    st.pyplot(fig8)


# Filter Functions
################################################################

## LOW PASS
@st.cache
def intit_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

@st.cache
def lowpass_filter(data, cutoff, fs, order=5):
    b, a = intit_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

## HIGH PASS
@st.cache
def init_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

@st.cache
def highpass_filter(data, cutoff, fs, order=5):
    b, a = init_highpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

@st.cache
## BANDPASS
def init_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

@st.cache
def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = init_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

# Plot filters for first sample
################################################################

st.markdown('# Applying Filters to First Sample')

# Low pass filter
cutoff_low = 1000
first_channel_filtered_low = lowpass_filter(first_channel, cutoff_low, sample_rate, order=5)
second_channel_filtered_low = lowpass_filter(second_channel, cutoff_low, sample_rate, order=5)

st.markdown('## Low Pass Filter')

st.markdown('### Spectrogram')
lp1 = st.checkbox("Show", key=9)

fig9, ax = plt.subplots(1, 2, figsize=(30, 15))
fig.tight_layout(pad=10.0)
ax[0].specgram(first_channel_filtered_low, Fs=sample_rate)
ax[0].set_xlabel(xlabel='Time [sec]', size=25)
ax[0].set_ylabel(ylabel='Frequency Amplitude [rad/s]', size=25)
ax[0].set_yticks(helper)
ax[0].set_yticklabels(spec_yticks)
ax[0].set_title("First Channel", fontsize=30)
ax[0].tick_params(axis='both', which='both', labelsize=18)

ax[1].specgram(second_channel_filtered_low, Fs=sample_rate)
ax[1].set_xlabel(xlabel='Time [sec]', size=25)
ax[1].set_ylabel(ylabel='Frequency Amplitude [rad/s]', size=25)
ax[1].set_yticks(helper)
ax[1].set_yticklabels(spec_yticks)
ax[1].set_title("Second Channel", fontsize=30)
ax[1].tick_params(axis='both', which='both', labelsize=18)
if lp1:
    st.pyplot(fig9)
## -------------------------------------
st.markdown('### Amplitude(mirrored) - Samples')
as3 = st.checkbox("Show", key=10)

fig10, ax = plt.subplots(1, 2, figsize=(30, 15))
fig.tight_layout(pad=10.0)
ax[0].plot(first_channel_filtered_low)
ax[0].set_xlabel(xlabel='Samples', size=25)
ax[0].set_ylabel(ylabel='Amplitude', size=25)
ax[0].set_title("First Channel", fontsize=30)
ax[0].tick_params(axis='both', which='both', labelsize=15)

ax[1].plot(second_channel_filtered_low)
ax[1].set_xlabel(xlabel='Samples', size=25)
ax[1].set_ylabel(ylabel='Amplitude', size=25)
ax[1].set_title("Second Channel", fontsize=30)
ax[1].tick_params(axis='both', which='both', labelsize=15)
if as3:
    st.pyplot(fig10)

# High pass filter
#####################################
cutoff_high = 20000
first_channel_filtered_high = highpass_filter(first_channel, cutoff_high, sample_rate)
second_channel_filtered_high = highpass_filter(second_channel, cutoff_high, sample_rate)

st.markdown('## High Pass Filter')

st.markdown('### Spectrogram')
spec3 = st.checkbox("Show", key=11)

fig11, ax = plt.subplots(1, 2, figsize=(30, 15))
fig.tight_layout(pad=10.0)
ax[0].specgram(first_channel_filtered_high, Fs=sample_rate)
ax[0].set_xlabel(xlabel='Time [sec]', size=25)
ax[0].set_ylabel(ylabel='Frequency Amplitude [rad/s]', size=25)
ax[0].set_yticks(helper)
ax[0].set_yticklabels(spec_yticks)
ax[0].set_title("First Channel", fontsize=30)
ax[0].tick_params(axis='both', which='both', labelsize=18)

ax[1].specgram(second_channel_filtered_high, Fs=sample_rate)
ax[1].set_xlabel(xlabel='Time [sec]', size=25)
ax[1].set_ylabel(ylabel='Frequency Amplitude [rad/s]', size=25)
ax[1].set_yticks(helper)
ax[1].set_yticklabels(spec_yticks)
ax[1].set_title("Second Channel", fontsize=30)
ax[1].tick_params(axis='both', which='both', labelsize=18)
if spec3:
    st.pyplot(fig11)

## ---------------------------------
st.markdown('### Amplitude(mirrored) - Samples')
as4 = st.checkbox("Show", key=12)

fig12, ax = plt.subplots(1, 2, figsize=(30, 15))
fig.tight_layout(pad=10.0)
ax[0].plot(first_channel_filtered_high)
ax[0].set_xlabel(xlabel='Samples', size=25)
ax[0].set_ylabel(ylabel='Amplitude', size=25)
ax[0].set_title("First Channel", fontsize=30)
ax[0].tick_params(axis='both', which='both', labelsize=15)

ax[1].plot(second_channel_filtered_high)
ax[1].set_xlabel(xlabel='Samples', size=25)
ax[1].set_ylabel(ylabel='Amplitude', size=25)
ax[1].set_title("Second Channel", fontsize=30)
ax[1].tick_params(axis='both', which='both', labelsize=15)
if as4:
    st.pyplot(fig12)

# Band pass filter
#####################################

st.markdown('## Band Pass Filter')
band_low = 1000
band_high = 20000
first_channel_filtered_band = bandpass_filter(first_channel, band_low, band_high, sample_rate)
second_channel_filtered_band = bandpass_filter(second_channel, band_low, band_high, sample_rate)

st.markdown('### Spectrogram')
spec4 = st.checkbox("Show", key=13)

fig13, ax = plt.subplots(1, 2, figsize=(30, 15))
fig.tight_layout(pad=10.0)
ax[0].specgram(first_channel_filtered_band, Fs=sample_rate)
ax[0].set_xlabel(xlabel='Time [sec]', size=25)
ax[0].set_ylabel(ylabel='Frequency Amplitude [rad/s]', size=25)
ax[0].set_yticks(helper)
ax[0].set_yticklabels(spec_yticks)
ax[0].set_title("First Channel", fontsize=30)
ax[0].tick_params(axis='both', which='both', labelsize=18)


ax[1].specgram(second_channel_filtered_band, Fs=sample_rate)
ax[1].set_xlabel(xlabel='Time [sec]', size=25)
ax[1].set_ylabel(ylabel='Frequency Amplitude [rad/s]', size=25)
ax[1].set_yticks(helper)
ax[1].set_yticklabels(spec_yticks)
ax[1].set_title("Second Channel", fontsize=30)
ax[1].tick_params(axis='both', which='both', labelsize=18)
if spec4:
    st.pyplot(fig13)

## ---------------------------------
st.markdown('### Amplitude(mirrored) - Samples')
as5 = st.checkbox("Show", key=14)

fig14, ax = plt.subplots(1, 2, figsize=(30, 15))
fig.tight_layout(pad=10.0)
ax[0].plot(first_channel_filtered_band)
ax[0].set_xlabel(xlabel='Samples', size=25)
ax[0].set_ylabel(ylabel='Amplitude', size=25)
ax[0].set_title("First Channel", fontsize=30)
ax[0].tick_params(axis='both', which='both', labelsize=15)

ax[1].plot(second_channel_filtered_band)
ax[1].set_xlabel(xlabel='Samples', size=25)
ax[1].set_ylabel(ylabel='Amplitude', size=25)
ax[1].set_title("Second Channel", fontsize=30)
ax[1].tick_params(axis='both', which='both', labelsize=15)
if as5:
    st.pyplot(fig14)

# Plot filters for second sample
################################################################

st.markdown('# Applying Filters to Second Sample')

# Low pass filter
cutoff_low = 1000
first_channel_filtered_low = lowpass_filter(first_channel_noisy, cutoff_low, sample_rate, order=5)
second_channel_filtered_low = lowpass_filter(second_channel_noisy, cutoff_low, sample_rate, order=5)

st.markdown('## Low Pass Filter')

st.markdown('### Spectrogram')
lp2 = st.checkbox("Show", key=15)

fig15, ax = plt.subplots(1, 2, figsize=(30, 15))
fig.tight_layout(pad=10.0)
ax[0].specgram(first_channel_filtered_low, Fs=sample_rate_noisy)
ax[0].set_xlabel(xlabel='Time [sec]', size=25)
ax[0].set_ylabel(ylabel='Frequency Amplitude [rad/s]', size=25)
ax[0].set_yticks(helper)
ax[0].set_yticklabels(spec_yticks)
ax[0].set_title("First Channel", fontsize=30)
ax[0].tick_params(axis='both', which='both', labelsize=18)

ax[1].specgram(second_channel_filtered_low, Fs=sample_rate_noisy)
ax[1].set_xlabel(xlabel='Time [sec]', size=25)
ax[1].set_ylabel(ylabel='Frequency Amplitude [rad/s]', size=25)
ax[1].set_yticks(helper)
ax[1].set_yticklabels(spec_yticks)
ax[1].set_title("Second Channel", fontsize=30)
ax[1].tick_params(axis='both', which='both', labelsize=18)
if lp2:
    st.pyplot(fig15)

## ---------------------------------
st.markdown('### Amplitude(mirrored) - Samples')
as6 = st.checkbox("Show", key=16)

fig16, ax = plt.subplots(1, 2, figsize=(30, 15))
fig.tight_layout(pad=10.0)
ax[0].plot(first_channel_filtered_low)
ax[0].set_xlabel(xlabel='Samples', size=25)
ax[0].set_ylabel(ylabel='Amplitude', size=25)
ax[0].set_title("First Channel", fontsize=30)
ax[0].tick_params(axis='both', which='both', labelsize=15)

ax[1].plot(second_channel_filtered_low)
ax[1].set_xlabel(xlabel='Samples', size=25)
ax[1].set_ylabel(ylabel='Amplitude', size=25)
ax[1].set_title("Second Channel", fontsize=30)
ax[1].tick_params(axis='both', which='both', labelsize=15)
if as6:
    st.pyplot(fig16)

# High pass filter
#####################################
cutoff_high = 10000
first_channel_filtered_high = highpass_filter(first_channel_noisy, cutoff_high, sample_rate)
second_channel_filtered_high = highpass_filter(second_channel_noisy, cutoff_high, sample_rate)

st.markdown('## High Pass Filter')

st.markdown('### Spectrogram')
spec5 = st.checkbox("Show", key=17)

fig17, ax = plt.subplots(1, 2, figsize=(30, 15))
fig.tight_layout(pad=10.0)
ax[0].specgram(first_channel_filtered_high, Fs=sample_rate_noisy)
ax[0].set_xlabel(xlabel='Time [sec]', size=25)
ax[0].set_ylabel(ylabel='Frequency Amplitude [rad/s]', size=25)
ax[0].set_yticks(helper)
ax[0].set_yticklabels(spec_yticks)
ax[0].set_title("First Channel", fontsize=30)
ax[0].tick_params(axis='both', which='both', labelsize=18)

ax[1].specgram(second_channel_filtered_high, Fs=sample_rate_noisy)
ax[1].set_xlabel(xlabel='Time [sec]', size=25)
ax[1].set_ylabel(ylabel='Frequency Amplitude [rad/s]', size=25)
ax[1].set_yticks(helper)
ax[1].set_yticklabels(spec_yticks)
ax[1].set_title("Second Channel", fontsize=30)
ax[1].tick_params(axis='both', which='both', labelsize=18)
if spec5:
    st.pyplot(fig17)

## ---------------------------------
st.markdown('### Amplitude(mirrored) - Samples')
as7 = st.checkbox("Show", key=18)

fig18, ax = plt.subplots(1, 2, figsize=(30, 15))
fig.tight_layout(pad=10.0)
ax[0].plot(first_channel_filtered_high)
ax[0].set_xlabel(xlabel='Samples', size=25)
ax[0].set_ylabel(ylabel='Amplitude', size=25)
ax[0].set_title("First Channel", fontsize=30)
ax[0].tick_params(axis='both', which='both', labelsize=15)

ax[1].plot(second_channel_filtered_high)
ax[1].set_xlabel(xlabel='Samples', size=25)
ax[1].set_ylabel(ylabel='Amplitude', size=25)
ax[1].set_title("Second Channel", fontsize=30)
ax[1].tick_params(axis='both', which='both', labelsize=15)
if as7:
    st.pyplot(fig18)


# Band pass filter
#####################################

st.markdown('## Band Pass Filter')
band_low = 1000
band_high = 20000
first_channel_filtered_band = bandpass_filter(first_channel_noisy, band_low, band_high, sample_rate)
second_channel_filtered_band = bandpass_filter(second_channel_noisy, band_low, band_high, sample_rate)

st.markdown('### Spectrogram')
spec8 = st.checkbox("Show", key=19)

fig19, ax = plt.subplots(1, 2, figsize=(30, 15))
fig.tight_layout(pad=10.0)
ax[0].specgram(first_channel_filtered_band, Fs=sample_rate_noisy)
ax[0].set_xlabel(xlabel='Time [sec]', size=25)
ax[0].set_ylabel(ylabel='Frequency Amplitude [rad/s]', size=25)
ax[0].set_yticks(helper)
ax[0].set_yticklabels(spec_yticks)
ax[0].set_title("First Channel", fontsize=30)
ax[0].tick_params(axis='both', which='both', labelsize=18)

ax[1].specgram(second_channel_filtered_band, Fs=sample_rate_noisy)
ax[1].set_xlabel(xlabel='Time [sec]', size=25)
ax[1].set_ylabel(ylabel='Frequency Amplitude [rad/s]', size=25)
ax[1].set_yticks(helper)
ax[1].set_yticklabels(spec_yticks)
ax[1].set_title("Second Channel", fontsize=30)
ax[1].tick_params(axis='both', which='both', labelsize=18)
if spec8:
    st.pyplot(fig19)

## ---------------------------------
st.markdown('### Amplitude(mirrored) - Samples')
as6 = st.checkbox("Show", key=20)

fig20, ax = plt.subplots(1, 2, figsize=(30, 15))
fig.tight_layout(pad=10.0)
ax[0].plot(first_channel_filtered_band)
ax[0].set_xlabel(xlabel='Samples', size=25)
ax[0].set_ylabel(ylabel='Amplitude', size=25)
ax[0].set_title("First Channel", fontsize=30)
ax[0].tick_params(axis='both', which='both', labelsize=15)

ax[1].plot(second_channel_filtered_band)
ax[1].set_xlabel(xlabel='Samples', size=25)
ax[1].set_ylabel(ylabel='Amplitude', size=25)
ax[1].set_title("Second Channel", fontsize=30)
ax[1].tick_params(axis='both', which='both', labelsize=15)
if as6:
    st.pyplot(fig20)
