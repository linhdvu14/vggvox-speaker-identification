import numpy as np
import librosa
from scipy.signal import lfilter, butter

import sigproc
import constants as c



def read_audio(filename, sample_rate):
	audio, sr = librosa.load(filename, sr=sample_rate, mono=True)
	audio = audio.flatten()
	return audio


def normalize_frames(m,epsilon=1e-12):
	return np.array([(v - np.mean(v)) / max(np.std(v),epsilon) for v in m])


# https://github.com/christianvazquez7/ivector/blob/master/MSRIT/rm_dc_n_dither.m
def rm_dc_n_dither(sin, sample_rate):
	if sample_rate == 16e3:
		alpha = 0.99
	elif sample_rate == 8e3:
		alpha = 0.999
	else:
		print("Sample rate must be 16kHz or 8kHz only")
		exit(1)
	sin = lfilter([1,-1], [1,-alpha], sin)
	dither = np.random.random_sample(len(sin)) + np.random.random_sample(len(sin)) - 1
	spow = np.std(dither)
	sout = sin + 1e-6 * spow * dither
	return sout


# https://github.com/amaurycrickx/recognito/blob/master/recognito/src/main/java/com/bitsinharmony/recognito/enhancements/Normalizer.java
def normalize(signal):
    m = max(abs(signal))
    # if m > 1.:
    #     print("Expected value for audio signal are in the range -1.0 <= v <= 1.0: got [{}]".format(m))
    #     exit(1)
    # if m < 5*np.nextafter(0.,float('inf')):
    #     return signal
    return signal/m


# https://stackoverflow.com/questions/12093594/how-to-implement-band-pass-butterworth-filter-with-scipy-signal-butter
def butter_bandpass_filter(data, lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low, high = lowcut / nyq, highcut / nyq
    b, a = butter(order, [low, high], btype='bandpass')
    y = lfilter(b, a, data)
    return y


def read_and_process_audio(filename, buckets):
	signal = read_audio(filename,c.SAMPLE_RATE)

	# # Filter out non-speech frequencies
	# lowcut, highcut = c.FILTER_RANGE
	# signal = butter_bandpass_filter(signal, lowcut, highcut, c.SAMPLE_RATE, 1)

	# # Normalize signal
	# signal = normalize(signal)

	signal *= 2**15

	# Process signal to get FFT spectrum
	signal = rm_dc_n_dither(signal, c.SAMPLE_RATE)
	signal = sigproc.preemphasis(signal, coeff=c.PREEMPHASIS_ALPHA)
	frames = sigproc.framesig(signal, frame_len=c.FRAME_LEN*c.SAMPLE_RATE, frame_step=c.FRAME_STEP*c.SAMPLE_RATE, winfunc=np.hamming)
	fft = abs(np.fft.fft(frames,n=c.NUM_FFT))
	fft_norm = normalize_frames(fft.T)

	# Truncate to middle MAX_SEC seconds
	rsize = max(k for k in buckets if k <= fft_norm.shape[1])
	rstart = int((fft_norm.shape[1]-rsize)/2)
	out = fft_norm[:,rstart:rstart+rsize]

	return out



def test():
	filename = "test/4s_Linh0A_0"
	signal = read_audio(filename + ".wav",c.SAMPLE_RATE)
	lowcut, highcut = c.FILTER_RANGE
	signal = butter_bandpass_filter(signal, lowcut, highcut, c.SAMPLE_RATE, 1)
	librosa.output.write_wav("{}_{}-{}.wav".format(filename,int(round(lowcut)),int(round(highcut))), signal, int(c.SAMPLE_RATE), norm=False)


if __name__ == '__main__':
	test()
