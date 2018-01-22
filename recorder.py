import pyaudio
import wave
import constants as c

def record():
	p = pyaudio.PyAudio()
	NUM_REPS = 1
	RECORD_SEC = 20
	FILENAME = "data/wav/Linh" 

	# for rep in range(NUM_REPS):
	# Record
	stream = p.open(format=c.FORMAT,channels=c.NUM_CHANNEL,rate=c.SAMPLE_RATE,input=True,frames_per_buffer=c.CHUNK)
	print("\nStart speaking")
	frames = []
	for i in range(0, int(c.SAMPLE_RATE / c.CHUNK*RECORD_SEC)):
	    data = stream.read(c.CHUNK)
	    frames.append(data)
	print("Done recording")

	stream.stop_stream()
	stream.close()

	# Save audio
	wf = wave.open("{}.wav".format(FILENAME), 'wb')
	# wf = wave.open("{}_{:0>3d}.wav".format(FILENAME,rep), 'wb')
	wf.setnchannels(c.NUM_CHANNEL)
	wf.setsampwidth(p.get_sample_size(c.FORMAT))
	wf.setframerate(c.SAMPLE_RATE)
	wf.writeframes(b''.join(frames))
	wf.close()

	p.terminate()


if __name__ == '__main__':
	record()
