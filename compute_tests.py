from model import vggvox_model
import librosa
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, euclidean, cosine

import pyaudio
import wave

from wav_reader import read_and_process_audio
import constants as c



def build_buckets(max_sec, step_sec):
	buckets = {}
	frames_per_sec = int(1/c.FRAME_STEP)
	end_frame = int(max_sec*frames_per_sec)
	step_frame = int(step_sec*frames_per_sec)
	for i in range(0,end_frame+1,step_frame):
		s = i
		s = np.floor((s-7+2)/2) + 1  # conv2
		s = np.floor((s-3)/2) + 1  # mpool1
		s = np.floor((s-5+2)/2) + 1  # conv2
		s = np.floor((s-3)/2) + 1  # mpool2
		s = np.floor((s-3+2)/1) + 1  # conv3
		s = np.floor((s-3+2)/1) + 1  # conv4
		s = np.floor((s-3+2)/1) + 1  # conv5
		s = np.floor((s-3)/2) + 1  # mpool5
		s = np.floor((s-1)/1) + 1  # fc6
		if s > 0:
			buckets[i] = int(s)
	return buckets



def forward_offline(model, file_dir, list_file, max_sec):
	buckets = build_buckets(max_sec,c.BUCKET_STEP_SEC)
	result = pd.read_csv(list_file, delimiter=",")
	print(result.head(10))
	result['file_path'] = file_dir + result['filename']
	result['features'] = result['file_path'].apply(lambda x: read_and_process_audio(x, buckets))
	result['embedding'] = result['features'].apply(lambda x: np.squeeze(model.predict(x.reshape(1,*x.shape,1))))
	return result[['filename','speaker','embedding']]


def forward_online(model, filename, max_sec):
	buckets = build_buckets(max_sec,c.BUCKET_STEP_SEC)
	signal = read_and_process_audio(filename, buckets)
	embedding = np.squeeze(model.predict(signal.reshape(1,*signal.shape,1)))
	return embedding



def batch_offline_test():
	print("Loading model for batch offline test from [{}]....".format(c.WEIGHTS_FILE))
	model = vggvox_model()
	model.load_weights(c.WEIGHTS_FILE)
	model.summary()

	print("Processing enroll samples in [{}]....".format(c.ENROLL_WAV_DIR))
	enroll_result = forward_offline(model, c.ENROLL_WAV_DIR, c.ENROLL_LIST_FILE, c.MAX_SEC_ENROLL)
	enroll_embs = np.array([emb.tolist() for emb in enroll_result['embedding']])
	speakers = enroll_result['speaker']

	print("Processing test samples in [{}]....".format(c.TEST_WAV_DIR))
	test_result = forward_offline(model, c.TEST_WAV_DIR, c.TEST_LIST_FILE, c.MAX_SEC_TEST)
	test_embs = np.array([emb.tolist() for emb in test_result['embedding']])

	print("Comparing test samples against enroll samples....")
	distances = pd.DataFrame(cdist(test_embs, enroll_embs, metric=c.COST_METRIC), columns=speakers)

	# get all speakers in top 10%
	num_speakers_top_1 = max(int(len(speakers) / 100),1)
	num_speakers_top_5 = max(int(len(speakers)*5 / 100),1)
	num_speakers_top_10 = max(int(len(speakers)*10 / 100),1)

	results = pd.DataFrame(distances.columns[distances.values.argsort(1)[:,:num_speakers_top_10]].values,index=distances.index)
	results = results.rename(columns = lambda x: 'result_{}'.format(x + 1))

	scores = pd.read_csv(c.TEST_LIST_FILE, delimiter=",",header=0,names=['test_file','test_speaker'])
	scores = pd.concat([scores,distances,results],axis=1)
	scores['correct'] = (scores['result_1'] == scores['test_speaker'])*1. # bool to int
	
	correct = scores['correct']
	for i in range(1,num_speakers_top_10+1,1):
		correct = np.logical_or(correct, scores['result_{}'.format(i)]==scores['test_speaker'])*1.
		if i == num_speakers_top_1:
			scores['correct_top_1%'] = correct
		elif i == num_speakers_top_5:
			scores['correct_top_5%'] = correct
		elif i == num_speakers_top_10:
			scores['correct_top_10%'] = correct

	# output
	print("Writing outputs to [{}]....".format(c.OFFLINE_RESULT_FILE))
	with open(c.OFFLINE_RESULT_FILE, c.OFFLINE_RESULT_WRITE_OPTION) as f:
		scores.to_csv(f, index=False)



def offline_test():
	print("Loading model for offline test from [{}]....".format(c.WEIGHTS_FILE))
	model = vggvox_model()
	model.load_weights(c.WEIGHTS_FILE)
	model.summary()

	print("Processing enroll samples in [{}]....".format(c.ENROLL_WAV_DIR))
	enroll_result = forward_offline(model, c.ENROLL_WAV_DIR, c.ENROLL_LIST_FILE, c.MAX_SEC_ENROLL)
	enroll_embs = np.array([emb.tolist() for emb in enroll_result['embedding']])
	speakers = enroll_result['speaker']

	print("Processing test samples in [{}]....".format(c.TEST_WAV_DIR))
	test_result = forward_offline(model, c.TEST_WAV_DIR, c.TEST_LIST_FILE, c.MAX_SEC_TEST)
	test_embs = np.array([emb.tolist() for emb in test_result['embedding']])

	print("Comparing test samples against enroll samples....")
	distances = pd.DataFrame(cdist(test_embs, enroll_embs, metric=c.COST_METRIC), columns=speakers)

	scores = pd.read_csv(c.TEST_LIST_FILE, delimiter=",",header=0,names=['test_file','test_speaker'])
	scores = pd.concat([scores,distances],axis=1)
	scores['result'] = scores[speakers].idxmin(axis=1)
	scores['correct'] = (scores['result'] == scores['test_speaker'])*1. # bool to int

	print("Writing outputs to [{}]....".format(c.OFFLINE_RESULT_FILE))
	with open(c.OFFLINE_RESULT_FILE, c.OFFLINE_RESULT_WRITE_OPTION) as f:
		scores.to_csv(f, index=False)


def online_test():
	print("Loading model for online test from [{}]....".format(c.WEIGHTS_FILE))
	model = vggvox_model()
	model.load_weights(c.WEIGHTS_FILE)
	model.summary()

	print("Processing enroll samples in [{}]....".format(c.ENROLL_WAV_DIR))
	enroll_result = forward_offline(model, c.ENROLL_WAV_DIR, c.ENROLL_LIST_FILE, c.MAX_SEC_ENROLL)
	enroll_embs = np.array([emb.tolist() for emb in enroll_result['embedding']])
	speakers = enroll_result['speaker']

	with open(c.ONLINE_RESULT_FILE, c.OFFLINE_RESULT_WRITE_OPTION) as f:
		f.write("condition,test_speaker,{},result,correct\n".format(','.join(speakers)))
	CSV_PREFIX = c.ONLINE_CONDITION + "," + c.ONLINE_SPEAKER + ","

	p = pyaudio.PyAudio()
	while True:
		# Record
		stream = p.open(format=c.FORMAT,channels=c.NUM_CHANNEL,rate=c.SAMPLE_RATE,input=True,frames_per_buffer=c.CHUNK)
		print("\nStart speaking")
		frames = []
		for i in range(0, int(c.SAMPLE_RATE / c.CHUNK*c.ONLINE_RECORD_SEC)):
		    data = stream.read(c.CHUNK)
		    frames.append(data)
		print("Done recording")

		stream.stop_stream()
		stream.close()

		# Save audio
		wf = wave.open(c.ONLINE_WAV_FILE, 'wb')
		wf.setnchannels(c.NUM_CHANNEL)
		wf.setsampwidth(p.get_sample_size(c.FORMAT))
		wf.setframerate(c.SAMPLE_RATE)
		wf.writeframes(b''.join(frames))
		wf.close()

		# Test against enrolled samples
		print("Comparing test sample against enroll samples....")
		emb = forward_online(model, c.ONLINE_WAV_FILE, c.MAX_SEC_TEST)
		buff = CSV_PREFIX
		min_dist, min_spk = 1., None
		for i,spk in enumerate(enroll_result['speaker']):
			if c.COST_METRIC == "euclidean":
				dist = euclidean(emb, enroll_result['embedding'][i])
			elif c.COST_METRIC == "cosine":
				dist = cosine(emb, enroll_result['embedding'][i])
			else:
				print("Invalid cost metric [{}]".format(c.COST_METRIC))
			if dist < min_dist:
				min_dist, min_spk = dist, spk
			buff += str(dist) + ","
			print("Distance with speaker [{}]:\t{}".format(spk, dist))
		print("-----> {}".format(min_spk))

		correct = int(min_spk == c.ONLINE_SPEAKER)
		buff += min_spk + "," + str(correct)
		with open(c.ONLINE_RESULT_FILE, 'a') as f:
			f.write(buff + "\n")

	p.terminate()



def test():
	TEST_WAV1 = "data/wav/prdcv-script/4s_Linh0A_0.wav"
	TEST_WAV2 = "data/wav/prdcv-script/10s_Linh0A_0.wav"
	model = vggvox_model()
	model.load_weights("data/model_weights/model_0.h5")
	buckets = build_buckets(c.MAX_SEC_TEST,c.BUCKET_STEP_SEC)
	spec1 = read_and_process_audio(TEST_WAV1,buckets)
	emb1 = model.predict(spec1.reshape(1,*spec1.shape,1))
	spec2 = read_and_process_audio(TEST_WAV2,buckets)
	emb2 = model.predict(spec2.reshape(1,*spec2.shape,1))
	dist = np.linalg.norm(emb1-emb2)
	print(dist)



if __name__ == '__main__':
	if c.TEST_MODE == "batch_offline":
		batch_offline_test()
	elif c.TEST_MODE == "offline":
		offline_test()
	elif c.TEST_MODE == "online":
		online_test()
