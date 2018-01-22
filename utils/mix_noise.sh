NOISE_DIR="data/wav/noises/"
NOISE="bus-interior-1"
NOISE_FILE="${NOISE_DIR}${NOISE}.wav"

SPEECH_DIR="data/wav/prdcv/"

OUT_DIR=${SPEECH_DIR/prdcv/prdcv_${NOISE}}
mkdir $OUT_DIR


for f in $(find $SPEECH_DIR -name '*.wav'); do 
	OUT_FILE=${f/prdcv/prdcv_${NOISE}}

	# convert to 48kHz mono
	sox $f -r 48k -c 1 temp_speech.wav
	sox $NOISE_FILE -r 48k -c 1 temp_noise.wav

	# concat noise sample to >= speech sample
	len_noise=`soxi -D temp_noise.wav | cut -d \. -f 1`  # drop decimal
	len_speech=`soxi -D temp_speech.wav | cut -d \. -f 1`
	t=$((len_speech / len_noise ))

	cp temp_noise.wav temp_noise2.wav
	for ((i=0; i < t; i++)); do
		cp temp_noise2.wav temp_noise2_bk.wav
		sox temp_noise2_bk.wav temp_noise.wav temp_noise2.wav
	done

	# merge and trim
	sox -m temp_speech.wav temp_noise2.wav temp_mixed.wav trim 0 `soxi -D temp_speech.wav`
	# sox -m -v 1 temp_speech.wav -v 0.3 temp_noise2.wav temp_mixed.wav trim 0 `soxi -D temp_speech.wav`

	# down to 16kHz
	sox temp_mixed.wav -r 16k -c 1 $OUT_FILE

	# clean up
	rm -f temp*

done
