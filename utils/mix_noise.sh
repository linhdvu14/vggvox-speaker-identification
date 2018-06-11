NOISE="18_Fan_USB"

NOISE_DIR="data/wav/noises/"
NOISE_FILE="${NOISE_DIR}${NOISE}.wav"
SPEECH_DIR="data/wav/prdcv_template/"


# Mix template (clean) audios with specified noise
OUT_DIR="data/wav/prdcv_mixed/prdcv_${NOISE}/"
rm -rf $OUT_DIR
mkdir $OUT_DIR


for f in $(find $SPEECH_DIR -name '*.wav'); do 
	fn=`echo $f | rev | cut -d \/ -f 1 | rev`  # filename w/o path
	OUT_FILE="${OUT_DIR}/$fn"

	# convert to 16kHz mono
	sox $f -r 16k -c 1 temp_speech.wav
	sox $NOISE_FILE -r 16k -c 1 temp_noise.wav

	# concat noise sample to >= length of speech sample
	len_noise=`soxi -D temp_noise.wav | cut -d \. -f 1`  # drop decimal
	len_speech=`soxi -D temp_speech.wav | cut -d \. -f 1`
	t=$(( len_speech / len_noise ))

	cp temp_noise.wav temp_noise2.wav
	for ((i=0; i < t; i++)); do
		cp temp_noise2.wav temp_noise2_bk.wav
		sox temp_noise2_bk.wav temp_noise.wav temp_noise2.wav
	done

	# merge and trim
	# sox -m temp_speech.wav temp_noise2.wav temp_mixed.wav trim 0 `soxi -D temp_speech.wav`
	sox -m -v 1 temp_speech.wav -v 3 temp_noise2.wav temp_mixed.wav trim 0 `soxi -D temp_speech.wav`

	# output
	sox temp_mixed.wav -r 16k -c 1 $OUT_FILE

	# clean up
	rm -f temp*

done



# Split mixed audio into short segments and rename
SEGMENT_LENGTH=4  # secs

for f in $(find $OUT_DIR -name '*.wav'); do 
	fn=`echo $f | cut -d \. -f 1`  # drop extension
	sox $f $fn-.wav trim 0 $SEGMENT_LENGTH : newfile : restart  # split audio
done


for f in $(find $OUT_DIR -name '*.wav'); do 
	length=`soxi -D $f`
	if [ $(echo " $SEGMENT_LENGTH == $length" | bc) -eq 1 ]; then  # rename split segments
		mv $f ${f/120s_/${SEGMENT_LENGTH}s_}
	elif (( $(echo "$SEGMENT_LENGTH > $length" |bc -l) )); then
		rm $f
	fi
done
