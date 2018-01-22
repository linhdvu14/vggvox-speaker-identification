IN_DIR="data/wav/prdcv_mixed/prdcv_18_Fan_USB"
OUT_DIR="${IN_DIR}_unmixed"

rm -rf $OUT_DIR
mkdir $OUT_DIR


for f in $(find $IN_DIR -name '*.wav'); do 
	fn=`echo $f | rev | cut -d \/ -f 1 | rev`  # filename w/o path
	OUT_FILE="${OUT_DIR}/$fn"

	# to 48kHz mono
	sox $f -r 48k -c 1 temp.wav

	# wav to raw
	sox -b 16 -c 1 -r 48k temp.wav temp.raw
	mv temp.raw temp.pcm

	# denoise
	bin/rnnoise_demo temp.pcm temp_clean.pcm

	# raw to wav
	sox -b 16 -c 1 -r 48k -t raw -e signed-integer temp_clean.pcm $OUT_FILE
done


# clean up
rm -f temp*
