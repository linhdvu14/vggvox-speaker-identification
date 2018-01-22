# Remove all .wav files under a certain duration

MIN_LENGTH=7  # secs
DIR='LibriSpeech/dev-clean_train_7s'

for f in $(find $DIR -name '*.wav'); do 
	length=`soxi -D $f`
	fi
done


# for f in $(find $DIR -name '*.txt'); do 
# 	rm $f
# done
