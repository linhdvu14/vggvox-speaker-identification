for f in *.wav; do
	mv $f ${f/120s_/4s_}
done