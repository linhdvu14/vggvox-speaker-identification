for f in *.csv; do
	mv $f ${f/results_/batch_results_}
done