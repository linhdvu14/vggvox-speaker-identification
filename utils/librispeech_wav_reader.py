import os
from glob2 import glob

import numpy as np
import pandas as pd

np.set_printoptions(threshold=np.nan)
pd.set_option('display.height', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('max_colwidth', 100)


DATA_DIR = 'dev-other'
BASE_OUTFILE = "batch"


def find_files(directory, pattern='**/*.wav'):
	"""Recursively finds all files matching the pattern."""
	return glob(os.path.join(directory, pattern), recursive=True)


def read_librispeech_structure(directory):
	libri = pd.DataFrame()
	libri['filename'] = find_files(directory)
	libri['filename'] = libri['filename'].apply(lambda x: x.replace('\\', '/')) # normalize windows paths
	libri['speaker'] = libri['filename'].apply(lambda x: int(x.split('/')[-3]))
	num_speakers = len(libri['speaker'].unique())
	print('Found {} files with {} different speakers.'.format(str(len(libri)).zfill(7), str(num_speakers).zfill(5)))
	print(libri.head(10))
	libri = libri[['filename','speaker']].sort_values(['speaker', 'filename'])
	return libri


def main():
	libri = read_librispeech_structure(DATA_DIR)
	libri_enroll = libri.groupby("speaker").head(1)  # enroll first file
	libri_test = libri.groupby('speaker').apply(lambda group: group.iloc[1:,:])
	with open(BASE_OUTFILE + ".csv", "w") as f:
		libri.to_csv(f, index=False)
	with open(BASE_OUTFILE + "_enroll.csv", "w") as f:
		libri_enroll.to_csv(f, index=False)
	with open(BASE_OUTFILE + "_test.csv", "w") as f:
		libri_test.to_csv(f, index=False)


if __name__ == '__main__':
	main()
