# vggvox

* Python adaptation of VGGVox speaker identification model, based on Nagrani et al 2017, "[VoxCeleb: a large-scale speaker identification dataset](https://arxiv.org/pdf/1706.08612.pdf)"
* Evaluation code only, based on the author's [Matlab code](https://github.com/a-nagrani/VGGVox/)
and [pretrained model](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/).

## Instructions
* Install python3 and the required packages
* Modify `cfg/enroll_list.csv` and `cfg/test_list.csv` to point to your local enroll/test wav files
* To run evaluation: `python3 scoring.py`
* Results will be stored in `res/results.csv`. Each line has format: `[path to test wav], [correct speaker], [distance to enroll speaker 1],...[distance to enroll speaker N], [predicted speaker], [correct?]`