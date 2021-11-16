# Stereo Matching

Naive stereo matching with sliding window and stereo matching with dynamic programming
for standard stereo image pairs.

## Project structure

* data_modified - Contains the input images from https://vision.middlebury.edu/stereo/data/
* disparity_maps - Computed disparity maps with naive stereo algorithm.
* disparity_maps_dp - Computed disparity maps with dynamic programming.
*points3d - Computed (x, y, z, r, g, b) coordinates for stereo reconstruction.
* results - Csv files contains the results of the experiments.
* metrics.py - Cost functions to measure image similarity.

The other files are self-explanatory. Jupyter notebooks are for demonstration
with examples.

## License
[MIT](https://choosealicense.com/licenses/mit/)