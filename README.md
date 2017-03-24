# Face Recognizer (laggy)

Analyzes video frame by frame and counts number of peoples then writes to csv

The csv file will be **people.csv**

Because the display of the video and analyzing of the frame is done in the same(main) thread, there will be lag when showing the video.

## Algorithm Used
[**HOG** (Histogram of Oriented Gradients)](https://en.wikipedia.org/wiki/Histogram_of_oriented_gradients)

### Parameters ([Explanation of parameters](http://www.pyimagesearch.com/2015/11/16/hog-detectmultiscale-parameters-explained/))
- **winStride** =   (4, 4)
- **padding**   =   (8, 8) 
- **scale**     =   1.05

## Getting Started

Run the script by specifying the video
```
python[3] detect.py -v "path to the video"
```

### Prerequisites

The script is coded using ```python 3.6```, ```opencv 3```. And you will need opencv compiled with `ffmpeg` to read video

### Installing

Install opencv3 on anaconda
```
conda install -c menpo opencv3

```
Or to compile opencv3 with ffmpeg follow instructions here [https://github.com/menpo/conda-opencv3](https://github.com/menpo/conda-opencv3) or run these commands

```
$ conda install conda-build
$ git clone https://github.com/menpo/conda-opencv3
$ cd conda-opencv3
$ conda config --add channels menpo
$ conda build conda/
$ conda install /PATH/TO/OPENCV3/PACKAGE.tar.gz
```

Before compiling change the flag ```-DWITH_FFMPEG=0 ``` to ```1``` on file ```conda/build.sh```.

Then install other libraries using ```pip``` or ```pip3```
```
pip[3] install imutils
```

Another reference on installing opencv on Ubuntu 16.04: [http://www.pyimagesearch.com/2016/10/24/ubuntu-16-04-how-to-install-opencv/](http://www.pyimagesearch.com/2016/10/24/ubuntu-16-04-how-to-install-opencv/)


## Output
The output csv will be in this format
``` 
1.0,6
2.0,6
3.0,6
4.0,6
5.0,5
6.0,6
7.0,6
8.0,6
9.0,6

```
The Delimiter is `,`. The first column is the `frame number` and the second column is the `number of people` in that frame.
