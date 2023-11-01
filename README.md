## Photometric Stereo

Implementation of calibrated photometric stereo with several datasets.
Images are taken from [here](http://courses.cs.washington.edu/courses/cse455/10wi/projects/project4/)

The light directions are estimated with the chrome dataset inside the image folder.

### Results

![Results](https://raw.githubusercontent.com/NewProggie/Photometric-Stereo/master/images/results.jpg)

### Build

```bash
~$ git clone https://github.com/SherAndrei/msu_photometric_stereo && cd msu_photometric_stereo/
~/msu_photometric_stereo$ mkdir build && cd build
~/msu_photometric_stereo/build$ cmake ..
~/msu_photometric_stereo/build$ cmake --build .
```

### Run

Seek help:
```bash
~/msu_photometric_stereo$ ./build/PhotometricStereo -h
USAGE:	./build/PhotometricStereo [OPTIONS]
OPTIONS:
  -h [ --help ]            produce this help message
  -c [ --calibration ] arg Path to a directory with calibration sphere images.
                           Names of image files should be specified, so for given `image_name` the next code is valid:
                           ```c
                            char name[256]; unsigned number; char extention[10];
                            assert(sscanf(image_name, "%s.%u.%s", name, &number, extention) == 3);
                           ```
                           Among the images there should be an image with name, which contains `.mask.` instead of a number of the image.
                           This image should contain mask of the object, so that the program could differentiate where the desired object is.
                           
  -m [ --model ] arg       Path to a directory with model images.
                           Requirements are the same as for calibration images.
                           Amount of model images must be exact as the amount of calibration images.
                           Each calibration image must correspond with one model image.
```

Run program for horse from dataset
```bash
~/msu_photometric_stereo$ ./build/PhotometricStereo -c dataset/chrome/ -m dataset/horse/
```

### Dependencies

1. Base
    ```bash
    ~$ sudo apt update
    ~$ sudo apt install -y g++ wget unzip build-essential cmake mesa-common-dev mesa-utils freeglut3-dev python3-dev python3-venv git-core ninja-build sudo lib{glvnd,boost,gtk2.0}-dev pkg-config
    ```
1. OpenCV (copied from [here](https://docs.opencv.org/4.x/d7/d9f/tutorial_linux_install.html))
    ```bash
    ~$ wget -O opencv.zip https://github.com/opencv/opencv/archive/4.x.zip
    ~$ unzip opencv.zip
    ~$ mv opencv-4.x opencv
    ~$ mkdir -p build && cd build
    ~/build$ cmake -GNinja ../opencv
    ~/build$ cmake --build .
    ~/build$ sudo make install
    ```
1. vtk (copied from [here](https://gitlab.kitware.com/vtk/vtk/-/blob/master/Documentation/dev/getting_started_linux.md))
    ```bash
    ~$ wget https://www.vtk.org/files/release/9.2/VTK-9.2.6.tar.gz
    ~$ tar -xf VTK-9-2-6.tar.gz
    ~$ mkdir vtk_build && cd vtk_build
    ~/vtk_build$ cmake -GNinja ../VTK-9-2-6
    ~/vtk_build$ cmake --build .
    ~/vtk_build$ sudo cmake -DCMAKE_BUILD_TYPE=Release -P cmake_install.cmake 
    ```
