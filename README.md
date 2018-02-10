# Image to latex
Mathematical expression recognition

Generating latex strings from math equation images using LSTM.
Early WIP. More to be added soon.

### Requirements
* python3.5 or above
* pytorch 0.3.0
* numpy, scipy
* OpenCV 3.4.0 for python
* matplotlib

### Dataset (for training purpose)
CROHME 2013, you can download it [here](http://www.iapr-tc11.org/mediawiki/index.php/CROHME:_Competition_on_Recognition_of_Online_Handwritten_Mathematical_Expressions).

After downloading, your dataset directory should be structured as follow:

* data
    * CROHME2013
        * TrainINKML
            * expressmatch
            * HAMEX
            * KAIST
            * MathBrush
            * MfrDB
        * TestINKML

You can modify the DATASET_PATH in config.py to point to the location of your CROHME2013 folder.
