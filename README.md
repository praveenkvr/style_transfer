## Fast style transfer

Implemented using tensorflow functional API.

examples:

Original image
![](images/original.jpg)

Style             |  Trained
:-------------------------:|:-------------------------:
![original](images/style1.jpg)  |  ![](images/style1-trained.jpg)
![original](images/style2.jpg)  |  ![](images/style2-trained.jpg)

After training to test the style on a new image

`python main.py --image <path-to-image>`