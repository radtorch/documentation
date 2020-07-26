
## Starting

Training a state-of-the-art DICOM image classifier can be done using the Image Classification Pipeline via:

```
from radtorch import pipeline

classifier = pipeline.Image_Classification(data_directory='path to data')
classifier.run()
```
<small>
The above 3 lines of code will run an image classifier pipeline using ResNet50 architecture with ImageNet pre-trained weights.
</small>



## Google Colab Playground
RADTorch playground for testing is provided on [Google Colab](https://colab.research.google.com/drive/1O7op_RtuNs12uIs0QVbwoeZdtbyQ4Q9i).



<small> Documentation Update: 08/01/2020 </small>
