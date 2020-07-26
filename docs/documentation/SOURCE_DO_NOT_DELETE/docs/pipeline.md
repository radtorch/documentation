# Pipeline Module <small> (radtorch.pipeline) </small>


<p style='text-align: justify;'>
Pipelines are the most exciting feature of RADTorch Framework. With only few lines of code, pipelines allow you to run state-of-the-art machine learning algorithms and much more.
</p>

<p style='text-align: justify;'>
RADTorch follows principles of <b>object-oriented-programming</b> (OOP) in the sense that RADTorch pipelines are made of <a href='../core/'>core building blocks</a> and each of these blocks has specific functions/methods that can be accessed accordingly.
</p>


<p style='text-align: justify;'>

For example,
</p>

    pipeline.Image_Classification.data_processor.dataset_info()

<p style='text-align: justify;'>
can be used to access the dataset information for that particular Image Classification pipeline.
</p>

## Import



    from radtorch import pipeline




## Image Classification
```
pipeline.Image_Classification(
              data_directory, name=None, table=None,
              image_path_column='IMAGE_PATH', image_label_column='IMAGE_LABEL',is_path=True,
              is_dicom=False, mode='RAW', wl=None, custom_resize=False,
              balance_class=False, balance_class_method='upsample',
              interaction_terms=False, normalize=((0,0,0), (1,1,1)),
              batch_size=16, num_workers=0,
              sampling=1.0, test_percent=0.2, valid_percent=0.2,
              model_arch='resnet50', pre_trained=True, unfreeze=False,
              type='nn_classifier', custom_nn_classifier=None,
              cv=True, stratified=True, num_splits=5, parameters={},
              learning_rate=0.0001, epochs=10, lr_scheduler=None,
              optimizer='Adam', loss_function='CrossEntropyLoss',
              loss_function_parameters={}, optimizer_parameters={},
              transformations='default', extra_transformations=None,
              device='auto',auto_save=False)

```

<!-- !!! quote "" -->


**Description**

Complete end-to-end image classification pipeline.

**Parameters**

| Parameter 	| Type 	| Description 	| Default Value 	|
|-	|-	|-	|-	|
| **General Parameters** 	|  	|  	|  	|
| name 	| string, optional 	| name to be given to this classifier pipeline 	|  	|
| data_directory 	| string, required 	| path to target data directory/folder 	|  	|
| is_dicom 	| boolean, optional 	| True if images are DICOM 	| False 	|
| table 	| string or pandas dataframe, optional 	| path to label table csv or name of pandas data table 	| None 	|
| image_path_column 	| string, optional 	| name of column that has image path/image file name. 	| 'IMAGE_PATH' 	|
| image_label_column 	| string, optional 	| name of column that has image label. 	| 'IMAGE_LABEL' 	|
| is_path 	| boolean, optional 	| True if file_path column in table is file path.  If False, this assumes that the column contains file names only and will append the data_directory to all files 	| True 	|
| mode 	| string, optional 	| mode of handling pixel values from DICOM to numpy array. Option={'RAW': raw pixel values, 'HU': converts pixel values to HU using slope and intercept, 'WIN':Applies a certain window/level to HU converted DICOM image, 'MWIN': converts DICOM image to 3 channel HU numpy array with each channel adjusted to certain window/level 	| 'RAW' 	|
| wl 	| tuple or list of tuples, optional 	| value of Window/Level to be used. If mode is set to 'WIN' then wl takes the format (level, window). If mode is set to 'MWIN' then wl takes the format [(level1, window1), (level2, window2), (level3, window3)] 	|  	|
| balance_class 	| boolean, optional 	| True to perform oversampling in the train dataset to solve class imbalance 	| True 	|
| balance_class_method 	| string, optional 	| methodology used to balance classes. Options={'upsample', 'downsample'} 	| 'upsample' 	|
| interaction_terms 	| boolean, optional 	| create interaction terms between different features and add them as new features to feature table 	| False 	|
| normalize 	| boolean/False or Tuple, optional 	| normalizes all datasets by a specified mean and standard deviation. Since most of the used CNN architectures assumes 3 channel input, this follows the following format ((mean, mean, mean), (std, std, std)) 	| ((0,0,0), (1,1,1)) 	|
| batch_size 	| integer, optional 	| batch size for dataloader 	| 16 	|
| num_workers 	| integer, optional 	| number of CPU workers for dataloader 	| 0 	|
| sampling 	| float, optional 	| fraction of the whole dataset to be used 	| 1.0 	|
| test_percent 	| float, optional 	| percentage of data for testing 	| 0.2 	|
| valid_percent 	| float, optional 	| percentage of data for validation 	| 0.2 	|
| custom_resize 	| integer, optional 	| by default, the data processor resizes the image in dataset into the size expected bu the different CNN architectures. To override this and use a custom resize, set this to desired value 	| False 	|
| transformations 	| list, optional 	| list of pytorch transformations to be applied to all datasets. By default, the images are resized, channels added up to 3 and greyscaled 	| 'default' 	|
| extra_transformations 	| list, optional 	| list of pytorch transformations to be extra added to train dataset specifically 	| None 	|
| model_arch 	| string, required 	| CNN model architecture that this data will be used for default image resize, feature extraction and model training (model training only if type is nn_classifier) 	| 'resnet50' 	|
| pre_trained 	| boolean, optional 	| initialize CNN with ImageNet pretrained weights or not 	| True 	|
| unfreeze 	| boolean, required 	| unfreeze all layers of network for retraining 	| False 	|
| device 	| string, optional 	| device to be used for training. Options{'auto': automatic detection of device type, 'cpu': cpu, 'cuda': gpu} 	| 'auto' 	|
| type 	| string, required 	| type of classifier. For complete list refer to settings 	| 'logistic_regression' 	|
| **Classifier specific parameters** 	|  	|  	|  	|
| cv 	| boolean, required 	| True for cross validation 	| True 	|
| stratified 	| boolean, required 	| True for stratified cross validation 	| True 	|
| num_splits 	| integer, required 	| number of K-fold cross validation splits 	| 5 	|
| parameters 	| dictionary, optional 	| optional user-specified parameters passed to the classifier. Please refer to sci-kit-learn documentation. 	|  	|
| **NN_Classifier specific parameters** 	|  	|  	|  	|
| learning_rate 	| float, required 	| CNN learning rate 	| 0.0001 	|
| epochs 	| integer, required 	| number of training epochs 	| 10 	|
| optimizer 	| string, required 	| neural network optimizer type. Please see radtorch.settings for list of approved optimizers 	| 'Adam' 	|
| optimizer_parameters 	| dictionary, optional 	| optional user-specific extra parameters for optimizer as per pytorch documentation. 	|  	|
| loss_function 	| string, required 	| neural network loss function. Please see radtorch.settings for list of approved loss functions 	| 'CrossEntropyLoss' 	|
| loss_function_parameters 	| dictionary, optional 	| optional user-specific extra parameters for loss function as per pytorch documentation. 	|  	|
| custom_nn_classifier 	| pytorch model, optional 	| Option to use a custom made neural network classifier that will be added after feature extracted layers 	| None 	|
| **Beta** 	|  	|  	|  	|
| lr_scheduler 	| string, optional 	| ***in progress*** 	|  	|
| auto_save 	| boolean, optional 	| ***in progress*** 	|  	|
|  	|  	|  	|  	|

**Methods**

In addition to [core component methods](../core/), image classification pipeline specific methods include:

| Method 	| Description 	| Parameters 	| Default Value 	|
|-	|-	|-	|- |
| .info() 	| show information of the image classification pipeline. 	|  	|  	|
| .run() 	| starts the image classification pipeline training. 	|  	|  	|
| .metrics(figure_size=(700, 350) 	| displays the training metrics of the image classification pipeline. 	| figure_size: Size of display figure 	| (700,350) 	|
| .export(output_path) 	| exports the pipeline to output path. 	| output_path: path to exported pipeline file 	|	|
|.cam(target_image_path, target_layer, type='scorecam', figure_size=(10,5), cmap='jet', alpha=0.5)   | diplays class activation maps from a specific layer of the trained model on a target image  | target_image_path: path to target image <br> <br> target_layer: target layer in trained model <br> <br> type: cam type . Options = 'cam', 'gradcam', 'gradcampp', 'smoothgradcampp', 'scorecam' <br> <br> figure_size: size of display figure <br> <br> cmap: color map <br> <br> alpha: overlay alpha | <br> <br> <br>  <br> <br> <br>  'scorecam'  <br> <br> <br> <br>  <br><br> (10,5) <br><br><br> 'jet' <br><br> 0.5	|
| **Beta** 	|  	|  	|  	|
| .deploy	|  	|  ***in progress*** 	|  	|

!!! info "Visualize Class Activatin Maps (CAM) for a trained image classification pipeline"

      **Requirements:**

      1. Trained image classification pipeline.

      2. Select a target image.

      3. Identify a target layer in the trained model. This is done using the **show_model_layers** method of the [nn_classifier core module](../core/#nn_classifier).

      4. Use the **.cam** method.

      **Example**

      Assume that your trained image classifier pipeline is ***clf*** and the target image is '/image/test.png'. To select the target layer use:

      ```
      clf.classifier.show_model_layers()

      ```

      Then use the following method to show CAM :

      ```
      clf.cam(target_image_path='/image/test.png', target_layer=clf.trained_model.layer4[2].conv3)

      ```
      <div style="text-align:center"><img src="/img/cam_example.png" /></div>


## Hybrid Image Classification

```
pipeline.Hybrid_Image_Classification(
              data_directory, name=None, table=None,
              image_path_column='IMAGE_PATH', image_label_column='IMAGE_LABEL',is_path=True,
              is_dicom=False, mode='RAW', wl=None, custom_resize=False,
              balance_class=False, balance_class_method='upsample',
              interaction_terms=False, normalize=((0,0,0), (1,1,1)),
              batch_size=16, num_workers=0,
              sampling=1.0, test_percent=0.2, valid_percent=0.2,
              model_arch='resnet50', pre_trained=True, unfreeze=False,
              type='xgboost'
              cv=True, stratified=True, num_splits=5, parameters={},
              transformations='default', extra_transformations=None,
              device='auto',auto_save=False)

```

**Description**

Complete end-to-end image classification pipeline that uses combination of automatically generated imaging features and user provided clinical/laboratory features.

**Parameters**

The hybrid image classification pipeline uses the same parameters as the [Image_Classification](#Image-classification) pipeline.


!!! warning "Format of provided data table for hybrid model"

      The Hybrid Image Classification pipeline expects the data table to have the following headers :

      1. Column for Image Path (same as the one specified in 'image_label_path')

      2. Column for Image Label (same as the one specified in 'image_label_column')

      3. Columns for clinical/laboratory features.


!!! danger "Supported classifiers"

      The Hybrid Image Classification pipeline uses CNNs for automatic imaging feature extraction. Training of CNNs (nn_classifier) using hybrid model is not yet supported.


!!! info "Handling Categorical and Numerical Clinical/Laboratory Data"

      The hybrid image classification pipeline is capable of automatically handling categorical and numerical clinical data provided by the user.

      For example, if the data provided looks like that :

      <div style="text-align:center"><img src="/img/hybrid_1.png" /></div>

      the pipeline will automatically detect types of variables and modify accordingly to get this:

      <div style="text-align:center"><img src="/img/hybrid_2.png" /></div>



**Methods**

Hybrid image classification pipeline follows the same methods as [Image_Classification](#Image-classification) pipeline.






## Feature Extraction
```
pipeline.Feature_Extraction(
                name=None, data_directory, table=None, is_dicom=True,
                normalize=((0, 0, 0), (1, 1, 1)),
                batch_size=16, num_workers=0,
                model_arch='resnet50', custom_resize=False,
                pre_trained=True, unfreeze=False,
                label_column='IMAGE_LABEL')
```

**Description**

Image Feature Extraction Pipeline.

**Parameters**

| Parameter 	| Type 	| Description 	| Default Value 	|
|-	|-	|-	|-	|
| name 	| string, optional 	| name to be given to this classifier pipeline 	|  	|
| data_directory 	| string, required 	| path to target data directory/folder 	|  	|
| table 	| string or pandas dataframe, required 	| path to label table csv or name of pandas data table 	| None 	|
| is_dicom 	| boolean, required 	| True if images are DICOM 	| False 	|
| normalize 	| boolean/False or Tuple, optional 	| normalizes all datasets by a specified mean and standard deviation. Since most of the used CNN architectures assumes 3 channel input, this follows the following format ((mean, mean, mean), (std, std, std)) 	| ((0,0,0), (1,1,1)) 	|
| batch_size 	| integer, optional 	| batch size for dataloader 	| 16 	|
| num_workers 	| integer, optional 	| number of CPU workers for dataloader 	| 0 	|
| model_arch 	| string, required 	| CNN model architecture that this data will be used for default image resize, feature extraction and model training (model training only if type is nn_classifier) 	| 'resnet50' 	|
| custom_resize 	| integer, optional 	| by default, the data processor resizes the image in dataset into the size expected bu the different CNN architectures. To override this and use a custom resize, set this to desired value 	| False 	|
| pre_trained 	| boolean, optional 	| initialize CNN with ImageNet pretrained weights or not 	| True 	|
| unfreeze 	| boolean, required 	| unfreeze all layers of network for retraining 	| False 	|
| label_column 	| string, required 	| name of column that has image label. 	| 'IMAGE_LABEL' 	|

**Methods**

In addition to [core component methods](../core/),  feature extraction pipeline specific methods include:

| Method 	| Description 	| Parameters 	|
|-	|-	|-	|
| .info() 	| show information of the  pipeline. 	|  	|
| .run() 	| starts the feature extraction pipeline. 	|  	|
| .export(output_path) 	| exports the pipeline to output path. 	| output_path: path to exported pipeline file 	|


!!! info "Visualize Extracted Features"

      You can use the **plot_extracted_features** method of the [feature_extractor core module](../core/#feature_extractor) to visualize the extracted features as below:

      ```
      Assume that your feature extraction pipeline is called X,

      X.feature_extractor.plot_extracted_features(num_features=100, num_images=100)

      ```



## Generative Adversarial Networks

```
core.GAN(
        data_directory, table=None, is_dicom=False, is_path=True,
        image_path_column='IMAGE_PATH', image_label_column='IMAGE_LABEL',
        mode='RAW', wl=None, batch_size=16, normalize=((0,0,0),(1,1,1)),
        num_workers=0, label_smooth=True, sampling=1.0, transformations='default',

        discriminator='dcgan', generator='dcgan',
        generator_noise_size=100, generator_noise_type='normal',
        discriminator_num_features=64, generator_num_features=64,
        image_size=128, image_channels=1,

        discrinimator_optimizer='Adam', generator_optimizer='Adam',
        discrinimator_optimizer_param={'betas':(0.5,0.999)},
        generator_optimizer_param={'betas':(0.5,0.999)},
        generator_learning_rate=0.0001, discriminator_learning_rate=0.0001,        

        epochs=10, device='auto')

```

**Description**

Generative Advarsarial Networks Pipeline.


**Parameters**

| Parameter 	 	| Type 	 	| Description 	 	| Default Value 	 	|
|-	|-	|-	|-	|
| -	 	| -	 	| -	 	| -	 	|
| **General Parameters** 	|  	|  	|  	|
|  data_directory 	 	| string, required 	 	|  path to target data directory/folder 	 	|   	 	|
|  is_dicom 	 	| boolean, optional 	 	|  True if images are DICOM 	 	|  False 	 	|
|  table 	 	| string or pandas   dataframe, optional 	 	|  path to label table csv or name of pandas data table 	 	|  None 	 	|
|  image_path_column 	 	| string, optional 	 	|  name of column that has image path/image file name. 	 	|  'IMAGE_PATH' 	 	|
|  image_label_column 	 	| string, optional 	 	|  name of column that has image label. 	 	|  'IMAGE_LABEL' 	 	|
|  is_path 	 	| boolean, optional 	 	|  True if file_path column in table is file path. If False, this assumes that the column contains file names only and will append the data_directory to all file names. 	|  True 	 	|
|  mode 	 	| string, optional 	 	|  mode of handling pixel values from DICOM to numpy array. Option={'RAW': raw pixel values, 'HU':converts pixel values to HU using slope and intercept, 'WIN':Applies a   certain window/level to HU converted DICOM image, 'MWIN': converts DICOM image to 3 channel HU numpy array with each channel adjusted to certain   window/level 	 	|  'RAW' 	 	|
|  wl 	 	| tuple or list of   tuples, optional 	 	|  value of Window/Level to be used. If mode is set to 'WIN' then wl takes the format (level, window).   If mode is set to 'MWIN' then wl takes the format [(level1, window1),   (level2, window2), (level3, window3)]    	 	|   	 	|
|  batch_size 	 	| integer, optional 	 	|  batch size for dataloader 	 	| 16 	 	|
|  num_workers 	 	| integer, optional 	 	|  number of CPU workers for dataloader 	 	| 0 	 	|
|  sampling 	 	| float, optional 	 	|  fraction of the whole dataset to be used 	 	| 1.0 	|
|  transformations 	 	| list, optional 	 	|  list of pytorch transformations to be applied to all datasets. By default, the images are resized, channels added up to 3 and greyscaled 	 	| 'default' 	 	|
|  normalize 	 	| boolean/False or Tuple,   optional 	 	| normalizes all datasets by a specified mean and standard deviation. Since most of the used CNN architectures assumes 3 channel input, this follows the following format   ((mean, mean, mean), (std, std, std)) 	 	| ((0,0,0), (1,1,1)) 	|
|  epochs 	 	| integer, required 	 	| number of training epochs 	 	| 10 	 	|
| image_channels 	| integer, required 	| number of channels for discriminator input and generator output 	| 1 	|
| image_size 	| integer, required 	| image size for discriminator input and generator output 	| 128 	|
|  device 	 	| string, optional 	 	| device to be used for training. Options{'auto': automatic detection of device type, 'cpu': cpu,   'cuda': gpu} 	 	| 'auto' 	 	|
| **Discriminator Parameters** 	|  	|  	|  	|
| discriminator 	| string, required 	| type of discriminator network. Options = {'dcgan', 'vanilla', 'wgan'} 	| 'dcgan' 	|
| discriminator_num_features 	| integer, required 	| number of features/convolutions for discriminator network 	| 64 	|
| label_smooth 	| boolean, optioanl 	| by default, labels for real images as assigned to 1. If label smoothing is set to True, lables of real images will be assigned to 0.9.   <br>(Source: https://github.com/soumith/ganhacks#6-use-soft-and-noisy-labels) 	| True 	 	|
| discrinimator_optimizer | string, required 	| discriminator network optimizer type. Please see radtorch.settings for list of approved optimizers 	| 'Adam' 	|
| discrinimator_optimizer_param | dictionary, optional 	| optional extra parameters for optimizer as per pytorch  documentation. | {'betas':(0.5,0.999)} <br> for Adam optimizer. 	|
| discriminator_learning_rate	|float, required 	| discrinimator network learning rate | 0.0001 	|
| **Generator Parameters** 	|  	|  	|  	|
| generator 	| string, required 	| type of generator network. Options = {'dcgan', 'vanilla',   'wgan'} 	| 'dcgan' 	|
| generator_noise_type 	| string, optional 	| shape of noise to sample from. Options={'normal', 'gaussian'}   <br>(Source: (https://github.com/soumith/ganhacks#3-use-a-spherical-z) 	| 'normal' 	|
| generator_noise_size 	| integer, required 	| size of the noise sample to be generated 	| 100 	|
| generator_num_features 	| integer, required 	| number of features/convolutions for generator network 	| 64 	|
| generator_optimizer 	| string, required 	| generator network optimizer type. Please see radtorch.settings for list of approved optimizers 	| 'Adam' 	|
| generator_optimizer_param 	| dictionary, optional 	| optional extra parameters for optimizer as per pytorch documentation 	| {'betas':(0.5,0.999)} <br> for Adam optimizer. 	|
| generator_learning_rate 	| float, required 	| generator network learning rate 	| 0.0001 	|

**Methods**
```
.run(self, verbose='batch', show_images=True, figure_size=(10,10))
```

- Runs the GAN training.

- Parameters:

    - verbose (string, required): amount of data output. Options {'batch': display info after each batch, 'epoch': display info after each epoch}.default='batch'

    - show_images (boolean, optional): True to show sample of generatot generated images after each epoch.

    - figure_size (tuple, optional): Tuple of width and length of figure plotted. default=(10,10)


```
.sample(figure_size=(10,10), show_labels=True)
```

- Displays a sample of real data.

- Parameters:

    - figure_size (tuple, optional): Tuple of width and length of figure plotted. default=(10,10).

    - show_labels (boolean, optional): show labels on top of images. default=True.

```
.info()
```

- Displays different parameters of the generative adversarial network.

```
.metrics(figure_size=(700,350))
```

- Displays training metrics for the GAN.

- Explanation of metrics:

    - *D_loss*: Total loss of discriminator network on both real and fake images.

    - *G_loss*: Loss of discriminator network on detecting fake images as real.

    - *d_loss_real*: Loss of discriminator network on detecting real images as real.

    - *d_loss_fake*: Loss of discriminator network on detecting fake images as fake.

- Parameters:

    - figure_size (tuple, optional): Tuple of width and length of figure plotted. default=(700,350).



 <small> Documentation Update: 08/01/2020 </small>
