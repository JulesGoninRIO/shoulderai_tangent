### Local environment
You can create the environment:

```bash
conda create --name myenv
conda activate myenv
```

And install the requirements using conda and conda-forge:

```bash 
conda install -c conda-forge --file requirements.txt 
```


### Conversion of original data to JPG format for DL models
A local shoulderai_poc/data folder is created automatically and contains the JPG images and masks derived from the DICOM images and CSV annotations. Its content is as follows:
```
       shoulderai_poc/data
    ├── split_labels_fold_0.csv # 10 csv files containing train-test splits
    ├── ...
    ├── split_labels_fold_9.csv  
    ├── images # contains MRI images in form of MRI scans.     
        ├── sagittal
    ├── masks  
        ├── T1_sag
            ├── tangent_sign # contains jpg images of the mask
            ├── supraspinatus # contains jpg images of the mask          
    ├── labels.csv  

In images and masks folder jpg images have names that are ids of the patient. 
To reproduce the results of this project, one should place the MRI scans and masks in the corresponding folders in jpg format and ensure that files names of corresponding patient are the same and is id of the patient.

```

### How to run a tangent sign Unet segmentation model
The following command will create test/train splits, if they don't exist, train and test the model, will generate the csv file with dice score and differnce of the percentage of the muscle, figures presented in the paper. 

To train segmentation model run:
```bash 
python segmentation_model.py --mode train
```
To test segmentation model run:
```bash 
python segmentation_model.py --mode test
```