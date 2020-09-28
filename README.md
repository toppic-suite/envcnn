# EnvCNN Manual #
Note: Please be sure to add ```/envcnn/src``` into *$PYTHONPATH* before running the code.

## Data Pre-Processing ##
#### Required Files ####
1.	An mzML/mzXML file containing top-down spectral data. Alternatively, a Raw file can also be used.
2.	A corresponding protein database file in fasta format.  
#### Process ####
1.	Use ProteoWizard MSConvert to convert raw file into mzML file. 
    1.	Make sure you have selected the peak picking option for obtaining the centroided data. 
    2.	The process is described in *"File format conversion"* at http://proteomics.informatics.iupui.edu/software/toppic/tutorial.html 
2.	Use TopFD to deconvolute mzML/mzXML file.  
```./topfd spectrum.mzml```
    1.	You can download/clone the code from https://github.com/toppic-suite/toppic-suite/tree/master using GitHub branch *"for_EnvCNN"*.
    2.  The process will generate feature files, msalign files and a list of env files.
3.	Use TopPIC to perform proteoforms search.  
```./toppic -k -p 0 -d -t FDR -v 1E-10 -T FDR -V 1E-10 protein_db.fasta spectrum.msalign```
    1.  It will utilize protein database (*.fasta) file, msalign files and feature files.
    2.  Make sure you save temporary files.  

## Generating Training Data ##
#### Required Files ####
1.	Envelope files (*.env) files generated by TopFD.
2.	PrSM files (*.xml) files reported by TopPIC.  
#### Process ####
1.	Rename the PrSM (*.xml) files obtained using TopPIC. The files are renamed to “sp_spectrumID.xml”.  
```envcnn/src/EnvCNN/linux_batch_scripts/rename_xml_files.sh```
2.	Rename the Envelope (*.env) files obtained using TopFD. The files are renamed to “sp_spectrumID.env”.  
```envcnn/src/EnvCNN/linux_batch_scripts/rename_env_files.sh```
    1.  Make a new directory named *data*.
    2.  Copy the renamed files (both “sp_spectrumID.xml” and “sp_spectrumID.env” data files) into the *data* directory. 
    2.  Switch to the *data* directory for the onward analysis.
3.	Generate the annotated spectra files.  
```envcnn/src/EnvCNN/linux_batch_scripts/add_anno.sh```
    1.  The script will read the “sp_spectrumID.env” and “sp_spectrumID.xml” files from the *data* directory.
    2.  It will generate the "annotated_spectrumID.env" files.
4.	Generate the feature file.  
```envcnn/src/EnvCNN/linux_batch_scripts/add_feature.sh```
    1.  The script will read the “annotated_spectrumID.env” files from the *data* directory.
    2.  It will generate the "feature_spectrumID.env" files.
5.	Generate the training data.   
```envcnn/src/EnvCNN/linux_batch_scripts/generate_training_data.sh```
    1.  The script will read the “feature_spectrumID.env” files from the *data* directory.
    2.  The script will generate a folder named *TrainData*.
    3.  *TrainData* folder will contain training data matrix files for each envelope, "matrix_spectrumID_envelopeID.csv", a label file and a parameter file.
6.	HDF5 file generation  
```/envcnn/src/EnvCNN/Exec/create_hdf5_file.py TrainData/```
    1.  Move outside *data* directory.
    2.  Using the Training data in *TrainData* directory, create an HDF5 file.
    3.  The HDF5 file contains three partitions for training, validation, and test data.

## Train Model ##
1.  To train the model using a GPU/CPU.  
```/envcnn/src/EnvCNN/Exec/train_model_hdf5.py dataset.hdf5```
2.  To train the model using multiple GPUs.  
```/envcnn/src/EnvCNN/Exec/train_model_hdf5_multiGPU.py dataset.hdf5```
    1.	The script uses HDF5 file for training the model.
    2.  The script will generate a directory named *output*.
    3.  It will save the trained model to the *output* directory in *model.h5* file.
    3.  It will also generate the training plots (loss and accuracy).

## Test Model ##
1.	Test the model performance.  
```/envcnn/src/EnvCNN/Exec/test_model.py dataset.hdf5 output/```
    1.  The script uses the trained model *output/model.h5* and the test (*.hdf5) file for testing the model.
    2.  It will report the test data accuracy, loss, ROC curve, label distribution, and performance on b- and y-ions.
2.	Generate prediction score.  
```/envcnn/src/EnvCNN/Exec/add_prediction_score.py data/ output/```
    1.  The script uses the “feature_spectrumID.env” files and the trained model *output/model.h5*.
    2.  The script will generate a new directory named *output_envelopes*.
    3.  It will add *predicted_spectrumID.env* files with prediction score in the *output_envelopes* directory.

## Comaprison of EnvCNN prediction score and MS-Deconv score ##
1.	Generate the Rank plot.  
```/envcnn/src/EnvCNN/Exec/generate_rank_plot.py ./output/output_envelopes/ > roc.txt```
    1.  The script uses the *predicted_spectrumID.env* files.
    2.  It will generate the *ROC.png* file comparing the performance of EnvCNN and MS-Deconv and will output the AUC values in *roc.txt*.
2. Generate the ROC plot.  
```/envcnn/src/EnvCNN/Exec/generate_roc_plot.py ./output/output_envelopes/```
    1.  The script uses the *predicted_spectrumID.env* files.
    2.  It will generate the *rank.png* file comparing the performance of EnvCNN and MS-Deconv.
3. Generate the rank-sum.  
```/envcnn/src/EnvCNN/Exec/compute_ranksum.py ./output/output_envelopes/ > ranksum.txt```
    1.  The script uses the *predicted_spectrumID.env* files.
    2.  It will output the rank-sum values of EnvCNN and MS-Deconv.

## Deconvoluting MS/MS spectra using EnvCNN ##
1.  Download TopFD utilizing EnvCNN model for scoring isotopmer envelopes.
    1.  You can downlaod the binaries using: https://github.com/toppic-suite/envcnn/releases/download/v1.0-beta/topfd-with-EnvCNN-linux-1.0-beta.zip 
    2.  You can download/clone the modified version of the TopFD from https://github.com/toppic-suite/toppic-suite using GitHub branch *"with_EnvCNN"*.
2. Accomodate your own trained model.
    1.	Use the *model_convert.py* provided in Frugally-deep library to convert your model to json file.
        1. The source code of frugally deep is available at https://github.com/Dobiasd/frugally-deep 
    2.	Copy the json file to the *toppic_resources* directory in the toppic repository and run the modified TopFD version available at aforementioned links.
        1. A pre-trained version of the model is already available in the *toppic_resources* directory
        
## Training Data ##
1. Ovarian tumor data (used for training the model) has been made available at https://iu.box.com/s/xiantaww5n23f5u5eb7yusf3lc7hulcp
    1. There is a directory for each replicate containing corresponding envelope, prsm, annotated, feature and hdf5 files. 
    2. A file named *"dataset.hdf5"* contains training, validation and test data generated using all replicates.
        1.  First nine replicates have been used for generating training and validation data.
        2.  Last replicate (Replicate_10) has been used for generating test data.
