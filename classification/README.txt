#####################################################
                                                    
# written by Rija Tonny C. Ramarolahy               
                                                     
# 2020                                               
                                                    
# email: rija@aims.edu.gh / minuramaro@gmail.com   
                                                     
#####################################################



Malaria blood detector based Deep Learning.

These are the codes used in the project.

The code is made of two part: training and testing model & detection of parasites on an image(automated detection).

In this code we used two datasets.
1-The first data contains individual cells from https://ceb.nlm.nih.gov/repositories/malaria-datasets/.
2-The second contains thick smear image from http://air.ug/microscopy/.

We created a CNN model and decided to train on both datasets.

So if you want to train by yourself, make sure that you have the datasets.
And you can decide which data should you train. You can run main.py in the terminal and add the directory of the data that you want to train.
For example: python3 main.py './cell_images' 
In this case, you decided to train the individual cells datasets.

For this first dataset, it will be better if you use GPU. If you don't have, you can use google COLAB like I did.
For the use of COLAB, you need to have the data of the individual cells i.e train, test, validation. 
So you need to find a way to run the function data_preparation() inside Data_Ind_cells.py and save the data in your drive.
If you got the data, then you can connect your drive to COLAB and run 'Training_ind_cells.ipynb'.



For the second part, the detection, you need an image of either thick nor thin blood smear and run the appropriate code. Make sure that you have the model before running the code.

To run this code, you need to put the model and the image like this python3 thin_evaluation.py model image.
For example: python3 thin_evaluation.py models/CNN_ind_cells.h5 '2a_002.JPG' 

For thick_evaluation.py, if you do not have the annotated data (.xml), you can comment line 86-95.


These codes were runned on tensorflow 2.1.0

NB: -You need some image of thin smear and thick smear for the second part of the code. For thick smear, you can use the data itself.
For thin smear, you can try this https://drive.google.com/drive/folders/1EMJ7dg0TBs34sDWcj7Tj1wozXJC0wtbc

    -Everyone can improve the codes especially for the automated detection.

    -Based on the segmentation code(thick_evaluation.py), you can also create a data from thin image datasets and re-train the model. 
