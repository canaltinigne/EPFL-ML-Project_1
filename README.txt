{\rtf1\ansi\ansicpg1252\cocoartf1561\cocoasubrtf600
{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww13300\viewh10380\viewkind0
\pard\tx566\tx1133\tx1700\tx2267\tx2834\tx3401\tx3968\tx4535\tx5102\tx5669\tx6236\tx6803\pardirnatural\partightenfactor0

\f0\fs24 \cf0 Authors:\
- Can Yilmaz Altinigne (can.altinigne@epfl.ch)\
- Gunes Yurdakul (gunes.yurdakul@epfl.ch)\
- Bahar Aydemir (bahar.aydemir@epfl.ch)\
\
\
Description:\
Discovering evidence of the Higgs boson is a particular game-changer in the field of particle physics. Besides the conventional methods, machine learning techniques are proven to be effective on complex and high-dimensional datasets. In this project, we have developed and compared several models by various machine learning techniques. Our model's objective is to predict whether given features of a collision event is a result of Higgs boson or a background noise.\
\
\
PUT TRAIN AND TEST SET INTO /data\
\
We have separated our code to improve modularity. For data analysis, feature engineering, training and test purposes we have used Jupyter notebook. It provided us easy visualization and code manageability. We have also added the ipynb files to present our exact process through the project. \
\
cross_validation.py : Methods for building indices, splitting dataset and cross validation for different models are implemented.\
\
errors.py : Calculation of MSE, MAE, log loss provided. \
\
gradient.py : Gradient and logistic gradient calculations reside here.\
\
implementations.py : The 6 basic method implementations are implemented.\
\
polynomial.py :  Builds polynomial dataset with the given degree.\
\
proj1_helpers.py : Loading and writing to csv functions are provided.\
\
run.py : The py version of the Test Set Prediction Notebook. The features are selected according to the process explained in the project report. The test set is also divided into subsets. The features that have meaningless values according to the PRI_jet_num category also deleted from these sets. Since our model takes a long time to train from the beginning, we have added a flag at the top of the script. By disabling it, you can produce predictions with pretrained weights by loading the weights.\
\
}