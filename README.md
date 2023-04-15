# Project_4  - "Predicting Diabetes with Artificial Intelligence"
Isabella Taylor, Leah Nash, Sandra Braun, Valerie Grannemann-Barber

## Background:
Diabetes is a condition where the body is unable make enough insulin to clear glucose from the bloodstream. This results in increased glucose levels that can seriously effcect a person's health. Diabetes can cause heart problems, kidney damage, hearing problems or in extreme cases, death.  According to the Centers for Disease Control, over 37 million people in the United States have the condition.

There are interventions that can improve the condition, but people often do not realize that they have diabetes. This is espcially the case with prediabetes, which is just before the onset of diabetes and often doesn't have any symptons.  However, if we could determine how suspetible a person is to getting diabetes or prediabetes early on, medical professionals would have the opportunity to help patients change their trajectory. In the long run, it would improve a patient's quality of life and also reduce healthcare costs. So as a group, we set out to build a model to predict diabetes and prediabetes.


## Data Source: 
Our data was sourced from Kaggle. The "Diabetes Health Indicators Dataset Notebook" is a subset of a larger dataset from Behavioral Risk Factor Surveillance System(BRFSS). BRFSS conducts over 400,000 surveys each year and the resulting dataset includes participant responses to hundreds of health related questions.  Our reduced dataset has 22 variables ranging from high blood pressure to BMI to Education (full list below), and we used Diabetes_012 as our target variable. Diabetes_012 has has three distinct classes: "0"= no diabetes, "1" - prediabetes, "2" -diabetes. 

Dataset: https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset

Variables:
Diabetes_012
HighBP
HighChol
CholCheck
BMI
Smoker
Stroke
HeartDiseaseorAttack
PhysActivity
Fruits
Veggies
HvyAlcoholConsump
AnyHealthcare
NoDocbcCost
GenHlth
MentHlth
PhysHlth
DiffWalk
Sex
Age
Education
Income

 

Technologies:
Spark
Spark.sql
Pandas
Tableau
Sklearn 
Matplotlib
Imblearn
Tensorflow




## Process Overview:
To begin, we imported the data using Spark and explored the dataset.  We used Sparksql queries to understand the distribution of the classes by age, education, income, etc. We then transitioned from using a Spark dataframe to a Pandas dataframe which allowed us to further review the data. We felt the dataset did not require any cleaning so we moved forward with constructing machine learning models. 

We each built and tested several models, a general breakdown of the types are listed below:

    Logistic Regression: Sandra, Leah, Valerie
    Random Forrest: Isabella
    Neural Networks: Valerie, Isabella

We experimented with different configurations, creating a new target by combining features and also extracting classes from the target. 
   
After building our first set of models, we realized that our data needed to be cleaned some more. So, we encoded the Diabetes_012 , Age, Income and Education cloumns, and also over sampled the data to offset the negative effect of having an imbalanced dataset. Ultimately, we chose the models that performed the best to feature in our final notebook. 


Outline of Google Colab Models/Cleaning

1. Random Forest Model on Unclean Data
- didn't predict for pre-diabetes well
- did give us feature importance
![feature importance graph](/Project_4/Images/FeagureImportances.png)
![Random Forrest confusion matrix](/Project_4/Images/RandomForrestCofusion.png)
![accuracy score](/Project_4/Images/RFAccuracy.png)
![classification report](/Project_4/Images/RFclassificaton.png)

2. Data Modeling, Cleaning CSV
- split all categorical columns with get_dummies
- Diabetes_012, Age, Education, Income


3. Over Sampling
- utilized synthetic minority over-sampling technique (smote) to address class
imbalance in the pre-diabetes disease status population
![insert image of oversample distribution plot]()


4. Logistic Regression using Oversampled Data
- created a logistic regression model 
- Oversampling still couldn't predict prediabetes
![insert image of classification report]()

5. Neural Network Model Using Oversampled Data
- created a neural network model and added layers from
a previously successful model (not in the notebook)
- Oversampling still couldn't predict prediabetes
![5NN confusion matrix](/Images/image.png)
![5NN accuracy score](/Images/5NN_accuracty.png)
![5NN classification report](/Images/5NN_classification.png)

6. Auto NN w/o prediabetes
- created an automated neural network to test the best hyperparameter options
- left prediabetes out of the target and added to the features
- got the best hyperparameters
- go the best model

7. NN Model using Best Hyperparameters
- created a model using the best hyperparameters from the auto NN
- Did not perform how we'd like
![7NN loss plot](/Images/7NN_loss_plot.png)
![7NN accuracy plot](/Images/7NN_acc_plot.png)


8. NN Model using Best Model
- created a model using the best model from the auto NN
- better accuracy and loss
![insert loss plot]()
![insert accuracy plot]()
![insert loss and accuracy]()
![insert image of confusion matrix]()
![insert image of accuracy score]()
![insert image of classification report]() 


Conclusions:
What we learned was not all models work a like or for our set of data. Models were ran on the original dataset from Kaggle
and then was cleaned for our target variables.

We can use machine learning to help predict indicators that cause 
diabetes and/or prediabetes.  In our code and vizualizations, we shared interesting
facts and collaborations based on other demographic, economic, and physical health indicators.

https://github.com/L-Nash/Project_4