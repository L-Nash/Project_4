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


## Process Overview:
To begin, we imported the data using Spark and explored the dataset.  We used Sparksql queries to understand the distribution of the classes by age, education, income, etc. We then transitioned from using a Spark dataframe to a Pandas dataframe which allowed us to further review the data. At the time, we felt the dataset did not require cleaning so we moved forward with constructing machine learning models. 

We each built and tested several models, a general breakdown of the types are listed below:

    Logistic Regression: Sandra, Leah, Valerie
    Random Forrest: Isabella
    Neural Networks: Valerie, Isabella

We experimented with different configurations, creating a new target by combining features and also extracting classes from the target. Ultimately, we chose the models that performed the best to feature in our final notebook. A descripton of each is listed in the steps below:


1. Random Forest Model on Original Data:
Although this model had high accuracy, it didn't do a good job of predicing pre-diabetes. However, we were able to extract the feature importances which helped us to better understand the data.
![feature importance graph](/Images/FeagureImportances.png)
![Random Forrest confusion matrix](/Images/RandomForrestCofusion.png)
![accuracy score](/Images/RFAccuracy.png)
![classification report](/Images/RFclassificaton.png)

2. Data Cleaning: 
After building our first set of models, we realized that our data needed further cleaning, so we encoded the categorical columns (Diabetes_012 , Age, Income and Education).


3. Over Sampling:
We utilized synthetic minority over-sampling technique (smote) to address class
imbalance in the pre-diabetes disease status population
![oversample distribution plot](/Images/Count_status.png)


4. Logistic Regression using Oversampled Data:
We'd hoped that oversampling would allow our model to predict prediabetes, but unfortunately that was not the case.
![classification report](/Images/logistic_classification.png)

5. Neural Network Model Using Oversampled Data:
The first neural network model feature in the notebook used layer specificatons from
a previously successful model (not in the notebook). And, just like with the logistic regression model, oversampling did not fix the prediabetes prediction problem. 

![5NN confusion matrix](/Images/5NN_confustion.png)
![5NN accuracy/loss](/Images/5NN_loss_accuracy.png)
![5NN accuracy score](/Images/5NN_accuracty.png)
![5NN classification report](/Images/5NN_classification.png)

6. Auto NN w/o prediabetes:
Next, we created an automated neural network to test the best hyperparameter options, excluded prediabetes from the target and added it as a feature. We used the results to build the model in the next step. 

7. NN Model using Best Hyperparameters:
Using the hyperparameters from our previous step, we created another neural network (excluding prediabetes from the target). Unfortunately, it was not quite as accurate as we would have liked. 
![7NN loss plot](/Images/7NN_loss_plot.png)
![7NN accuracy plot](/Images/7NN_acc_plot.png)


8. NN Model using Best Model:
Lastly, we created another neural networking using our best model (see #5), and prediabetes was again excluded from the target. This resulted in accuracy that was slightly better than the model in step 7.
![8NN loss plot](/Images/8NN_loss.png)
![8NN accuracy plot](/Images/8NN_accuracy.png)
![8NN loss and accuracy](/Images/8NN_loss_accuracy.png)
![8NN image of confusion matrix](/Images/8NN_confusion.png)
![8NN image of accuracy score](/Images/8NN_accuracy_score.png)
![8NN image of classification report](/Images/8NN_classification.png) 


## Conclusion:
Although we were unable to predict prediabetes we learned a lot about the need to incorporate methods to balance our sample. Further research is needed to 


What we learned was not all models work a like or for our set of data. Models were ran on the original dataset from Kaggle
and then was cleaned for our target variables.

