# Project_4  - "Predicting Diabetes with Artificial Intelligence"
Isabella Taylor, Leah Nash, Sandra Braun, Valerie Grannemann-Barber

## Background:
Diabetes is a medical condition where the body is unable to make enough insulin to clear glucose from the bloodstream. This results in increased glucose levels that can seriously affect a person's health. Diabetes can cause heart problems, kidney damage or hearing problems. According to the Centers for Disease Control, it is the eighth leading cause of death in the United States.  Additionally, the diabetes population has more than doubled in the last 20 years and currently over 37 million people in the United States have the condition.

There are interventions that can improve the condition, but about 20% of people do not realize that they have it. This is especially the case with prediabetes, which is just before the onset of diabetes and often doesn't have any symptoms.  However, if we could determine how susceptible a person is to getting diabetes or prediabetes early on, medical professionals would have the opportunity to help patients change their trajectory. In the long run, being able to make these predictions could improve a patient's quality of life and reduce healthcare costs. We wanted to take on this challenge and set out to build a machine learning model that can predict diabetes and prediabetes.


## Data Source: 
Our data was sourced from Kaggle. The "Diabetes Health Indicators Dataset Notebook" is a subset of a larger dataset from the Behavioral Risk Factor Surveillance System (BRFSS). BRFSS conducts over 400,000 surveys each year and the resulting dataset includes participant responses to hundreds of health-related questions.  Our reduced dataset has 22 variables ranging from high blood pressure to BMI to Education (full list below), and we used 'Diabetes_012' as our target variable. 'Diabetes_012' has three distinct classes: "0" = no diabetes, "1" = prediabetes, "2" = diabetes. 

Dataset: https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset

Variables:
* **Diabetes_012** - Whether the participant has no diabetes, diabetes, or prediabetes.
* **HighBP** - Whether the participant has ever been told they have high blood pressure.
* **HighChol** - Whether the participant has ever been told they have high cholesterol. 
* **CholCheck** - Whether the participant has had their cholesterol checked in the last 5 years.
* **BMI** - What the participants body mass index is.
* **Smoker** - Whether the participant has smoked at least 100 cigarettes in their entire life.
* **Stroke** - Whether the participant has ever been told they had a stroke.
* **HeartDiseaseorAttack** - Whether the participant has ever been told they have coronary heart disease or myocardial infarction.
* **PhysActivity** - Whether the participant has had any physical activity in the past 30 days that isn't their job.
* **Fruits** - Whether the participant consumes 1 or more fruits per day.
* **Veggies** - Whether the participant consumes 1 or more vegetables per day.
* **HvyAlcoholConsump** - Whether the participant is a heavy drinker. (Having more than 14 drinks per week for men, and 7 for women).
* **AnyHealthcare** - Whether the participant has any kind of health care coverage.
* **NoDocbcCost** - Whether the participant needed to see a doctor but couldn't because of cost within the past 12 months.
* **GenHlth** - Whether the participant would say what their health is like in general on a scale of 1-5. 1 being excellent and 5 being poor.
* **MentHlth** - How many days on a scale of 1-30 that the participant has felt like their mental health has been poor. (Depression, stress, and problems with emotions). 
* **PhysHlth**- How many days on a scale of 1-30 that the participant has felt like their physical health has been poor. (Physical illness and injury).
* **DiffWalk** - Whether the participant has serious difficulty walking or climbing stairs.
* **Sex** - Whether the participant is female or male.
* **Age** - 13 level age category. 1 being 18-24, 13 being 80+
* **Education** - What the participants education category is.
* **Income** - What the participants income category is.


## Process Overview:
To begin, we imported the data using Spark and explored the dataset.  We used Sparksql queries to understand the distribution of the classes by age, education, income, etc. We then transitioned from using a Spark dataframe to a Pandas dataframe which allowed us to further review the data. At the time, we felt the dataset did not require cleaning, so we moved forward with constructing machine learning models. 

We each built and tested several models; a general breakdown of the types is listed below:

    Logistic Regression: Sandra, Leah, Valerie, Isabella
    Random Forest: Isabella
    Automated Neural Networks: Isabella
    Neural Networks: Valerie, Isabella
    Oversampling: Sandra

We experimented with different configurations, creating a new target by combining features and extracting classes from the target. Ultimately, we chose the models that performed the best to feature in our final notebook. A description of each is listed in the steps below:


## 1. Random Forest Model on Original Data:
Although this model had high accuracy, it didn't do a good job of predicting prediabetes. However, we were able to extract the feature importance which helped us to better understand the data.
![feature importance graph](/Images/FeagureImportances.png)

![Random Forrest confusion matrix](/Images/RandomForrestCofusion.png)

![accuracy score](/Images/RFAccuracy.png)

![classification report](/Images/RFclassificaton.png)


## 2. Data Cleaning: 
After building our first set of models, we realized that our data needed further cleaning, so we encoded the categorical columns ('Diabetes_012', 'Age', 'Income', and 'Education').


## 3. Over Sampling:
We utilized synthetic minority over-sampling technique (smote) to address class imbalance in the prediabetes disease status population.

![oversample distribution plot](/Images/Count_status.png)


## 4. Logistic Regression using Oversampled Data:
We'd hoped that oversampling would allow our model to predict prediabetes, but unfortunately that was not the case.

![classification report](/Images/logistic_classification.png)

## 5. Neural Network Model Using Oversampled Data:
The first neural network model featured in the notebook used layer specifications from
a previously successful model (not in the notebook). And just like with the logistic regression model, oversampling did not fix the prediabetes prediction problem. 

![5NN confusion matrix](/Images/5NN_confustion.png)

![5NN accuracy/loss](/Images/5NN_loss_accuracy.png)

![5NN classification report](/Images/5NN_classification.png)

## 6. Auto NN w/o prediabetes:
Next, we created an automated neural network to test the best hyperparameter options, excluded prediabetes from the target and added it as a feature. We used the results to build the model in the next step. 

## 7. NN Model using Best Hyperparameters:
Using the hyperparameters from our previous step, we created another neural network (excluding prediabetes from the target). Unfortunately, it was not quite as accurate as we would have liked.

![7NN loss plot](/Images/7NN_loss_plot.png)

![7NN accuracy plot](/Images/7NN_acc_plot.png)


## 8. NN Model using Best Model:
Lastly, we created another neural network using our best model (see #6), and prediabetes was again excluded from the target. This resulted in accuracy that was slightly better than the model in step 7.

![8NN loss plot](/Images/8NN_loss.png)

![8NN accuracy plot](/Images/8NN_accuracy.png)

![8NN loss and accuracy](/Images/8NN_loss_accuracy.png)

![8NN image of confusion matrix](/Images/8NN_confusion.png)

![8NN image of classification report](/Images/8NN_classification.png) 


## Conclusion:
We were able to achieve a good overall accuracy score on our final model. But results specifically for the diabetes class were not as expected.  And, although we were unable to predict prediabetes, we learned a lot about the need to incorporate methods to balance our sample. Overall, our model needs further refinement before we would be able to use it in a practical application.  

# References

Data sourced from [Kaggle](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)

BRFSS [codebook](https://www.cdc.gov/brfss/annual_data/2015/pdf/codebook15_llcp.pdf)