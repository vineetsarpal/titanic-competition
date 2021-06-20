#Load R packages
library(tidyverse)
library(ggplot2)
library(corrplot)
library(caTools)
library(naniar)
library(modeest)
library(randomForest)
library(scales)
library(ROCR)

#=================
# Step 1 Read Data
#=================

df_train <- read_csv("~/R/Data/titanic/train.csv")
df_test <- read_csv("~/R/Data/titanic/test.csv")

# Combine train and test data to perform EDA together
df_train$IsTrainSet <- TRUE
df_test$IsTrainSet <- FALSE
df_test$Survived <- NA
df_titanic_full <- bind_rows(df_train,df_test)


#===================
# Step 2 Perform EDA
#===================

# Find missing vales in the dataset
vis_miss(df_titanic_full)
vis_miss(df_train)
vis_miss(df_test)

# OBS: Lot of Age and Cabin values are missing in both train and test data

# Looking at survival nos by sex
df_train %>% 
  group_by(Sex, Survived) %>% 
  summarise(no_of_ppl = n()) %>% 
  ggplot(aes(x = factor(Survived), y = no_of_ppl, fill = Sex)) + geom_bar(stat="identity", position = "dodge")

# OBS: A large no of males have not survived

# Looking at Survival nos by class
df_train %>% 
  group_by(Pclass, Survived) %>% 
  summarise(no_of_ppl = n()) %>% 
  ggplot(aes(x = factor(Survived), y = no_of_ppl, fill = factor(Pclass))) + geom_col(stat="identity", position = "dodge")

# OBS: A huge no of Class 3 passengers did not survive while majority of Class 1 passengers did


# Distribution plot of Age for full data
df_titanic_full %>%
  drop_na(Age) %>% 
  ggplot(aes(x = Age)) + geom_histogram()

# OBS: Maximum no of passengers are in their early 20s


#==================
# Step 3 Clean Data
#==================

# Fixing missing values on full data

# 1) Age: Finding details of Age per class
df_titanic_full %>% 
  drop_na(Age) %>% 
  ggplot(aes(x=factor(Pclass), y=Age)) + geom_boxplot() + stat_summary(fun.y = "mean")

# OBS: The mean and median ages differ by passenger class

# Setting missing Age values by median (since there are some outliers) of the Age per passenger class
median_age_pclass1 <- median(df_titanic_full[!is.na(df_titanic_full$Age) & df_titanic_full$Pclass==1, ]$Age)
median_age_pclass2 <- median(df_titanic_full[!is.na(df_titanic_full$Age) & df_titanic_full$Pclass==2, ]$Age)
median_age_pclass3 <- median(df_titanic_full[!is.na(df_titanic_full$Age) & df_titanic_full$Pclass==3, ]$Age)

df_titanic_full[is.na(df_titanic_full$Age) & df_titanic_full$Pclass==1, "Age"] <- round(median_age_pclass1)
df_titanic_full[is.na(df_titanic_full$Age) & df_titanic_full$Pclass==2, "Age"] <- round(median_age_pclass2)
df_titanic_full[is.na(df_titanic_full$Age) & df_titanic_full$Pclass==3, "Age"] <- round(median_age_pclass3)


# 2) Cabin: Adding column to capture first letter of cabin to analyze
df_titanic_full <- df_titanic_full %>% 
  mutate(CabinClass = substr(df_titanic_full$Cabin,1,1))

table(df_titanic_full$CabinClass, df_titanic_full$Pclass)

# OBS: It seems cabin has been assigned to mostly Pclass 1 passengers 

df_titanic_full %>% 
  drop_na() %>% 
  group_by(CabinClass, Survived) %>% 
  summarise(no_of_ppl = n()) %>% 
  ggplot(aes(x=CabinClass, y=no_of_ppl, fill=Survived)) + geom_col()

# OBS: There's no clear indicator if people from a particular cabin had a significantly better chance of survival over the other
# We can broadly say that people who were assigned cabins are from Pclass 1
# Thus, we can use Pclass instead and drop Cabin cols

# Drop cabin columns
df_titanic_full <- subset(df_titanic_full, select = -c(Cabin, CabinClass))

# 3) Fare: Finding missing Fare record
sum(is.na(df_titanic_full$Fare))
df_titanic_full[is.na(df_titanic_full$Fare), ]

# The missing record is from Pclass 3

# Settting the avg Fare for Pclass 3 for the missing record
mean_fare_pclass3 <- mean(df_titanic_full[!is.na(df_titanic_full$Fare) & df_titanic_full$Pclass==3, ]$Fare)
df_titanic_full[is.na(df_titanic_full$Fare), "Fare"] <-round(mean_fare_pclass3, 4)

# 4) Embarked: Finding missing Embarked records
sum(is.na(df_titanic_full$Embarked))
df_titanic_full[is.na(df_titanic_full$Embarked), ]

table(df_titanic_full$Pclass, df_titanic_full$Embarked)
# Missing values are Pclass 1 passengers and Pclass 1 passengers embarked roughly equally from 'Q' and 'S'

# Assigning the mode value of Embarked to the missing records
mode_embarked <- mfv(df_titanic_full$Embarked)
df_titanic_full[is.na(df_titanic_full$Embarked), "Embarked"] <- mode_embarked


#===================
# Step 4 Build Model
#===================

# Finding correlation between variables
corr = cor(select_if(df_train,is.numeric))
corrplot(corr, type = "lower")

# OBS: Survival rate has a correlation with Pclass and Pclass in turn has correlation with Age and Fare


# Splitting the full titanic data back to train and test
df_train <- df_titanic_full[df_titanic_full$IsTrainSet == TRUE, ]
df_test <- df_titanic_full[df_titanic_full$IsTrainSet == FALSE, ]


# Split existing train data into a subset of train & test data to run test predictions
split <- sample.split(df_train$Survived, SplitRatio = 0.65)
df_model_train <- subset(df_train, split == TRUE)
df_model_test <- subset(df_train, split == FALSE)


# 1) Random Forest
#=================
model_rf <- randomForest(formula = Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked, data = df_model_train)
summary(model_rf)
res_rf <- predict(model_rf, newdata = df_model_test)
confusion_matrix_rf <- table(actual_value = df_model_test$Survived, predicted_value = res_rf > 0.5)
confusion_matrix_rf

# Model prediction stats
TN_rf <- confusion_matrix_rf[1,1]
FN_rf <- confusion_matrix_rf[1,2]
FP_rf <- confusion_matrix_rf[2,1]
TP_rf <- confusion_matrix_rf[2,2]

accuracy_rf <- (TN_rf+TP_rf) / (TN_rf+FN_rf+FP_rf+TP_rf)
precision_rf <- TP_rf / (TP_rf+FP_rf)
recall_rf <- TP_rf / (TP_rf+FN_rf)
print(paste("Accuracy = ",percent(accuracy_rf,0.01),"| Precision = ",percent(precision_rf,0.01), "| Recall = ",percent(recall_rf,0.01)))


# 2) Logistic Regression
#=======================
model_lr <- glm(formula = Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked, data = df_model_train, family = "binomial")
summary(model_lr)
res_lr <- predict(model_lr,newdata = df_model_test, type = "response")
confusion_matrix_lr <- table(actual_value = df_model_test$Survived, predicted_value = res_lr > 0.4)
confusion_matrix_lr

# Model prediction stats
TN_lr <- confusion_matrix_lr[1,1]
FN_lr <- confusion_matrix_lr[1,2]
FP_lr <- confusion_matrix_lr[2,1]
TP_lr <- confusion_matrix_lr[2,2]

accuracy_lr <- (TN_lr+TP_lr) / (TN_lr+FN_lr+FP_lr+TP_lr)
precision_lr <- TP_lr / (TP_lr+FP_lr)
recall_lr <- TP_lr / (TP_lr+FN_lr)
print(paste("Accuracy = ",percent(accuracy_lr,0.01),"| Precision = ",percent(precision_lr,0.01), "| Recall = ",percent(recall_lr,0.01)))

# Both models are giving roughly same results


#===============
# Step 5 Predict
#===============

#Applying both models to predict

# 1) Random forest
res_rf_final <- predict(model_rf, newdata = df_test)
df_submission_rf <- df_test[, "PassengerId"]
df_submission_rf$Survived <- if_else(res_rf_final<0.5,0,1)
#write_csv(df_submission_rf, file = "~/R/Projects/titanic-competition/submission_rf")

# 2) Logistic Regression
res_lr_final <- predict(model_lr, newdata = df_test, type = "response")
df_submission_lr <- df_test[, "PassengerId"]
df_submission_lr$Survived <- if_else(res_lr_final<0.5,0,1)
#write_csv(df_submission_rf, file = "~/R/Projects/titanic-competition/submission_lr")