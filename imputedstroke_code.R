#import both data sets.
df1 <- read.csv("~/M445_HW/project/imputed_stroke.csv") #values for BMI and smoking status were imputed
df2 <- read.csv("~/M445_HW/project/stroke_miss.csv") #nas are present in the data
df2 = subset(df2, select = -c(id))

library(MASS)
library(mice)
library(caret)
library(tidyverse)
library(data.table)
library(randomForest)

#testing out mice. holds no bearing on final analysis.
df2$ss_ordinal <- as.factor(df2$ss_ordinal)
set.seed(505)
imp1 = mice(df2, m = 11, method = c('logreg','','','','','','','','pmm','','lda'), maxit=75)
fitm1 <- with(imp1, glm(stroke~gender+age+hypertension+heart_disease+
                          ever_married+work_type+Residence_type+avg_glucose_level+bmi+
                          factor(ss_ordinal), family = binomial()))

#in this case, 0 is never smoked; 1 is used to smoke; 2 is smokes.
summary(pool(fitm1))

mod1 = glm(stroke~., data = df1, family = binomial(link='probit'))
summary(mod1)

# 90% of the sample size
smp_size <- floor(0.9 * nrow(df1))

# set the seed to make your partition reproducible
set.seed(505)
train_ind <- sample(seq_len(nrow(df1)), size = smp_size)

train <- df1[train_ind, ]
test <- df1[-train_ind, ]

#using ALL predictors to create a model
mod2 = glm(stroke~., data = train, family = binomial(link = 'probit'))
summary(mod2)

predvals = predict(mod2, newdata = test, type = 'response')
plot(test$stroke,predvals)

bb = lm(predvals~test$stroke)
abline(bb)

binary.pred = ifelse(predvals > 0.16, 1, 0)
confusionMatrix(factor(binary.pred), factor(test$stroke))

#getting rid of the smoking status to make predictions.
mod3 = update(mod2, .~. -smoking_status)
summary(mod3)

predvals = predict(mod3, newdata = test, type = 'response')
plot(test$stroke,predvals)

bb = lm(predvals~test$stroke)
abline(bb)

binary.pred = ifelse(predvals > 0.17, 1, 0)
confusionMatrix(factor(binary.pred), factor(test$stroke))

#dropping gender, bmi, residence, and marital status to make predictions.
mod4 = update(mod2, .~. -Residence_type -gender -bmi -ever_married)
summary(mod4)

predvals = predict(mod4, newdata = test, type = 'response')
plot(test$stroke,predvals)

bb = lm(predvals~test$stroke)
abline(bb)

binary.pred = ifelse(predvals > 0.17, 1, 0)
confusionMatrix(factor(binary.pred), factor(test$stroke))
#note that this has predictions that are just as good as the one with all the predictors.


#time to use other methods of classification.

#random forest
rfset_train = subset(train, select = -c(gender, work_type, ever_married, Residence_type, bmi))
rfset_test = subset(test, select = -c(gender, work_type, ever_married, Residence_type, bmi))
rf_classifier = randomForest(factor(stroke) ~ ., data=rfset_train, ntree=200, mtry=2, importance=TRUE)
test_set_predictors = subset(rfset_test, select = -c(stroke))
prediction_for_table <- predict(rf_classifier, test_set_predictors) 
confusionMatrix(as.factor(prediction_for_table), as.factor(rfset_test$stroke))

#qda 
qda.mod = qda(stroke~age+avg_glucose_level+heart_disease+smoking_status, data = train)
qda.pred <- predict(qda.mod, test)
confusionMatrix(as.factor(qda.pred$class), as.factor(test$stroke))

#lda
lda.mod = lda((stroke~age+avg_glucose_level+heart_disease+smoking_status), data = train)
lda.pred <- lda.mod %>% predict(test)
confusionMatrix(as.factor(lda.pred$class), as.factor(test$stroke))

#create the best logistic model, starting from all the data.
summary(mod1)
bestmod1 = update(mod1, .~. -gender -Residence_type -ever_married -bmi)
summary(bestmod1)

testmod1 = update(bestmod1, .~. -work_type)
summary(testmod1)
anova(testmod1, bestmod1, test = 'LRT')

testmod2 = update(bestmod1, .~. -smoking_status)
summary(testmod2)
anova(testmod2, bestmod1, test = 'LRT')

#here is the overall best model!
summary(bestmod1)