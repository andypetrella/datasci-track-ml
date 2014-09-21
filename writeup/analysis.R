dl <- function (u, ...) {
  f <- basename(u)
  if (!file.exists(f))
    download.file(u, destfile=f, method="curl", ...)
  f    
}

set.seed(12345)

training.url <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testing.url <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
training <- read.csv(dl(training.url))
testing <- read.csv(dl(testing.url))

# look at them
summary(training)

# X is just an index, let's remove it first
if (names(training)[1] != "X")
  warning("Apparently `X` is no more the first variable in the df")
training <- training[, -1]


# lot's of empty strings, NA, #DIV/0!, (Other)
dd <- dim(training)
Filter(function(x) x > 0, sapply(training, function(col) sum(col == "")) / dd[1])
Filter(function(x) x > 0, sapply(training, function(col) sum(col == "#DIV/0!")) / dd[1])

# re read with "", "NA", and "#DIV/0!" as NA's
na.strings <- c("", "NA", "#DIV/0!")
training <- read.csv(dl(training.url), na.strings=na.strings)
testing <- read.csv(dl(testing.url), na.strings=na.strings)

# "X" is simply the index of the rows so we can get rid of it
training <- training[, -1]

with.nas <- sort(Filter(function(x) x > 0, sapply(training, function(x)sum(is.na(x))) / dd[1]))

with.too.many.nas <- Filter(function(x) x > 0.95, with.nas)

#percentage of almost empty values
(length(with.too.many.nas) / dd[2])*100

# now we remove the columns out of the datasets
with.too.many.nas.indexes <- which(names(training) %in% names(with.too.many.nas))

training.neat <- training[, -with.too.many.nas.indexes]
testing.neat <- testing[, -with.too.many.nas.indexes]
summary(training.neat)
dim(training.neat)

training.classe <- training.neat$classe

non.numeric.cols <- Filter(function(x) !is.numeric(training.neat[, x]), names(training.neat))
training.neat <- training.neat[, -(which(names(training.neat) %in% non.numeric.cols))]
training.neat <- data.frame(classe=training.classe, training.neat)

testing.neat <- testing.neat[, -(which(names(testing.neat) %in% non.numeric.cols))]

library(caret)
inTrain <- createDataPartition(training.neat$classe, p=0.70, list=FALSE)
training.set <- training.neat[inTrain,]
validation.set <- training.neat[-inTrain,]

# let's run a RF on the remaning vars in the training set
# first let's try to use cores efficiently...
library(doMC)
registerDoMC(cores = 8)

rf.fit.cross.validation <- train(
                            training.set$classe ~ .,
                            data=training.set, 
                            method="rf",
                            trControl=trainControl(method = "cv", number = 10)
                          )
save(rf.fit.cross.validation,file="rf.fit.cross.validation_wo_X.RData")

rf.fit.cross.validation$results
rf.fit.cross.validation$bestTune
rf.fit.cross.validation$finalModel

confusionMatrix(predict(rf.fit.cross.validation, newdata=validation.set), validation.set$classe)

plot(varImp(rf.fit.cross.validation))




# let's try to get some sense of the data by exploring it a bit...
# possible analysis are
# kmeans => based on the number of classes? Or kind of testers
# pca => number of significant v
# some metrics of the data
# see if groups have distinct means, perform t-tests
# see if freq groups are different using Chi-sq ?

# the paper is referring a specific use of adaBoost
# it sounds good to use ada boost since if focuses on outliers and hard cases
# why it sounds good is because the tests are only slightly different, so we can expect a lot of 
# similarities in the observation

# Since we're doing a classification, the real AdaBoost is the most approrpriate (I almost know) which
# use tree to output probability to be of a certain class.
library(caret)
library(ada)
library(rpart)
adaboostModel <- train(y=training$classe, x=training[,-160], method="ada", type="real", data=training, iter=10, nu=0.1, maxdepth=10)

adaboostModel <- train(y=training$classe, x=training[,-160], method="ada")

# I could also try to mimic the exact analysis that the paper mentions, AdaBoost with C4.5!



install.packages("adabag")
library(adabag)
library(rpart)
v <- data.frame(classe = c("a", "b", "a"), d=c(1, 2, 1))
b <- boosting(classe ~ . , data=v, boos=TRUE, mfinal=1)


## rpart library should be loaded
data(iris)
iris.adaboost <- boosting(Species~., data=iris, boos=TRUE, mfinal=10)

sub <- c(sample(1:50, 25), sample(51:100, 25), sample(101:150, 25))
iris.bagging <- bagging(Species ~ ., data=iris[sub,], mfinal=10)
iris.predbagging<- predict.bagging(iris.bagging, newdata=iris[-sub,])
