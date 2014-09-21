if (basename(getwd()) != "writeup")
  setwd("./writeup/")

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
dim(training)
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
sort(names(with.too.many.nas))

#percentage of almost empty values
(length(with.too.many.nas) / dd[2])*100

# now we remove the columns out of the datasets
training.neat <- training[, -which(names(training) %in% names(with.too.many.nas))]
testing.neat <- testing[, -which(names(testing) %in% names(with.too.many.nas))]
summary(training.neat)
dim(training.neat)

training.classe <- training.neat$classe

non.numeric.cols <- Filter(function(x) !is.numeric(training.neat[, x]), names(training.neat))
# remove all non numeric cols
training.neat <- training.neat[, -(which(names(training.neat) %in% non.numeric.cols))]
training.neat <- data.frame(classe=training.classe, training.neat)

# remove unecessary variables
testing.neat <- testing.neat[, -(which(names(testing.neat) %in% non.numeric.cols))]

# we'll use the generic interface caret to run our model
library(caret)
# let's split into a training set and validation set, even though cross validation will be used -- so we can check easily the OOB
inTrain <- createDataPartition(training.neat$classe, p=0.70, list=FALSE)
training.set <- training.neat[inTrain,]
validation.set <- training.neat[-inTrain,]

# let's run a RF on the remaning vars in the training set
# first let's try to use cores efficiently...
library(doMC)
registerDoMC(cores = 8)

if (file.exists("rf.fit.cross.validation_wo_X.RData")) {
  load(file="rf.fit.cross.validation_wo_X.RData")
} else {
  # train a random forest on the cleaned training set with 10 folds
  rf.fit.cross.validation <- train(
    training.set$classe ~ .,
    data=training.set, 
    method="rf",
    trControl=trainControl(method = "cv", number = 10)
  )
  save(rf.fit.cross.validation,file="rf.fit.cross.validation_wo_X.RData")  
}

# look at several outputs/characteristics of the model
rf.fit.cross.validation$results
rf.fit.cross.validation$bestTune
print(rf.fit.cross.validation$finalModel)

# head to the result for the cross valication example
confusionMatrix(predict(rf.fit.cross.validation, newdata=validation.set), validation.set$classe)

# see the variable importance
plot(varImp(rf.fit.cross.validation))

# interesting cases: the two first variables (like it was with X must less importantly)
#  we can check that it's indeed the same problem than with X (relation with the user and index)
plot(training$raw_timestamp_part_1, col=training$user_name)
plot(training$raw_timestamp_part_2, col=training$user_name)
plot(training$num_window, col=training$user_name)


test_prediction<-predict(rf.fit.cross.validation, newdata=testing.neat)
# show the prediction for the non classified data
test_prediction

if (!file.exists("./problem_id_1.txt")) {
  # write the submissions in different files
  pml_write_files = function(x){
    n = length(x)
    for(i in 1:n){
      filename = paste0("problem_id_",i,".txt")
      write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
    }
  }
  pml_write_files(test_prediction)  
}