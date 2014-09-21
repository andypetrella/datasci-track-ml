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
sort(names(with.too.many.nas))

#percentage of almost empty values
(length(with.too.many.nas) / dd[2])*100

# now we remove the columns out of the datasets
with.too.many.nas.indexes <- which(names(training) %in% names(with.too.many.nas))

training.neat <- training[, -which(names(training) %in% names(with.too.many.nas))]
testing.neat <- testing[, -which(names(testing) %in% names(with.too.many.nas))]
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

test_prediction<-predict(rf.fit.cross.validation, newdata=testing.neat)
test_prediction
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}
pml_write_files(test_prediction)

