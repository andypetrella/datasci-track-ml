dl <- function (u, ...) {
  f <- basename(u)
  if (!file.exists(f))
    download.file(u, destfile=f, method="curl", ...)
  f    
}

training <- read.csv(dl("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"))
testing <- read.csv(dl("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"))

# look at them
summary(training)


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


# I could also try to mimic the exact analysis that the paper mentions, AdaBoost with C4.5!




