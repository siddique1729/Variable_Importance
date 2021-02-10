# The code is written by Jawwad Shadman Siddique | R11684947
# Date of Submission: 11 / 25  / 2020
# This Step performs Data Cleaning 
# It converts the No3 to log10No3
# It cleans the non-standard missing values by converting it into NA's
# It removes the outlier data by inferencing from the Boxplot
# Total Raw Data initial = 957
# Total Data after cleaning = 929

# Checking the directory
# getwd()
# setwd("D:/R Practice Programs")
# getwd()

# reading the culvert data
a = read.csv('OgallalaNewData1.csv')

# converting to log10No3

log10NO3 = a$NO3
log10NO3 = log10(log10NO3)
summary(log10NO3)
print(log10NO3)

# Removing the non-standard and Inf values
anew = data.frame(a, log10NO3)
anew = subset(anew, log10NO3 != -Inf)
summary(anew)
colnames(anew)
anew = anew[, -33]


# checking the summary of the new data and output column
summary(anew$log10NO3)
summary(anew)

# checking the total 'na' values
sum(is.na(anew))

# checking the outlier data
boxplot(anew$KSAT)
boxplot(anew$F)
boxplot(anew$WellDepth)
boxplot(anew$Mg)
boxplot(anew$Na)
boxplot(anew$SARGW)
boxplot(anew$AqDepth)
boxplot(anew$DWT)
boxplot(anew$Cl)
boxplot(anew$K)
boxplot(anew$As)
boxplot(anew$TDS)
boxplot(anew$HCO3)
boxplot(anew$Aqbott)

# Removing the outlier data by subsetting and checking with the boxplot
# hello - variable for the cleaned dataset

hello = subset(anew, KSAT <= 100)
hello = subset(hello, F <= 20)

hello = subset(hello, WellDepth != 812)
hello = subset(hello, WellDepth != 778)
hello = subset(hello, WellDepth != 760)
hello = subset(hello, WellDepth != 721)

hello = subset(hello, Mg <= 300)
hello = subset(hello, Na <= 600)
hello = subset(hello, SARGW <= 8)
hello = subset(hello, SARGW <= 8)

hello = subset(hello, AqDepth != 778)

hello = subset(hello, K <= 662)
hello = subset(hello, K != 283)

hello = subset(hello, As <= 148)
hello = subset(hello, As <= 58)

hello = subset(hello, As <= 148)
hello = subset(hello, As <= 58)

hello = subset(hello, B <= 2660)
hello = subset(hello, B <= 2000)


hello = subset(hello, Aqbott != 866)


# checking the cleaned dataset

summary(hello)
boxplot(hello)

# writing the cleaned dataset to the csv file

# write.csv(hello, 'ogalanew1_clean.csv')