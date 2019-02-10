##### Association rules. Using apriori packages for association rules algorithm
import pandas as pd
from apyori import apriori ### Useful for some of the operations
import matplotlib.pyplot as plt
import numpy as np

#### ===== This is where you imput the values you want to test for

support_threshold = 0.015 # You might want to modify these depending on what you are working on
confidence_threshold = 0.01 #Same as above
value_arranged_by = "lift"  # "confidence" or "lift" or others


###Since the DB i was using before recieving the homework was csv (for testing)
data_test = pd.read_csv("store_data.csv", header = None) ## This DB was found online, I have not yet recieved the homework database

#print(data_test.head())


###If i wanted to open it without PANDAS. (Not used in this)
import csv
data_test_1 = open("store_data.csv")

#print(len(data_test)) ### Information used to do the next step

### Converting PANDAS DATAFRAME into List of lists
linesOfData = list()
for i in range(0, 7501):
    linesOfData.append([str(data_test.values[i,j]) for j in range(0,20)])

### Removing the "nan" strings that are in the data
linesOfData2 = list()
for line in linesOfData:
    templine = list()
    for item in line:
        if item != 'nan':
            templine.append(item)
        else:continue
    linesOfData2.append(templine)

#print(linesOfData[0]) ### Verifying that we removed the "nan" - chane the 0 to other numbers to verify


### With this apriori/mlxtend package, we can extract the support of the different items
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori

#### Apriori Algorithm - (Agrawal et Srikantt 1994)
#### STEP 1 - Fix minimal support and get a list for all mixes of items
te = TransactionEncoder()
te_data = te.fit(linesOfData2).transform(linesOfData2)
te_df = pd.DataFrame(te_data, columns=te.columns_)
support = apriori(te_df, min_support = support_threshold, use_colnames = True)

#print(support)  #This prints out all of the supports we calculated.

#Prints out the top supports
resultsupport = support.sort_values(by=["support"], ascending = False) ### WE have the top support at the begining, but they are for single items
#print(resultsupport.head())

###Adding a lenght value in the result Pandas Database. This is useful as we will take the values above 1.
resultsupport['length'] = resultsupport['itemsets'].apply(lambda x: len(x))

### Now we can take the itemsets (2 or more together) that have high support enough
resultsupportmin2 = resultsupport[(resultsupport["length"]>=2)]
#print(resultsupportmin2)  ### This prints out the correct list of items


### I don't see the whole data, so we have to reformat PANDAS
### =====This code I found online and did not write it, it has the perk of making print_full(pandasDF) show everything
def print_full(x):
    pd.set_option('display.max_rows', len(x))
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 2000)
    pd.set_option('display.float_format', '{:20,.2f}'.format)
    pd.set_option('display.max_colwidth', -1)
    print(x)
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')
    pd.reset_option('display.width')
    pd.reset_option('display.float_format')
    pd.reset_option('display.max_colwidth')
###====


## ========= Try to get the confidence with this one
from mlxtend.frequent_patterns import association_rules
confidence_support = association_rules(support, metric="confidence", min_threshold=confidence_threshold)
#print_full(confidence_support)

#Prints out the top confidence
resultsconfidence_support = confidence_support.sort_values(by=[value_arranged_by], ascending=False)
print_full(resultsconfidence_support.head())

### Export the data out to an excel file to better visualise and undersdand the data
resultsconfidence_support.to_excel("name_of_exit_file.xlsx")
