##### Association rules. Using apriori packages for association rules algorithm
import pandas as pd
from apyori import apriori ### Useful for some of the operations
import matplotlib.pyplot as plt
import numpy as np

#### Import SAS Database with PD
#database = pd.read_sas("nameofdb") ### This would be what I would do with the db from the Prof


###Since the DB i was using before recieving the homework was csv (for testing)
data_test = pd.read_csv("store_data.csv", header = None)

#print(data_test.head())   ### Useful to inspect the data


###If i wanted to open it without PANDAS. (Not used in this)
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


### With this apriori package, we can extract the support of the different items
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori

te = TransactionEncoder()
te_data = te.fit(linesOfData2).transform(linesOfData2)
te_df = pd.DataFrame(te_data, columns=te.columns_)
support = apriori(te_df, min_support = 0.01, use_colnames = True)

#print(support)  #This prints out all of the supports we calculated.

#Prints out the top supports
resultsupport = support.sort_values(by=["support"], ascending = False)
print(resultsupport.head())

## ========= Try to get the confidence with this one
from mlxtend.frequent_patterns import association_rules
confidence_support = association_rules(support, metric="confidence", min_threshold=0.01)
#print(confidence_support)

#Prints out the top confidence
resultsconfidence_support = confidence_support.sort_values(by=["confidence"], ascending=False)
print(resultsconfidence_support.head())



###Using the other apriori package. We can set the target support, confidence and lift in the same command.

from apyori import apriori

regleAssociation = apriori(linesOfData2, min_support = 0.008, min_confidence = 0.2, min_lift = 2, min_len = 3)
resultatsAssociation = list(regleAssociation)
#print(resultatsAssociation[0])
#print(len(resultatsAssociation))

## This prints out the results of the above rule
#for item in resultatsAssociation:

#    pair = item[0]
#    items = [x for x in pair]
#    try :
#        print("Rule: " + items[0] + " -> " + items[1] + " + " + items[2])

#    except:
#        print("Rule: " +items[0] + " -> " + items[1] )

    ##second index of the inner list
#    print("Support: " + str(item[1]))

    ##third index of the list located at 0th
    ##of the third index of the inner list

#    print("Confidence: " + str(item[2][0][2]))
#    print("Lift: " + str(item[2][0][3]))
#    print("=====================================")


############## ============ APRIORI FINISH ==============================
