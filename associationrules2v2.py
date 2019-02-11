import pandas as pd

### This is the package I use for the Apriori algorithm. I verified the results and the math seems good
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori


### Here you put the threshold you want to see. To choose the confidence, 
support_threshold = 0.015
confidence_threshold = 0.01  ## you can set the support first and see your data to choose the right confidence by removing the "#" on line 28
value_arranged_by = "lift"  # "confidence" or "lift" strings work well here


database2 = pd.read_sas("data_du_menager.sas7bdat") 
#print(database2)

dt = database2.groupby(["no_transaction"])["item"].apply(list) ## Group the items in transactions
#print(dt)


#### Apriori Algorithm - (Agrawal et Srikantt 1994)
#### STEP 1 - Fix minimal support and get a list for all combinations of items
te = TransactionEncoder()
te_data = te.fit(dt).transform(dt)
te_df = pd.DataFrame(te_data, columns=te.columns_)
support = apriori(te_df, min_support = support_threshold, use_colnames = True)

#print(support)  

#Prints out the top supports
resultsupport = support.sort_values(by=["support"], ascending = False) ### WE have the top support at the begining, but they are for single items
#print(resultsupport.head())

###Adding a lenght value in the result Pandas Database. This is useful as we will take the values above 1.
resultsupport['length'] = resultsupport['itemsets'].apply(lambda x: len(x))

### Now we can take the itemsets (2 or more together) that have high support enough
resultsupportmin2 = resultsupport[(resultsupport["length"]>=2)]
#print(resultsupportmin2)  


### I don't see the whole data, this script helps print out the full dataset
### =====This code I found online and did not write it. 
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


## his sets our confidence threshold ( we have set the confidence_threshold at the start of the script)
from mlxtend.frequent_patterns import association_rules
confidence_support = association_rules(support, metric="confidence", min_threshold=confidence_threshold)
#print_full(confidence_support)

#This prints out the results by the top value arranged how we have set it.
resultsconfidence_support = confidence_support.sort_values(by=[value_arranged_by], ascending=False)
print_full(resultsconfidence_support.head(10))

## This sends the results to the excel file for my homework
resultsconfidence_support.to_excel("travail_en_classe.xlsx")
