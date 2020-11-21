⁷


##importing the dataset 
with open("E:\\python\\association rule\\groceries.csv","r") as f:
    groceries = f.read()
   
## spliting the data
groceries= groceries.split("\n")

## creating a empty list and inserting the items into them
groceries_list = []

for i in groceries:
    groceries_list.append(i.split(","))
    
    
import pandas as pd 
## converting into a series 
groceries_series  = pd.DataFrame(pd.Series(groceries_list))
 
groceries_series.columns = ["transactions"]
    
## converting the dataset into dummy variables
X = groceries_series['transactions'].str.join(sep='*').str.get_dummies(sep='*')

from mlxtend.frequent_patterns import association_rules,apriori

##applying apriori
frequent_items = apriori (X, min_support=0.02, use_colnames=True, max_len=None)
frequent_items.head()
frequent_items.shape
    

##applying association rule
rule1= association_rules(frequent_items, metric='confidence', min_threshold=0.07)
rule2= association_rules(frequent_items, metric='support',min_threshold=0.02)
rule3= association_rules(frequent_items, metric='lift', min_threshold=0.8)

## so can see the antecedents and the respective concequents with respect to different rules

## now let us visualize the different results

import matplotlib.pyplot as plt

##support vs confidence
plt.scatter(rule1['support'], rule1['confidence'])
plt.xlabel('support')
plt.ylabel('confidence')
plt.title('Support vs Confidence')
plt.show()
    

plt.scatter(rule2['support'], rule2['confidence'])
plt.xlabel('support')
plt.ylabel('confidence')
plt.title('Support vs Confidence')
plt.show()


plt.scatter(rule3['support'], rule3['confidence'])
plt.xlabel('support')
plt.ylabel('confidence')
plt.title('Support vs Confidence')
plt.show()

##support vs lift
plt.scatter(rule1['support'], rule1['lift'])
plt.xlabel('support')
plt.ylabel('lift')
plt.title('Support vs lift')
plt.show()

##lift vs confidence
plt.scatter(rule1['lift'], rule1['confidence'])
plt.xlabel('lift')
plt.ylabel('confidence')
plt.title('lift vs Confidence')
plt.show()
    

## elimination of redundancy in the rules
def to_list(i):
    return (sorted(list(i)))


ma_X = rule1.antecedents.apply(to_list)+rule1.consequents.apply(to_list)


ma_X = ma_X.apply(sorted)

rules_sets = list(ma_X)

unique_rules_sets = [list(m) for m in set(tuple(i) for i in rules_sets)]
index_rules = []
for i in unique_rules_sets:
    index_rules.append(rules_sets.index(i))


##getting rules without any redudancy 
rules_no_redudancy  = rule1.iloc[index_rules,:]

rules_no_redudancy.head()

