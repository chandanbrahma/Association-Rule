# -*- coding: utf-8 -*-

import pandas as pd 

##importing dataset
book= pd.read_csv('E:\\submitted assignment\\assocation\\book.csv')
book.columns

from mlxtend.frequent_patterns import association_rules,apriori

##applying apriori
frequent_books = apriori (book, min_support=0.02, use_colnames=True, max_len=None)
frequent_books.head()


##applying association rule
rule1= association_rules(frequent_books, metric='confidence', min_threshold=0.8)
rule2= association_rules(frequent_books, metric='support', min_threshold=0.09)
rule3= association_rules(frequent_books, metric='lift', min_threshold=6)



import matplotlib.pyplot as plt

##support vs confidence
plt.scatter(rule1['support'], rule1['confidence'], alpha=0.8)
plt.xlabel('support')
plt.ylabel('confidence')
plt.title('Support vs Confidence')
plt.show()


plt.scatter(rule2['support'], rule2['confidence'], alpha=0.8)
plt.xlabel('support')
plt.ylabel('confidence')
plt.title('Support vs Confidence')
plt.show()


plt.scatter(rule3['support'], rule3['confidence'], alpha=0.8)
plt.xlabel('support')
plt.ylabel('confidence')
plt.title('Support vs Confidence')
plt.show()


##support vs lift
plt.scatter(rule1['support'], rule1['lift'], alpha=0.8)
plt.xlabel('support')
plt.ylabel('lift')
plt.title('Support vs lift')
plt.show()

##lift vs confidence
plt.scatter(rule1['lift'], rule1['confidence'], alpha=0.8)
plt.xlabel('lift')
plt.ylabel('confidence')
plt.title('lift vs Confidence')
plt.show()
