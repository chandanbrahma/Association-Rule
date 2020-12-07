# -*- coding: utf-8 -*-

import pandas as pd  

##importing dataset
movies= pd.read_csv('E:\\assignment\\assocation\\my_movies.csv')
movies.columns


from mlxtend.frequent_patterns import association_rules,apriori

##applying apriori
frequent_movies = apriori (movies, min_support=0.09, use_colnames=True, max_len=None)
frequent_movies.head


##applying association rule
rule1= association_rules(frequent_movies, metric='confidence', min_threshold=0.8)
rule2= association_rules(frequent_movies, metric='support', min_threshold=0.1)
rule3= association_rules(frequent_movies, metric='lift', min_threshold=3)


import matplotlib.pyplot as plt

##support vs confidence

plt.scatter(rule2['support'], rule2['confidence'], alpha=1)
plt.xlabel('support')
plt.ylabel('confidence')
plt.title('Support vs Confidence')
plt.show()

##support vs lift
plt.scatter(rule2['support'], rule2['lift'], alpha=1)
plt.xlabel('support')
plt.ylabel('lift')
plt.title('Support vs lift')
plt.show()

##lift vs confidence
plt.scatter(rule2['lift'], rule2['confidence'], alpha=0.8)
plt.xlabel('lift')
plt.ylabel('confidence')
plt.title('lift vs Confidence')
plt.show()
