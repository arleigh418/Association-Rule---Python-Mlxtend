
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import re 
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import jieba


x = pd.read_csv('D:/ml-latest-small/tags.csv')
x= x.dropna()
movieid = x['movieId'].tolist()
comment = x['tag'].tolist()

combine = list(zip(movieid,comment))


all_comment = []
store = []
# c=0
for k in range(len(combine)):
    try:
        if movieid[k] == movieid[k+1]:
            store.append(comment[k])
            # c=c+1
            continue
        elif movieid[k] != movieid[k+1] and movieid[k] == movieid[k-1]:
            store.append(comment[k])
            all_comment.append(store)
            store=[]
        elif movieid[k] != movieid[k-1] and movieid[k] != movieid[k+1]:
            store.append(comment[k])
            all_comment.append(store)
            store = []
    except:
        break
        


print(all_comment)





te = TransactionEncoder()	
df_tf = te.fit(all_comment).transform(all_comment)
# print(df_tf)
# df_01 = df_tf.astype('int')			
# df_name = te.inverse_transform(df_tf)		
df = pd.DataFrame(df_tf,columns=te.columns_)
# print(df)



frequent_itemsets = apriori(df,min_support=0.005,use_colnames=True)	
print(frequent_itemsets)
frequent_itemsets.sort_values(by='support',ascending=False,inplace=True)	

association_rule = association_rules(frequent_itemsets,metric='confidence',min_threshold=0.01)	
association_rule.sort_values(by='leverage',ascending=False,inplace=True)
print(association_rule)    


