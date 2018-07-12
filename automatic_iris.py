#!/usr/bin/python2

from  sklearn.datasets  import  load_iris
from  sklearn.model_selection    import train_test_split
from  sklearn  import tree
from   sklearn.metrics  import  accuracy_score 
import  matplotlib.pyplot  as plt
import  mpld3
# loading  dataset
iris=load_iris()
print dir(iris)

#  checking number of rows in  IRIS feature data  
print  iris.data.shape
#  checking complete data 
#print  iris.data

#  label data
#print  iris.target
#  spliting  train and test data in  10%
train_data,test_data,train_target,test_target=train_test_split(iris.data,iris.target,test_size=0.1)

#  calling  decision tree classifier 
clf=tree.DecisionTreeClassifier()
trained=clf.fit(train_data,train_target)


#  now predicting  test_data  output
output=trained.predict(test_data)

print  output

#  calculating  accuracy score  
pct=accuracy_score(test_target,output)
print  type(pct)
x=["showing pct"]


plt.bar(x,pct)
mpld3.show()
















