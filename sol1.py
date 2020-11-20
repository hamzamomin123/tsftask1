# Commented out IPython magic to ensure Python compatibility.
# Importing all libraries required in this notebook
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  
# %matplotlib inline

# Reading data from remote link
url = "http://bit.ly/w-data"
s_data = pd.read_csv(url)
print("Data imported successfully")

s_data.head(10)

"""Let's plot our data points on 2-D graph to eyeball our dataset and see if we can manually find any relationship between the data. We can create the plot with the following script:"""

# Plotting the distribution of scores
s_data.plot(x='Hours', y='Scores', style='o')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()

"""**From the graph above, we can clearly see that there is a positive linear relation between the number of hours studied and percentage of score.**

### **Preparing the data**

The next step is to divide the data into "attributes" (inputs) and "labels" (outputs).
"""

X = s_data.iloc[:, :-1].values  
y = s_data.iloc[:, 1].values  

#print(X)
#print(y)

"""Now that we have our attributes and labels, the next step is to split this data into training and test sets. We'll do this by using Scikit-Learn's built-in train_test_split() method:"""

from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                            test_size=0.25, random_state=0)

"""### **Training the Algorithm**
We have split our data into training and testing sets, and now is finally the time to train our algorithm.
"""

from sklearn.linear_model import LogisticRegression 

logisticRegr = LogisticRegression()
logisticRegr.fit(X_train, y_train)

print("Training complete.")

# Plotting for the test data
plt.scatter(X, y)
#plt.plot(X, line);
plt.show()

"""### **Making Predictions**
Now that we have trained our algorithm, it's time to make some predictions.
"""

print(X_test) # Testing data - In Hours
y_pred = logisticRegr.predict(X_test) # Predicting the scores

# Comparing Actual vs Predicted
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df

# You can also test with your own data
hours = np.array(9.5).reshape(1,-1)
own_pred = logisticRegr.predict(hours)
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(own_pred[0]))

"""### **Evaluating the model**

The final step is to evaluate the performance of algorithm. This step is particularly important to compare how well different algorithms perform on a particular dataset. For simplicity here, we have chosen the mean square error. There are many such metrics.
"""

from sklearn import metrics  
print('Mean Absolute Error:', 
      metrics.mean_absolute_error(y_test, y_pred))