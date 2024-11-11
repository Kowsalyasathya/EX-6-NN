<H3>NAME: KOWSALYA M</H3>
<H3>REGISTER NO: 212222230069</H3>
<H3>EX. NO.6</H3>
<H3>DATE:</H3>
<H1 ALIGN =CENTER>Heart attack prediction using MLP</H1>
<H3>Aim:</H3>  To construct a  Multi-Layer Perceptron to predict heart attack using Python

<H3>Algorithm:</H3>
Step 1:Import the required libraries: numpy, pandas, MLPClassifier, train_test_split, StandardScaler, accuracy_score, and matplotlib.pyplot.<BR>

Step 2:Load the heart disease dataset from a file using pd.read_csv().<BR>

Step 3:Separate the features and labels from the dataset using data.iloc values for features (X) and data.iloc[:, -1].values for labels (y).<BR>

Step 4:Split the dataset into training and testing sets using train_test_split().<BR>

Step 5:Normalize the feature data using StandardScaler() to scale the features to have zero mean and unit variance.<BR>

Step 6:Create an MLPClassifier model with desired architecture and hyperparameters, such as hidden_layer_sizes, max_iter, and random_state.<BR>

Step 7:Train the MLP model on the training data using mlp.fit(X_train, y_train). The model adjusts its weights and biases iteratively to minimize the training loss.<BR>

Step 8:Make predictions on the testing set using mlp.predict(X_test).<BR>

Step 9:Evaluate the model's accuracy by comparing the predicted labels (y_pred) with the actual labels (y_test) using accuracy_score().<BR>

Step 10:Print the accuracy of the model.<BR>

Step 11:Plot the error convergence during training using plt.plot() and plt.show().<BR>

## Program: 

```
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

data=pd.read_csv('https://raw.githubusercontent.com/Lavanyajoyce/EX-6-NN/main/heart.csv')

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

mlp = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=1000, random_state=42)
training_loss = mlp.fit(X_train, y_train).loss_curve_

y_pred = mlp.predict(X_test)

plt.plot(training_loss)
plt.title("MLP Training Loss Convergence")
plt.xlabel("Iteration")
plt.ylabel("Training Loss")
plt.show()


conf_matrix=confusion_matrix(y_test,y_pred)
classification_rep=classification_report(y_test,y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(classification_rep)

```
## Output:

![image](https://github.com/user-attachments/assets/6506129f-e034-41fd-b677-e934f58f87e2)

![image](https://github.com/user-attachments/assets/ea9090f5-64bd-4b5b-81c6-94f01ca73fb5)


## Results:
Thus, an ANN with MLP is constructed and trained to predict the heart attack using python.
