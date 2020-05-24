'''The dataset contains a walkatime data of past students and how they performed in
the evaluation exam. The task is to predict the score you will get given the amount of time you
spend on coding daily.
Input : You are given one feature corresponding to time noted by walkatime.
Output : A scalar denoting the level of performance student achieved by devoting the given time.
TASK : Build a Linear Regression model on the dataset.'''

# Importing Libraries
import matplotlib.pyplot as plt
import pandas as pd 
from sklearn.linear_model import LinearRegression
def main():
    # Importing Data
    x_train = pd.read_csv('Linear_X_Train.csv')
    y_train = pd.read_csv('Linear_Y_Train.csv')
    x_test = pd.read_csv('Linear_X_Test.csv')

    # Training the model
    regressor = LinearRegression()
    regressor.fit(x_train, y_train)

    # Predicting the test results
    y_pred = regressor.predict(x_test)
    with open ("Predic.csv",'w') as file:
        for i in y_pred:
            file.write(str(i)+'\n')
    # Ploting the Results
    plt.scatter(x_train, y_train, color = 'c', label = 'Train-Set Observations')
    plt.scatter(x_test, y_pred, color = 'b',label = 'Test Predictions')
    plt.plot(x_train, regressor.predict(x_train), color = 'r', label = 'Regression Line')
    plt.title('Performance vs Time Devoted (Test set)')
    plt.xlabel('Time Devoted')
    plt.ylabel('Performance')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()