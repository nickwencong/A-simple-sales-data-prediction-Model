import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import svm
from sklearn import neighbors
from sklearn.neighbors import NearestCentroid
from sklearn import tree
import sklearn.naive_bayes as nb
from sklearn.metrics import r2_score
from sklearn.linear_model import QuantileRegressor


#import data
df = pd.read_csv(r'Shop.csv')

#add a new value 'Delay' to the table, here we define delay as the number of months between a good was booked and shipped,
df['Delay'] = (df['Shippedd Year'] - df['Booked Year'])*12 + df['Shippedd Month'] - df['Booked Month']


def get_pivotMonth(month, category):
    start = 2017
    end = 2019
    if month <= 6 :
        start = 2018
    pivot = 0
    i = start
    while i <= end:
        df_t = df.loc[(df['Shippedd Year'] == i) & (df['Shippedd Month'] == month) & (df['Category'] == category)]
        for j in range(0, 36):
            df_pivot = df.loc[(df['Shippedd Year'] == i) & (df['Shippedd Month'] == month) & (df['Delay'] > j) & (df['Category'] == category)]
            if df_pivot.size / df_t.size < 0.1 :
                pivot = pivot + j
                break
        i += 1
    pivot = pivot / (end - start + 1)
    return round(pivot)


#print(get_pivotMonth(2, 'Bedroom'))



def delay(month, category, pivot):
    delay_tb = np.zeros((pivot+2,3))
    i = int(month)
    ct = category
    for j in range(2017, 2020):
        df_t = df.loc[(df['Shippedd Year'] == j) & (df['Shippedd Month'] == i) & (df['Category'] == ct)]
        for k in range(0, pivot + 1):
            df_0 = df.loc[(df['Shippedd Year'] == j) & (df['Shippedd Month'] == i) & (df['Delay'] == k) & (
                    df['Category'] == ct)]
            delay_tb[k][j - 2017] = df_0.size / df_t.size
        df_1 = df_4 = df.loc[(df['Shippedd Year'] == j) & (df['Shippedd Month'] == i) & (df['Delay'] > pivot) & (
                    df['Category'] == ct)]
        delay_tb[pivot+1][j - 2017] = df_1.size / df_t.size
    #print(delay_tb)
    return delay_tb

def delay_r(delay_tb, month, pivot):
    rst = np.zeros(pivot+2)
    if month > 6:
        for i in range(0, pivot+2):
            rst[i] = np.average(delay_tb[i])
    else:
        for i in range(0, pivot+2):
            rst[i] = (delay_tb[i][1] + delay_tb[i][2]) / 2
    return rst


def get_book(start, end, category):
    end = int(end)
    start = int(start)
    data = np.zeros((end-start+1, 12))
    for i in range(1, 13):
        for j in range(start, end+1):
            df_t = df.loc[(df['Booked Year'] == j) & (df['Booked Month'] == i) & (df['Category'] == category)]
            cnt = df_t.size
            data[j-start][i-1] = cnt
    return data

def get_ship(start, end, category):
    end = int(end)
    start = int(start)
    data = np.zeros((end-start+1, 12))
    for i in range(1, 13):
        for j in range(start, end+1):
            df_t = df.loc[(df['Shippedd Year'] == j) & (df['Shippedd Month'] == i) & (df['Category'] == category)]
            cnt = df_t.size
            data[j-start][i-1] = cnt
    return data

def get_pastmonth(year, month, category, pivot):
    rst = np.zeros(pivot+1)
    tb_1 = get_book(year, year, category)
    m = month
    for i in range(0, pivot+1):
        if m == 0:
            tb_1 = get_book(year-1, year-1, category)
            m = 12
        rst[i] = tb_1[0][m-1]
        m = m - 1
    return rst

#Linear Regression
def Linear(X,y,X_test):
    lr = LinearRegression()
    lr.fit(X, y)
    y_pred = lr.predict(X_test)
    return y_pred


# Support Vector Machine
def SVM(X,y,X_test):
    regr = svm.SVR()
    regr.fit(X, y)
    y_pred = regr.predict(X_test)
    return y_pred


#Nearest Centroid Classifier
def NCC(X,y,X_test):
    clf = NearestCentroid()
    clf.fit(X, y)
    return clf.predict(X_test)


#K-Nearest-Neighbour
def KNN(X,y,X_test,n):
    knn = neighbors.KNeighborsRegressor(n, weights='uniform')
    y_pred = knn.fit(X, y).predict(X_test)
    return y_pred


#Decision Tree
def Decision_Tree(X,y,X_test):
    clf = tree.DecisionTreeRegressor()
    clf = clf.fit(X, y)
    return clf.predict(X_test)


# Gaussian Naive Bayes
def GNB(X,y,X_test):
    gnb = nb.GaussianNB()
    y_pred = gnb.fit(X, y).predict(X_test)
    return y_pred

#Quantile
def Quantile(X,y,X_test):
    qr = QuantileRegressor(quantile=0.5, alpha=0, solver='highs')
    y_pred = qr.fit(X, y).predict(X_test)
    return y_pred

#Main function
def main():
    year = int(input("Enter the year you would like to check: "))
    month = int(input("Enter the month you would like to check: "))
    if(year > 2022):
        #if we want predict the data after 2022, ask the customer to input data
        book = input("Please enter your booked sales: ").split()
        book_list = list(map(int, book))
        ship = input("Please enter your shipped sales: ").split()
        ship_list = list(map(int, ship))
        b_size = len(book_list)
        book_predict = book_list[b_size-1]
        s_size = len(ship_list)
        #the number of booked sales must be shipped sales + 1, otherwise quit
        if (b_size != s_size + 1):
            print("Incorrect input format!")
            quit()
    category = input("Enter the type of product you would like to check: ")

    #get train data
    Data_train = get_book(2017, 2019, category)
    Data_s_train = get_ship(2017, 2019, category)

    #get test data
    book_2020 = get_book(2020, 2020, category)
    ship_2020 = get_ship(2020, 2020, category)

    #get check data
    Data_test = get_book(year, year, category)
    Data_s_test = get_ship(year, year, category)

    #Set X, x and y for the models
    x = np.concatenate((Data_train[0], Data_train[1], Data_train[2]), axis=0)
    y = np.concatenate((Data_s_train[0], Data_s_train[1], Data_s_train[2]), axis=0)
    x.reshape(-1, 1)
    X = np.zeros((36, 1))
    for i in range(0, 36):
        X[i][0] = x[i]
    y.reshape(-1, 1)
    X_test = np.zeros((12, 1))

    #read in sample test data
    for i in range(0, 12):
        X_test[i][0] = Data_test[0][i]
    X_2020 = np.zeros((12, 1))
    for i in range(0, 12):
        X_2020[i][0] = book_2020[0][i]
    #Option: print the scatter plot
    print("Do you want to check the scatter plot?")
    opt = int(input("1:Yes  2:No  "))
    if opt == 1:
        plt.scatter(x, y)
        plt.show()

    #choose model
    rst_linear = Linear(X,y,X_2020)
    rst_SVM = SVM(X,y,X_2020)
    rst_KNN = KNN(X,y,X_2020,5)
    rst_DT = Decision_Tree(X,y,X_2020)
    rst_GNB = GNB(X,y,X_2020)
    rst_Q = Quantile(X,y,X_2020)

    #check r2 value
    r2_linear = r2_score(ship_2020[0], rst_linear)
    r2_SVM = r2_score(ship_2020[0], rst_SVM)
    r2_KNN = r2_score(ship_2020[0], rst_KNN)
    r2_DT = r2_score(ship_2020[0], rst_DT)
    r2_GNB = r2_score(ship_2020[0], rst_GNB)
    r2_Q = r2_score(ship_2020[0], rst_Q)

    #try to find best-fit model by comparing r2 value
    max_r2 = max(r2_linear, r2_SVM, r2_KNN, r2_DT, r2_GNB, r2_Q)
    global rst
    global decision

    #Generate predictions from selected model
    if year > 2022:
        X_test = np.zeros((1, 1))
        X_test[0][0] = book_predict
    if max_r2 == r2_linear:
        print("Linear is the recommend model.")
        print("1:Continue with Linear  2:Choose other model")
        decision = int(input())
        if decision == 1:
            rst = Linear(X,y,X_test)
    elif max_r2 == r2_SVM:
        print("SVM is the recommend model.")
        print("1:Continue with SVM  2:Choose other model")
        decision = int(input())
        if decision == 1:
            rst = SVM(X,y,X_test)
    elif max_r2 == r2_KNN:
        print("KNN is the recommend model.")
        print("1:Continue with KNN  2:Choose other model")
        decision = int(input())
        if decision == 1:
            rst = KNN(X,y,X_test,5)
    elif max_r2 == r2_DT:
        print("DT is the recommend model.")
        print("1:Continue with DT  2:Choose other model")
        decision = int(input())
        if decision == 1:
            rst = Decision_Tree(X,y,X_test)
    elif max_r2 == r2_GNB:
        print("GNB is the recommend model.")
        print("1:Continue with GNB  2:Choose other model")
        decision = int(input())
        if decision == 1:
            rst = GNB(X,y,X_test)
    elif max_r2 == r2_Q:
        print("Quantile is the recommend model.")
        print("1:Continue with Quantile  2:Choose other model")
        decision = int(input())
        if decision == 1:
            rst = Quantile(X,y,X_test)
    if decision == 2:
        print("Enter the type of model you prefer to use for prediction:")
        print("1:Linear  2:Support Vector Machine  3:KNN ")
        print("4:Decision Tree  5:Gaussian Naive bayes  6:Quantile Regression")
        model = int(input())
        if model == 1:
            rst = Linear(X,y,X_test)
        elif model == 2:
            rst = SVM(X,y,X_test)
        elif model == 3:
            rst = KNN(X,y,X_test,5)
        elif model == 4:
            rst = Decision_Tree(X,y,X_test)
        elif model == 5:
            rst = GNB(X,y,X_test)
        elif model == 6:
            rst = Quantile(X,y,X_test)
    if year <= 2022:
        print("Number of shipped products based on prediction model: ", rst[month - 1])
        print("Number of products actually being shipped: ", Data_s_test[0][month - 1])
        print("Gap: ", rst[month - 1] - Data_s_test[0][month - 1])
    else:
        print("Number of shipped products based on prediction model: ", rst)

    #delay method
    pivot = get_pivotMonth(month, category)
    delay_tb = delay(month, category, pivot)
    delay_rst = delay_r(delay_tb, month, pivot)
    delay_predict = 0
    if year > 2022:
        for i in range(0, pivot+1):
            delay_predict = delay_predict + book_list[b_size-1-i]*delay_rst[i]
        delay_predict = delay_predict / (1-delay_rst[pivot+1])
    else:
        book_temp = get_pastmonth(year, month, category, pivot)
        for i in range(0, pivot+1):
            delay_predict = delay_predict + book_temp[i]*delay_rst[i]
        delay_predict = delay_predict / (1-delay_rst[pivot+1])
        #delay_predict = (book_temp[3] * delay_rst[0] + book_temp[2] * delay_rst[1] + book_temp[1] * delay_rst[2] + book_temp[0] * delay_rst[3]) / (1 - delay_rst[4])
    print("Delay method prediction result: ", delay_predict)

if __name__ == "__main__":
    main()


