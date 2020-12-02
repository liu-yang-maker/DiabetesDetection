import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Dropout
from flask import Flask, request, render_template
app = Flask(__name__)


# import train data
# transform class label into number


def label_transform(x):
    class_label = {b"tested_positive": 1, b"tested_negative": 0}
    return class_label[x]


def txt_float(x):
    return float(x)


def import_data():
    conv = {0:txt_float,1:txt_float,2:txt_float,3:txt_float,4:txt_float,5:txt_float,6:txt_float,7:txt_float,8:label_transform}
    data = np.loadtxt('Diabetes.csv', dtype=np.float, delimiter=",", skiprows=1, converters=conv)
    y = data[:,-1].astype("int") # train label
    x = data[:,:-1] # train data
    return x, y


# ----------------------------------------------------------------------------------------------------


def get_accuracy(y_true,y_pred):
    return sum(y_true==y_pred)/len(y_true)


''' train data and test data '''

df = pd.read_csv("Diabetes.csv")
y = pd.get_dummies(df['Class'])['tested_positive']
X = df[df.columns[~df.columns.isin(['Class'])]]
X = df[['Age','Number of times pregnant','Diastolic blood pressure ','Body mass index ']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=2020)

scaler = preprocessing.StandardScaler().fit(X_train)
scaled_X_train = scaler.transform(X_train)
scaled_X_test = scaler.transform(X_test)


''' linear regression'''

# linear classifier train
linear_classifier = linear_model.LogisticRegression(random_state=123)
linear_classifier.fit(scaled_X_train, y_train)
y_pred = linear_classifier.predict(scaled_X_test)
# cm = confusion_matrix(y_test,y_pred)
# plot_confusion_matrix(cm)
print("The Accuracy for Regression is: {0:.2f}%".format(get_accuracy(y_test, y_pred)*100))
linear_acc = get_accuracy(y_test, y_pred)

# linear regression predict


def linear_decision(Age, Number_of_times_pregnant, Blood_pressure, BMI):
    result = linear_classifier.predict(np.array([Age, Number_of_times_pregnant, Blood_pressure, BMI]).reshape(1, -1))[0]
    if result == 0:
        return "negative"
    else:
        return "positive"


'''decision tree'''

# decision tree train
tree_clf = tree.DecisionTreeClassifier()
tree_clf = tree_clf.fit(scaled_X_train, y_train)
y_pred = tree_clf.predict(scaled_X_test)
# cm = confusion_matrix(y_test,y_pred)
# plot_confusion_matrix(cm)
print("The Accuracy for Decision Tree is: {0:.2f}%".format(get_accuracy(y_test, y_pred)*100))
tree_acc = get_accuracy(y_test, y_pred)

# decision tree prediction


def tree_decision( Age, Number_of_times_pregnant, Blood_pressure, BMI):
    result = tree_clf.predict(np.array([Age, Number_of_times_pregnant, Blood_pressure, BMI]).reshape(1,-1))[0]
    if result == 0:
        return "negative"
    else:
        return "positive"


'''neural network'''


def get_model(input_dim):
    i = Input(shape=(input_dim))
    x = Dense(216, activation='relu')(i)
    x = Dropout(0.2)(x)
    x = Dense(216, activation='sigmoid')(i)
    x = Dropout(0.2)(x)
    x = Dense(1, activation='relu')(x)
    model = Model(inputs=[i], outputs=[x])
    model.compile(loss='binary_crossentropy',
              optimizer='Rmsprop',
              metrics=['accuracy'])
    # print(model.summary())
    return model


nn = get_model(4)

scaled_X_train = np.asarray(scaled_X_train)
y_train = np.asarray(y_train)
nn.fit(scaled_X_train, y_train,
        batch_size= 16, epochs=100,
        validation_split = 0.2, verbose=0)

scaled_X_test = np.asarray(scaled_X_test)
y_test = np.asarray(y_test)
loss, acc = nn.evaluate(scaled_X_test, y_test)
y_pred =  (nn.predict(scaled_X_test)>0.5).reshape(y_test.shape)
# cm = confusion_matrix(y_test,y_pred)
# plot_confusion_matrix(cm)
nn_acc= get_accuracy(y_test, y_pred)
print("The Accuracy for NN is: {0:.2f}%".format(get_accuracy(y_test, y_pred)*100))


def nn_decision( Age, Number_of_times_pregnant, Blood_pressure, BMI):
    result = nn.predict(np.array([Age, Number_of_times_pregnant, Blood_pressure, BMI]).reshape(1,-1))[0]
    if result == 0:
        return "negative"
    else:
        return "positive"


''' get input data from web page '''


@app.route("/", methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        Age = int(request.form.get('Age'))

        Number_of_times_pregnant = int(request.form.get('Pregnant'))

        Blood_pressure = int(request.form.get('Blood_pressure'))

        Height = float(request.form.get('Height'))

        Weight = float(request.form.get('Weight'))

        BMI = Weight/(Height*Height)

        lr_result = linear_decision(Age, Number_of_times_pregnant, Blood_pressure, BMI)

        dt_result = tree_decision(Age, Number_of_times_pregnant, Blood_pressure, BMI)

        nn_result = nn_decision(Age, Number_of_times_pregnant, Blood_pressure, BMI)

        # sentence0 = "BMI: " + str(BMI) +"\n"
        sentence1 = "Linear Regression：" + str(lr_result) + ", Accuracy: {0:.2f}%".format(linear_acc*100)
        sentence2 = "Decision Tree：" + str(dt_result) + ", Accuracy: {0:.2f}%".format(tree_acc*100)
        sentence3 = "Neural Network：" + str(nn_result) + ", Accuracy: {0:.2f}%".format(nn_acc*100)
        sentence = "Average accuracy: {0:.2f}%".format(((linear_acc+tree_acc+nn_acc)/3)*100)

        response_dict = {
            'sentence': sentence,
            'sentence1': sentence1,
            'sentence2': sentence2,
            'sentence3': sentence3,
            'age':Age,
            'times':Number_of_times_pregnant,
            'pressure':Blood_pressure,
            'weight': Weight,
            'height': Height,
        }

        return render_template("demo.html",
                           **response_dict)
    else:
        response_dict_2 = {
            'sentence': "\t",
            'sentence1': "\t",
            'sentence2': "\t",
            'sentence3': "\t",
            'age': "--Please Enter your Age--",
            'times': "--Please Enter your Pregnant Times--",
            'Pressure': "--Please Enter your Blood Pressure--",
            'Weight': "--Please Enter your Weight(in KG)--",
            'Height': "--Please Enter your Height(m)--",
        }
        return render_template("demo.html", **response_dict_2)

if __name__ == "__main__":
    app.run(debug=False,host='0.0.0.0')