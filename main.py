import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA

# import the data
iris = datasets.load_iris()
X = iris.data[:, :]  # we only take all the features.
y = iris.target

# present the first three PCA directions in 3D graph
fig = plt.figure()
axes3D = Axes3D(fig, elev=-150, azim=110)
X_reduced = PCA(n_components=4).fit_transform(X)
axes3D.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], X_reduced[:, 3], c=y, cmap=plt.cm.Set1, edgecolor='k',
               s=40)
axes3D.set_title("First three PCA directions")
axes3D.set_xlabel("1st eigenvector")
axes3D.w_xaxis.set_ticklabels([])
axes3D.set_ylabel("2nd eigenvector")
axes3D.w_yaxis.set_ticklabels([])
axes3D.set_zlabel("3rd eigenvector")
axes3D.w_zaxis.set_ticklabels([])
plt.show()

# Divide the dataset to training set and testing set
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

## Classification:
# Logistic Regression
from sklearn.linear_model import LogisticRegression

classifier_logistic_regression = LogisticRegression()
classifier_logistic_regression.fit(X_train, y_train)
y_pred = classifier_logistic_regression.predict(X_test)
cm_logistic_regression = confusion_matrix(y_test, y_pred)
accuracy_score_LR = accuracy_score(y_test, y_pred)

# K-Nearest Neighbors (K-NN)
from sklearn.neighbors import KNeighborsClassifier

classifier_KNeighbors = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
classifier_KNeighbors.fit(X_train, y_train)
y_pred = classifier_KNeighbors.predict(X_test)
cm_KN = confusion_matrix(y_test, y_pred)
accuracy_score_KN = accuracy_score(y_test, y_pred)

# SVC - Support Vector Machine
from sklearn.svm import SVC

classifier_SVM = SVC(kernel='linear')
classifier_SVM.fit(X_train, y_train)
y_pred = classifier_SVM.predict(X_test)
accuracy_score_SVM = accuracy_score(y_test, y_pred)
cm_SVM = confusion_matrix(y_test, y_pred)

# Kernel SVM - Support Vector Machine with Kernel
kernel = 'rbf'  # type of Kernel
classifier_Kernel = SVC(kernel=kernel)
classifier_Kernel.fit(X_train, y_train)
y_pred = classifier_Kernel.predict(X_test)
accuracy_score_Kernel = accuracy_score(y_test, y_pred)
cm_Kernel = confusion_matrix(y_test, y_pred)

# naive bayes
from sklearn.naive_bayes import GaussianNB

classifier_NB = GaussianNB()
classifier_NB.fit(X_train, y_train)
y_pred = classifier_NB.predict(X_test)
accuracy_score_NB = accuracy_score(y_test, y_pred)
cm_NB = confusion_matrix(y_test, y_pred)

# Decision Tree
from sklearn.tree import DecisionTreeClassifier

criterion = 'entropy'
classifier_DT = DecisionTreeClassifier(criterion=criterion)
classifier_DT.fit(X_train, y_train)
y_pred = classifier_DT.predict(X_test)
accuracy_score_DT = accuracy_score(y_test, y_pred)
cm_DT = confusion_matrix(y_test, y_pred)

# Random Forest
from sklearn.ensemble import RandomForestClassifier

number_of_trees = 10
criterion = 'entropy'
classifier_RF = RandomForestClassifier(n_estimators=10, criterion=criterion)
classifier_RF.fit(X_train, y_train)
y_pred = classifier_DT.predict(X_test)
accuracy_score_RF = accuracy_score(y_test, y_pred)
cm_RF = confusion_matrix(y_test, y_pred)

tree_tubs = "\n" + "\t" * 3


def print_CM(title, cm):
    space = "\n" + "\t" * 3
    return (f"\t{title}:{space}{cm[0]}"
            f"\t{space}{cm[1]}"
            f"\t{space}{cm[2]}\n")


print("confusion matrices:\n"
    +print_CM("Logistic Regression",cm_logistic_regression)
    +print_CM("K-Nearest Neighbors (K-NN)",cm_KN)
    +print_CM("Support Vector Machine (SVM)", cm_SVM)
    +print_CM("Kernel SVM", cm_Kernel)
    +print_CM("Naive Bayes", cm_NB)
    +print_CM("Decision Tree", cm_DT)
    +print_CM("Random Forest", cm_RF)
    )

print("Accuracy Scores:\n"
      f"\tLogistic Regression: {accuracy_score_LR.round(1)},\n"
      f"\tK-Nearest Neighbors (K-NN): {accuracy_score_KN.round(1)},\n"
      f"\tSupport Vector Machine (SVM): {accuracy_score_SVM.round(1)},\n"
      f"\tKernel SVM: {accuracy_score_Kernel.round(1)},\n"
      f"\tNaive Bayes: {accuracy_score_NB.round(1)},\n"
      f"\tDecision Tree: {accuracy_score_DT.round(1)},\n"
      f"\tRandom Forest :{accuracy_score_RF.round(1)}.\n"
      )
