import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
import random
import seaborn as sns

df = pd.read_csv("C:/Users/Hetal Atwal/OneDrive/Desktop/datasets/epileptic.csv")
print(df.head(10))

print(df.info())
print(df.isnull().sum())
print(df.describe())

df.loc[df.y == 2, 'y'] = 0
df.loc[df.y == 3, 'y'] = 0
df.loc[df.y == 4, 'y'] = 0
df.loc[df.y == 5, 'y'] = 0

# print(df.head())

y = df['y']
x = df.iloc[:, 1:179]

# print(x)
# print(y)


scaler = StandardScaler()
x = scaler.fit_transform(x)
# print(x)

cov = (x.T @ x) / (x.shape[0] - 1)
eig_values, eig_vectors = np.linalg.eig(cov)
idx = np.argsort(eig_values, axis=0)[::-1]
aftsort_eig_vectors = eig_vectors[:, idx]
# print(aftsort_eig_vectors)

cumsum = np.cumsum(eig_values[idx]) / np.sum(eig_values[idx])
xint = range(1, len(cumsum) + 1)
plt.plot(xint, cumsum)
plt.xlabel("Components")
plt.ylabel("Explained variance")
plt.show()

eig_scores = np.dot(x, aftsort_eig_vectors[:, :75])
print(eig_scores)
print(eig_scores.shape)

xnew = df.iloc[:, 1:75]
df_new = pd.concat([xnew, y], axis=1)

trainx, testx, trainy, testy = train_test_split(xnew, y, test_size=0.2, random_state=20)
print(xnew.shape)
print(y.shape)

error = []

for i in range(1, 20):
    n = KNeighborsClassifier(n_neighbors=i)
    n.fit(trainx, trainy)
    pred = n.predict(testx)
    error.append(np.mean(pred != testy))
    # print(pred)

n = KNeighborsClassifier(n_neighbors=3)
n.fit(trainx, trainy)
pred = n.predict(testx)
print(n.get_params(deep=True))
print(n.kneighbors(X=None, n_neighbors=None, return_distance=True))

# visualise knn
plt.figure(figsize=(10, 6))
plt.plot(range(1, 20), error, color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
plt.show()

plt.figure(figsize=(5, 7))
ax = sns.distplot(testy, hist=False, color="r", label="Actual Value")
sns.distplot(pred, hist=False, color="b", label="Predicted Values", ax=ax)
plt.title('Actual vs Precited value for outcome')
plt.show()
plt.close()

cv = KFold(n_splits=10, random_state=1, shuffle=True)
score = cross_validate(n, trainx, trainy, scoring='accuracy', cv=cv)
for trin, tein in cv.split(df_new):
    print(trin, tein)
# print the final test score
print(score['test_score'])


conf = confusion_matrix(testy, pred, labels=n.classes_)
print(conf)
disp = ConfusionMatrixDisplay(confusion_matrix=conf, display_labels=n.classes_)
disp.plot()
plt.show()
print(classification_report(testy, pred))

rf_clf = RandomForestClassifier(criterion='entropy')
rf_clf.fit(trainx, trainy)
y_predict = rf_clf.predict(testx)

print("Accuracy score for random forest: ", accuracy_score(testy, y_predict))
conf = confusion_matrix(testy, y_predict, labels=rf_clf.classes_)
print(conf)
disp = ConfusionMatrixDisplay(confusion_matrix=conf, display_labels=rf_clf.classes_)
disp.plot()
plt.show()
print(classification_report(testy, y_predict))


