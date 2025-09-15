import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score, Kfold
from sklearn.metrics import classification_report, confusion_matrix


#import data
banks_df = pd.read_csv('data/bank/bank.csv', sep=';')
#banks_full_df = pd.read_csv("data/bank/bank-full.csv")
#print(banks_df)

column_names = banks_df.columns.tolist()
print(column_names)

#print(banks_full_df)


# independent variables
X = banks_df.drop("y", axis=1).values 
# dependent variable
y = banks_df['y'].values


X_ed = X[:,3]
print(y.shape, X_ed.shape)


plt.scatter(X_ed, y)
plt.ylabel("Y")
plt.xlabel("Education")
plt.show()

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .25, random_state=123, stratify=y)

reg = LinearRegression()
reg.fit(X_ed, y)
reg.predict()

#R^2
reg.score(X_test, y_test)

#RMSE
root_mean_squared_error(y_test, y_pred)


# Cross Validation
kf = Kfold(n_split=6, shuffle=True, random_state=123)
cv_results = cross_val_score(reg, X, y, cv=kf) #defaults to R^2
print(cv_results)

print(np.mean(cv_results), np.std(cv_results))

print(np.quantile(cv_results, [0.025, 0.975]))

# Regularization - peanlize coefficients

# Ridge
from sklearn.linear_model import Ridge

scores = []
for alpha  in [0.1,1, 10]:
	ridge = Ridge(aha=alpha)
	ridge.fit(X_train, y_train)
	ridg_pred = ridge.predict(X_test)
	scores.append(ridg_pred.score(X_test, y_test))

print(scores)

# Lasso 

from sklearn.linear_model import Lasso

scores = []
for alpha  in [0.1,1, 10]:
	lasso = Lasso(alpha=alpha)
	lasso.fit(X_train, y_train)
	lasso_pred = lasso.predict(X_test)
	scores.append(lasso_pred.score(X_test, y_test))

print(scores)


# Ex
col_names = banks_df.drop("y", axis=1).columns
lasso = Lasso(alpha=0.1)
lasso_coef = lasso.fit(X, y).coef_ # extract coefficients


plt.bar(names, lasso_coef)
plt.xticks(rotation=45)
plt.show()


#consufision matric
knn = KNeightborsClassifer(n_neighbors = 7)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .4, random_state=123)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, x_test))


dropna
from sklearn.impute import SimpleImputer
X_cat = music_df['genre'].values.reshape(-1,1)
X_num = music_df.drop(['genere','popularity'], axis=1).values
y =  music_df['popularity'].values
X_train_cat, X_test_cat, y_train, y_test = train_test_split(X_cat, text_size=0.2, random_state=12)
X_train_num, X_test_num, y_train, y_test = train_test_split(X_num, text_size=0.2, random_state=12)

imp_cat = SimpleImputer(strategy="most_frequent")
X_train_cat = imp_cat.fit_transform(X_train_cat)
X_test_cat = imp_cat.transform(X_test_cat)


imp_num = SimpleImputer(strategy="most_frequent")
X_train_num = imp_num.fit_transform(X_train_num)
X_test_num = imp_num.transform(X_test_num)

X_train = np.append(X_train_num, X_train_cat, axis=1)
X_test = np.append(X_test_num, X_test_cat, axis=1)

klearn.pipeline import Pipeline


df.describe()


sklearn.preprosseing import StandardScaler
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_tesr_scaled = scaler.transform(X_test)