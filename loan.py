import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from random import randint 
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix,roc_auc_score
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold, cross_val_score,cross_validate,GridSearchCV
from sklearn.ensemble import RandomForestClassifier 
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text
import pydotplus
import pickle


train=pd.read_csv('./Loan_Data/train.csv')

#%%

Loan_status=train.Loan_Status
train.drop('Loan_Status',axis=1,inplace=True)
test=pd.read_csv('./Loan_Data/test.csv')
Loan_ID=test.Loan_ID
data=train.append(test)
data.head()
df=data.copy()

#%%

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())


check_df(df)

#%%

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car, = grab_col_names(df)

#%%
def cat_summary(dataframe, col_name, plot=True):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()


for col in cat_cols:
    cat_summary(df, col, plot=True)

#%%
def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)


for col in num_cols:
    num_summary(df, col, plot=True)
    
    
#%%

def target_summary_with_cat(df, target, categorical_col):
    print(categorical_col)
    print(pd.DataFrame({"TARGET_MEAN": df.groupby(categorical_col)[target].mean(),
                        "Count": df[categorical_col].value_counts(),
                        "Ratio": 100 * df[categorical_col].value_counts() / len(df)}), end="\n\n\n")


for col in cat_cols:
    target_summary_with_cat(df, "LoanAmount", col)


def missin_vals(data,na_name=False):
    na_columns = [col for col in data.columns if data[col].isnull().sum() > 0]

    n_miss = data[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (data[na_columns].isnull().sum() / data.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns
    
    #there s no null values 
miss=missin_vals(df,True)
#%%
# * 1.6.Data visualization *
#############################


corrmat=df.corr()
f,ax=plt.subplots(figsize=(9,9))
sns.heatmap(corrmat,vmax=.8,square=True)

df.drop('Loan_ID',inplace=True,axis=1)

df.info()
#%%
## Label encoding for gender
df.Gender=df.Gender.map({'Male':1,'Female':0})
df.Gender.value_counts()

## Labelling 0 & 1 for Marrital status
df.Married=df.Married.map({'Yes':1,'No':0})




## Labelling 0 & 1 for Dependents
df.Dependents=df.Dependents.map({'0':0,'1':1,'2':2,'3+':3})


df.Dependents.value_counts()

## Labelling 0 & 1 for Education Status
df.Education=df.Education.map({'Graduate':1,'Not Graduate':0})


df.Education.value_counts()

## Labelling 0 & 1 for Employment status
df.Self_Employed=df.Self_Employed.map({'Yes':1,'No':0})

df.Self_Employed.value_counts()


df.Property_Area=df.Property_Area.map({'Urban':2,'Rural':0,'Semiurban':1})
df.Property_Area.value_counts()




df.info()


## Dropping Loan ID from data, it's not useful

#%%


def fill_missing_values(df, column, strategy='random'):
    """
    Parameters:
    - df: DataFrame
      İşlem yapılacak DataFrame.
    - column: str
      Eksik değerleri doldurulacak sütun adı.
    - strategy: str, optional
      Kullanılacak doldurma stratejisi. 'random', 'median', veya 'mean' olabilir.
      Varsayılan değer 'random'.
    """
    if strategy == 'random':
        df[column].fillna(np.random.randint(0, 2), inplace=True)
    elif strategy == 'median':
        df[column].fillna(df[column].median(), inplace=True)
    elif strategy == 'mean':
        df[column].fillna(df[column].mean(), inplace=True)
    else:
        raise ValueError("Geçersiz doldurma stratejisi. 'random', 'median', veya 'mean' olabilir.")
    
    return df


df.describe().T


df = fill_missing_values(df, 'Married', strategy='random')
df = fill_missing_values(df, 'Gender', strategy='random')
df = fill_missing_values(df, 'Dependents', strategy='random')
df = fill_missing_values(df, 'Self_Employed', strategy='random')
df=fill_missing_values(df, 'Credit_History',strategy='random')
df=fill_missing_values(df, 'LoanAmount',strategy='median')
df=fill_missing_values(df, 'Loan_Amount_Term',strategy='median')

df.isnull().sum()
#%%

df.describe().T
#%%



train_X=df.iloc[:614,] ## all the data in X (Train set)
train_y=Loan_status  ## Loan status will be our Y



train_X,test_X,train_y,test_y=train_test_split(train_X,train_y,random_state=0)


train_X.head()

test_X.head()
"""
models=[]
models.append(("Logistic Regression",LogisticRegression()))
models.append(("Decision Tree",DecisionTreeClassifier()))
models.append(("Linear Discriminant Analysis",LinearDiscriminantAnalysis()))
models.append(("Random Forest",RandomForestClassifier()))
models.append(("K- Neirest Neighbour",KNeighborsClassifier()))
models.append(("Naive Bayes",GaussianNB()))

scoring='accuracy'

result=[]
names=[]
for name,model in models:
    kfold=KFold(n_splits=10,random_state=0)
    cv_result=cross_val_score(model,train_X,train_y,cv=kfold,scoring=scoring)
    result.append(cv_result)
    names.append(name)
    print(model)
    print("%s %f" % (name,cv_result.mean()))
    """
#%%
def base_models(X, y, scoring="roc_auc"):
    print("Base Models....")
    classifiers = [('LR', LogisticRegression()),
                   ('KNN', KNeighborsClassifier()),
                   ("SVC", SVC()),
                   ("CART", DecisionTreeClassifier()),
                   ("RF", RandomForestClassifier()),
                   ("Decision Tree",DecisionTreeClassifier()),
                   # ('CatBoost', CatBoostClassifier(verbose=False))
                   ]

    for name, classifier in classifiers:
        cv_results = cross_val_score(classifier, X, y, cv=3, scoring=scoring)
        print(name,cv_results.mean())

base_models(train_X, train_y, scoring="accuracy")
    
#%%%

"""
LR=LogisticRegression()
LR.fit(train_X,train_y)
pred=LR.predict(test_X)
print("Model Accuracy:- ",accuracy_score(test_y,pred))
print(confusion_matrix(test_y,pred))
print(classification_report(test_y,pred))

print(pred)"""



# Eğitim veri seti
train_y.replace({'N': 0, 'Y': 1}, inplace=True)

# Test veri seti
test_y.replace({'N': 0, 'Y': 1}, inplace=True)

# Modeli eğitme
LR = LogisticRegression()
LR.fit(train_X, train_y)

# Tahminler
pred = LR.predict(test_X)

# Sonuçları görüntüleme
print("Model Accuracy: ", accuracy_score(test_y, pred))

#pred))Model Accuracy:  0.8116883116883117
print(confusion_matrix(test_y, pred))
print(classification_report(test_y, pred))
print(pred)
#%%
"""
import os
model = "modelim"
file_path = 'model.pkl'

with open(file_path, 'wb') as f:
    pickle.dump(model, f)

# Model dosyasını bir klasöre taşı
target_folder = './modelim'

if not os.path.exists(target_folder):
    os.makedirs(target_folder)

target_path = os.path.join(target_folder, 'model.pkl')
os.rename(file_path, target_path)
"""
#%%

X_test=df.iloc[614:,] 
# X_test[sc_f]=SC.fit_transform(X_test[sc_f])

prediction = LR.predict(X_test)


print(prediction)


## TAken data from the dataset
sample = LR.predict([[1,	0.0,	0.0,	1,	0.0,	1811,	1666.0,	54.0,	360.0,	1.0,	2]])

#%%
df.head()
#%%


y=Loan_status
X=df.iloc[:614,] ## all the data in X (Train set)

cart_model = DecisionTreeClassifier(random_state=1).fit(X, y)

# Confusion matrix için y_pred:
y_pred = cart_model.predict(X)

# AUC için y_prob:
y_prob = cart_model.predict_proba(X)[:, 1]

# Confusion matrix
print(classification_report(y, y_pred))

# AUC
roc_auc_score(y, y_prob)
#%%

train_X, test_X, train_Y, test_Y = train_test_split(X, y, test_size=0.30, random_state=45)

model_tr = DecisionTreeClassifier(random_state=17).fit(train_X, train_Y)

# Train Hatası
y_pred = cart_model.predict(train_X)
y_prob = cart_model.predict_proba(train_X)[:, 1]
print(classification_report(train_Y, y_pred))
roc_auc_score(train_Y, y_prob)
#%%
# Test Hatası
y_pred = cart_model.predict(test_X)
y_prob = cart_model.predict_proba(test_X)[:, 1]
print(classification_report(test_Y, y_pred))
roc_auc_score(test_Y, y_prob)
#%%
#####################
# CV ile Başarı Değerlendirme
#####################

model_tr = DecisionTreeClassifier(random_state=17).fit(X, y)

#fiti i koymayabilirdik ama 

cv_results = cross_validate(model_tr,
                            X, y,
                            cv=5,
                            scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
# 0.7058568882098294
cv_results['test_f1'].mean()
# 0.5710621194523633
cv_results['test_roc_auc'].mean()
# 0.6719440950384347

# cv_results['test_accuracy'].mean()
# Out[24]: 0.6758629881380781

#%%
################################################
# 4. Hyperparameter Optimization with GridSearchCV
################################################

model_tr.get_params()

params = {'max_depth': range(1, 11),
               "min_samples_split": range(2, 20)}

best_grid = GridSearchCV(model_tr,
                              params,
                              cv=5,
                              n_jobs=-1,
                              verbose=1).fit(X, y)

best_grid.best_params_

best_grid.best_score_

random = X.sample(1, random_state=45)

best_grid.predict(random)

#%%
################################################
# 5. Final Model
################################################

#best parameteleri modele verip ölcüm yaptık

model_final = DecisionTreeClassifier(**best_grid.best_params_, random_state=17).fit(X, y)
model_final.get_params()

model_final = model_tr.set_params(**best_grid.best_params_).fit(X, y)

cv_results = cross_validate(model_final,
                            X, y,
                            cv=5,
                            scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()

cv_results['test_f1'].mean()

cv_results['test_roc_auc'].mean()

#%%

def tree_graph(model, col_names, file_name):
    tree_str = export_graphviz(model, feature_names=col_names, filled=True, out_file=None)
    graph = pydotplus.graph_from_dot_data(tree_str)
    graph.write_png(file_name)


tree_graph(model=model_final, col_names=X.columns, file_name="model_final.png")

model_tr.get_params()

#%%
################################################
# 9. Extracting Decision Rules
################################################

tree_rules = export_text(model_final, feature_names=list(X.columns))
print(tree_rules)

#%%


joblib.dump(model_final, "model_final.pkl")
from_disc = joblib.load("model_final.pkl")