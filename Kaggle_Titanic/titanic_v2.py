# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 21:51:32 2020

@author: gabri
"""

#Importando algumas bibliotecas padrões
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Ignorar os Warnings : ###############
import warnings
warnings.filterwarnings("ignore")
######################################
#Importando a base para o diretório ##
try:
    print(os.getcwd())
    os.chdir('..\gabri\Documents\Datasets\Kaggle_titanic')
    print(os.getcwd())
except:
    print('Não encontrado')
    
gender_submission = pd.read_csv('gender_submission.csv')

df_train = pd.read_csv('train.csv')
df_test  = pd.read_csv('test.csv')

df_train.head()
df_train.describe()
df_train.count()

df_test = pd.merge(df_test,gender_submission, 
                     on = 'PassengerId')

del gender_submission

df_train.count()
df_train.isna().sum()

df_full = df_train.append(df_test)

###########################################
#Respondendo alguns questionamentos
###########################################
#1. Qual foi a porcentagem dos passageiros sobreviventes?
print('Sobreviveu: ', round(df_full['Survived'].value_counts()[0]/len(df_full) * 100, 2),
      '% da base / Nº Observações: ', round(df_full['Survived'].value_counts()[0]))

print('Morreu: ', round(df_full['Survived'].value_counts()[1]/len(df_full) * 100, 2),
      '% da base / Nº Observações: ', round(df_full['Survived'].value_counts()[1]))

#2. Qual era a faixa etária dos passageiros que estavam no Titanic?
print('Faixa Média dos passageiros abordos: ', round(sum(df_full['Age'])/len(df_full),2))


fig, ax = plt.subplots(1,1,figsize=(10,6))
sns.distplot(df_full[df_full.Survived ==1]['Age'].dropna().values, color ='blue',
             label='Ñ Sobreviveu')
sns.distplot(df_full[df_full.Survived ==0 ]['Age'].dropna().values, color ='red',
             label='Sobreviveu')
fig.legend(labels=['Morreu','Sobreviveu'])
ax.set_title('Distribuição de Idade por público que estava no barco')

#3. Quantas crianças ou adultos sobreviveram?
print(' N° de Crianças sobreviventes: ', 
      df_full[((df_full.Age <  18) & (df_full.Survived == 1)) & (df_full['Age'].isna() == 0)].shape[0],
      '\n N° de Adultos  sobreviventes: ', 
      df_full[((df_full.Age >= 18) & (df_full.Survived == 1) & (df_full['Age'].isna() == 0))].shape[0])

#4. % de Sobreviventes por classe
print(df_full[['Pclass', 'Survived']].groupby(['Pclass'], 
        as_index=False).mean().sort_values(by='Survived', 
                            ascending=False))

###########################################
#Conhecendo melhor os dados das bases######
###########################################
a_bases_anl = ['df_full', 'df_train', 'df_test'] #Bases que serão alteradas

#Verifica os valores das variáveis Cabin e Ticket
x = pd.crosstab(df_full['Cabin'],df_full['Cabin'].count())
y = pd.crosstab(df_full['Ticket'],df_full['Ticket'].count())
print(x,y)
del x, y

#Tratando algumas variáveis#########
a_fxs_idade = (-1, 0, 1, 5, 12, 18, 25,35,60,999)
a_grp_idade = ['Desconhecido', 'Recem_Nascido','Bebe','Criança','Pre_Adolescente',
             'Adolescente', 'Adulto_Jovem','Adulto', 'Idoso']

for var_x in a_bases_anl:
    locals()[var_x]['Cabin_c'] = locals()[var_x].Cabin.fillna('N')
    locals()[var_x]['Cabin_c'] = locals()[var_x].Cabin_c.apply(lambda x: x[0])
    locals()[var_x]['Age'] = locals()[var_x].Age.fillna(-0.1)
    categorias = pd.cut(locals()[var_x].Age, a_fxs_idade, labels=a_grp_idade)
    locals()[var_x]['Age_c'] = categorias
    #Dropando a Variável de Cabine e Ticket e valores nulos
    locals()[var_x] = locals()[var_x].drop(['Cabin','Ticket'],axis = 1)
    locals()[var_x] = locals()[var_x].dropna()  
    locals()[var_x] = locals()[var_x].set_index('PassengerId')
    
df_full.isna().sum()
x = pd.crosstab(df_full['Age'],df_full['Age_c'])

df_full['Sex'].unique()

#########################################################################
#########################################################################
pd.crosstab(df_full['Embarked'],df_full['Embarked'].count())


##############Funções Úteis ############################
def var_exists(base,variavel):
    if str(variavel) in base.columns.tolist():
        return int(1)
    else:
        print('Variavel: ',str(variavel),'Não Existe')        
        return int(0)
########################################################


for var_x in a_bases_anl:
    #Criando novas variáveis
    locals()[var_x]['Title'] = locals()[var_x].Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    locals()[var_x]['Title'] = locals()[var_x]['Title'].replace(['Mlle','Ms'], 'Miss')
    locals()[var_x]['Title'] = locals()[var_x]['Title'].replace('Mme', 'Mrs')
    locals()[var_x]['Title'] = locals()[var_x]['Title'].replace(['Lady', 'Countess',
          'Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer',
          'Dona'],'Outros')
    locals()[var_x]['Tamanho_Familia'] = locals()[var_x]['SibSp'] + locals()[var_x]['Parch']
    locals()[var_x]['Solo'] = 0
    locals()[var_x].loc[locals()[var_x]['Tamanho_Familia'] == 0, 'Solo'] = 1    
    #Gerando Dummies
    locals()[var_x + '_d'] = pd.get_dummies(locals()[var_x][['Embarked','Sex',
           'Cabin_c','Title','Age_c']], prefix_sep='_',
           drop_first=True, dtype= int)    
    locals()[var_x] = pd.merge(locals()[var_x],locals()[var_x + '_d'], 
           how='left',left_index=True, right_index=True)
    #Removendo as variáveis que não serão utilizadas
    for var_cols in ('Embarked','Sex','Cabin_c','Title','Age_c','Name'):
        if var_exists(locals()[var_x], str(var_cols)) > 0:
            locals()[var_x] = locals()[var_x].drop(str(var_cols), axis=1)
    #Não vamos gerar dummies para as variáveis P_class, por existir ordem em seu
    #valor
del var_x,var_cols

pd.crosstab(df_full['Tamanho_Familia'],df_full['Tamanho_Familia'].count())

"""
################################################################################
#Aplicando os Encoders
##########################

#Get Dummies com o Pandas
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    
#1º: Label Encoder para transformar as variáveis categóricas strings em numéricas
le = LabelEncoder()

categorical_feature_mask = df_full.dtypes==object
categorical_cols = df_full.columns[categorical_feature_mask].tolist()


categorical_cols_le = []
for x in range(0,(len(categorical_cols))):
    categorical_cols_le.append(str(categorical_cols[x]) + '_le')

df_full[categorical_cols_le] = df_full[categorical_cols].apply(lambda col: le.fit_transform(col))


dict_title = df_full[['Title_le','Title']].drop_duplicates(subset=None, 
                    keep='first', inplace=False).set_index('Title_le').to_dict()

dict_embarked = df_full[['Embarked_le','Embarked']].drop_duplicates(subset=None, 
                    keep='first', inplace=False).set_index('Embarked_le').to_dict()

df_full = df_full.drop(['Title','Embarked'], axis = 1)

df_full.rename(columns = {'Title_le':'Title',
                          'Embarked_le':'Embarked'},inplace=True)

del categorical_cols_le

#######
#2º: Aplicamos o OneHotEncoder para gerar dummies
oe = OneHotEncoder(categorical_features = categorical_feature_mask, sparse=False)
# apply OneHotEncoder on categorical feature columns
df_oe = oe.fit_transform(df_full) # It returns an numpy array
print(df_oe)
"""

#############################################################################
#Transformando a escala #####################################################
#############################################################################

from sklearn.preprocessing import StandardScaler, RobustScaler

std_scaler = StandardScaler()
rbt_scaler = RobustScaler()

###Verificando variáveis de Fare e Age

fig, ax = plt.subplots(2,1,figsize=(10,6))

fare_val = df_full['Fare'].values
age_val = df_full['Age'].values

sns.distplot(fare_val, ax = ax[0], color='red')
ax[0].set_title('Distribuição da dos valores de fare', fontsize = 10)
ax[0].set_xlim([min(fare_val),max(fare_val)])

sns.distplot(age_val, ax = ax[1], color='blue')
ax[1].set_title('Distribuição dos valores de idade', fontsize = 10)
ax[1].set_xlim([min(age_val),max(age_val)])
plt.show()

#Aplicaremos o Robust Scaler nas variáveis
df_full['scaled_fare'] = rbt_scaler.fit_transform(df_full['Fare'].values.reshape(-1,1))
df_full['scaled_age']  = rbt_scaler.fit_transform(df_full['Age'].values.reshape(-1,1))

df_test = pd.merge(df_test, df_full[['scaled_fare','scaled_age']], how='left',
                    left_index=True, right_index=True)

df_train = pd.merge(df_train, df_full[['scaled_fare','scaled_age']], how='left',
                    left_index=True, right_index=True)

############################################
fig, ax = plt.subplots(1,1,figsize=(10,6))

plt.scatter(fare_val, age_val, c='blue', alpha = 0.5)
plt.show()


f, ax = plt.subplots(1,1, figsize=(12,12))
corr = df_full.corr()
sns.heatmap(corr, cmap ='coolwarm_r',annot_kws={'size':10})
ax.set_title('Correlação dos dados')
del fare_val, age_val, corr

for var_x in a_bases_anl:
    locals()[var_x] = locals()[var_x].drop(['Fare','Age'], axis = 1)
del var_x

##############################################
#Tratando os Outliers
#Trataremos Fare e Age
#Método de médias de desvio padrão
"""
for var_x in ('Fare','Age'):
    locals()['i_' + var_x + '_mean'] = np.mean(df_full[var_x].values)
    locals()['i_' + var_x + '_std']  = np.std(df_full[var_x].values)
    #Um ponto de corte de 3 * o desvio padrão da média 
    #diminui a eliminação dos dados
    locals()['i_' + var_x + '_cutoff'] = locals()['i_' + var_x + '_std'] * 3
    locals()['i_' + var_x + '_lower' ] = locals()['i_' + var_x + '_mean'] - locals()['i_' + var_x + '_cutoff']
    locals()['i_' + var_x + '_upper' ] = locals()['i_' + var_x + '_mean'] + locals()['i_' + var_x + '_cutoff']
""" 

#Como não tem nenhum valor discrepante como outlier na base, não vamos remove-los
#############################################

a_df_full_cols = list(df_full.columns) #Pega todas as variáveis da base

X_train, X_test = df_train.drop('Survived', axis = 1), df_test.drop('Survived', axis = 1)
y_train, y_test = df_train['Survived'], df_test['Survived']

#Números diferentes nas bases de teste e treino
#Verificando...
x = list(df_train.columns)
y = list(df_test.columns)

for i in range(0,31):
    if x[i] not in y:
        print(x[i], 'não existe')
        df_test[str(x[i])] = 0

del x,y

#Criando bases novamente, com coluns completas
X_train, X_test = df_train.drop('Survived', axis = 1), df_test.drop('Survived', axis = 1)
y_train, y_test = df_train['Survived'], df_test['Survived']



#Base toda tratada, podemos dar inicio aos modelos

#############################
###Modelos
#############################
import time
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


###################
#Random Forest
###################

rfc = RandomForestClassifier()

# Choose some parameter combinations to try
parameters = {'n_estimators': [4, 6, 9], 
              'max_features': ['log2', 'sqrt','auto'], 
              'criterion': ['entropy', 'gini'],
              'max_depth': [2, 3, 5, 10], 
              'min_samples_split': [2, 3, 5],
              'min_samples_leaf': [1,5,8]
             }

# Type of scoring used to compare parameter combinations
acc_scorer = make_scorer(accuracy_score)

# Run the grid search
grid_obj = GridSearchCV(rfc, parameters, scoring=acc_scorer)
grid_obj = grid_obj.fit(X_train, y_train)

# Set the clf to the best combination of parameters
rfc = grid_obj.best_estimator_

# Fit the best algorithm to the data. 
rfc.fit(X_train, y_train)

rfc_prediction = rfc.predict(X_test)
rfc_score=accuracy_score(y_test, rfc_prediction)

#Verificando a importância de cada variável no modelo
rfc_feature_imp = pd.Series(rfc.feature_importances_, 
                            index=X_train.columns).sort_values(ascending=False)

rfc_removal = []
for i in range(0,len(rfc_feature_imp)):
    if (rfc_feature_imp[i] < 0.03):
        print('Variável ', i, 'é insignificante')
        
i = 0
while (rfc_feature_imp[i] > 0.03):
    print('Variável ', i, 'é significante')
    i = i + 1
      
rfc_removal = rfc_feature_imp[i:] #Separando variávies para tirar do modelo
x = list(rfc_removal.index) #Pegando o nome dessas colunas

#Rodando o modelo novamente
rfc = RandomForestClassifier()

# Run the grid search
grid_obj2 = GridSearchCV(rfc, parameters, scoring=acc_scorer)
grid_obj2 = grid_obj.fit(X_train.drop(x, axis = 1), y_train)

print(list(X_train.drop(x, axis = 1).columns))

# Set the clf to the best combination of parameters
rfc2 = grid_obj2.best_estimator_

# Fit the best algorithm to the data. 
rfc2.fit(X_train.drop(x, axis = 1), y_train)

rfc_prediction2 = rfc2.predict(X_test.drop(x, axis = 1))
rfc_score2 =accuracy_score(y_test, rfc_prediction2)



###################
#XGBoosting
###################
import xgboost as xgb

xgboost = xgb.XGBClassifier(max_depth=3, n_estimators=300, 
                            learning_rate=0.05).fit(X_train, y_train)


X_test = X_test[list(X_train)] #Organizando a base de teste igual a de treino

xgb_prediction = xgboost.predict(X_test)

xgb_score = accuracy_score(y_test, xgb_prediction)

print(xgb_score)


xgb_feature_imp = pd.Series(xgboost.feature_importances_, 
                            index=X_train.columns).sort_values(ascending=False)

xgb_removal = []
for i in range(0,len(xgb_feature_imp)):
    if (xgb_feature_imp[i] < 0.03):
        print('Variável ', i, 'é insignificante')
        
i = 0
while (xgb_feature_imp[i] > 0.03):
    print('Variável ', i, 'é significante')
    i = i + 1
      
xgb_removal = xgb_feature_imp[i:] #Separando variávies para tirar do modelo
x = list(xgb_removal.index) #Pegando o nome dessas colunas

#Rodando o modelo novamente:
xgboost = xgb.XGBClassifier(max_depth=3, n_estimators=300, 
                            learning_rate=0.05).fit(X_train.drop(x, axis = 1), y_train)


X_test_xgb = X_test[list(X_train.drop(x, axis = 1).columns)] #Organizando a base de teste igual a de treino

xgb_prediction_2 = xgboost.predict(X_test_xgb)

xgb_score_2 = accuracy_score(y_test, xgb_prediction_2)

print(' #######Score \n Anterior:',round(xgb_score,4),'\n Novo:', round(xgb_score_2,4))


###################
#ExtraTrees
###################
from sklearn.ensemble import ExtraTreesClassifier

exttree = ExtraTreesClassifier(criterion='gini',
                               max_features='auto',
                               max_depth=None,
                               min_samples_split = 2,
                               min_samples_leaf = 1,
                               min_impurity_split=None,
                               random_state=None
                               )
exttree.fit(X_train, y_train)

ext_prediction = exttree.predict(X_test)
ext_score = accuracy_score(y_test, ext_prediction)
print(ext_score)

###################
#AdaBoost
###################
from sklearn.ensemble import AdaBoostClassifier

adab = AdaBoostClassifier(base_estimator = None,
                          n_estimators = 300,
                          learning_rate = .75,
                          algorithm='SAMME',
                          random_state=None)

adab.fit(X_train, y_train)

adab_prediction = adab.predict(X_test)
adab_score = accuracy_score(y_test,adab_prediction)
print(adab_score)








