#!/usr/bin/env python
# coding: utf-8

#In[1]
class ML:
## functions
    def norm(x, stats):
        if stats.empty:
            pass
        else:
            return (x - stats['mean']) / stats['std']

    def build_model(nn1=2, nn2=56, nn3 = 1, lr=0.01, decay=0., l1=0.01, l2=0.01, activation ='relu', dropout=0):    
        from tensorflow import keras
        from tensorflow.keras import layers

        opt = keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999,  decay=decay)
        # opt = keras.optimizers.RMSprop(learning_rate=0.001, rho= 0.9, momentum=0.0, epsilon=1e-07, centered=False, name='RMSprop')
        # reg = keras.regularizers.l1_l2(l1=l1, l2=l2)
        reg = keras.regularizers.l2(l2)
    
        model = keras.Sequential([
          layers.Dense( nn1, activation=activation, kernel_regularizer=reg, input_shape=[nn1] ),
          layers.Dense( nn2, activation=activation, kernel_regularizer=reg),  
          layers.Dense( nn2, activation=activation, kernel_regularizer=reg),   
          layers.Dense( nn2, activation=activation, kernel_regularizer=reg),                  
          layers.Dense( nn3, activation=activation )
          ])
        # 
        model.compile(
          loss='mse',
          optimizer=opt,
          metrics=['mae','mse'])
    
        return model
    

    def build_model2(iunit, ounit, train_dataset,activation="sigmoid",loss='binary_crossentropy',metrics=['accuracy']):
        from tensorflow import keras
        from tensorflow.keras import layers

        model = keras.Sequential([
          layers.Dense(iunit, activation='relu', 
                       kernel_regularizer=keras.regularizers.l2(0.01), 
                       input_shape=[len(train_dataset.keys())]),
          layers.Dense(iunit, activation='relu', 
                       kernel_regularizer=keras.regularizers.l2(0.001)
                       ), 
          # layers.Dense(iunit, activation='relu', 
          #              kernel_regularizer=keras.regularizers.l2(0.001)
          #              ),                     
          # layers.Dense(iunit, activation='relu', 
          #              kernel_regularizer=keras.regularizers.l2(0.001)
          #              ),                    
          layers.Dense(ounit,activation=activation)
          ])
        # optimizer = tf.keras.optimizers.RMSprop(lr=0.01, rho=0.9, epsilon=None, decay=0.0)
        #optimizer = tf.keras.optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)
        #optimizer = tf.keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
        #optimizer = tf.keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
        optimizer = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        #optimizer = tf.keras.optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
        #optimizer = tf.keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
        
        model.compile(
          loss=loss, #'mse', #'binary_crossentropy'
          optimizer=optimizer,
          # 'mae', 'mse', 'mape','accuracy'
          metrics=metrics)
    
        return model  

    def training(EPOCHS=1000, model=None, normed_train_data=None, train_labels=None, verbose=None, monitor='val_loss',  patience=5 ):
        # tensorflow libraries
        from tensorflow import keras
        import tensorflow_docs as tfdocs
        import tensorflow_docs.modeling
        if model:
            history = model.fit(
                normed_train_data, train_labels,
                epochs=EPOCHS, validation_split = 0.2, verbose=verbose,
                callbacks=[tfdocs.modeling.EpochDots()]
                )
                # callbacks=[keras.callbacks.TensorBoard(log_dir=dir+r'\logs')])
            print("% : Training : end", model) 
            early_stop = keras.callbacks.EarlyStopping(monitor=monitor, patience=patience) #loss,accuracy,val_loss,val_accuracy
            early_history = model.fit(normed_train_data, train_labels, 
                            epochs=EPOCHS, validation_split = 0.2, verbose=verbose, 
                            callbacks=[early_stop, tfdocs.modeling.EpochDots()]
                            # callbacks=[keras.callbacks.TensorBoard(log_dir=dir+r'\logs')]
                            )  
        
            return model, history, early_history   


# In[2]:
# Load Library
from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier


# In[3]:
# Step1: Create data set
# X, y = make_moons(n_samples=10000, noise=.5, random_state=0)
import os
wdir = r"F:\CON_TRENSYS_elasticFabric\STUDY\MachineLearning\2p-length-fScale\BRKT"
file = r"DOE_2p-length-fScale.csv"
csv = os.path.join(wdir,file)
print(csv)

import pandas as pd
dataset = pd.read_csv(csv)
print(dataset.head())

column_names = dataset.columns
print(column_names[0])

if column_names[0]=="Point":
    dataset = dataset.drop([column_names[0]], axis=1)
    column_names = dataset.columns
print(dataset.head())

#In[4]
# Step2: Split the training test set
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
nOut = int(2)

train_dataset = dataset.sample(frac=0.8, random_state=0)
test__dataset = dataset.drop(train_dataset.index)

nPar = int(len(column_names)-int(nOut))
lstmin = [train_dataset[column_names[i]].min() for i in range(nPar)]
lstmax = [train_dataset[column_names[i]].max() for i in range(nPar)]

import numpy as np
Ranges = np.vstack((np.array([ lstmin, lstmax ])))  

train__input = train_dataset.iloc[:, :nPar]
train_labels = train_dataset.iloc[:, nPar:]
train_stats  = train__input.describe()
train_stats  = train_stats.transpose()    

test___input = test__dataset.iloc[:, :nPar]
test__labels = test__dataset.iloc[:, nPar:]

normed_train_data  = ML.norm( train__input,  train_stats   )
normed_test__data  = ML.norm( test___input,  train_stats   )

#In[5]
# Step 3: Fit a Decision Tree model as comparison
# clf = DecisionTreeClassifier()
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)
# accuracy_score(y_test, y_pred)

# Hyperparameter turning
# neurons in each layer
nn1     = [len(normed_train_data.keys())]
nn2     = [nn1[0]^i for i in range(1,10)]
nn3     = [nOut]

# learning algorithm parameters
lr      = [0.0001, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
decay   = [1e-6,1e-9,0]

# dropout and regularisation
l1      = [0, 0.01, 0.003, 0.001,0.0001]
l2      = [0, 0.01, 0.003, 0.001,0.0001]
dropout = [0, 0.1, 0.2, 0.3]

# dictionary summary
grid    = dict(nn1=nn1, nn2=nn2, nn3=nn3, lr=lr, decay=decay, l1=l1, l2=l2, dropout=dropout,)

from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
model = KerasRegressor(build_fn=ML.build_model, epochs=10, batch_size=20, verbose=2)

# from sklearn.model_selection import GridSearchCV, KFold
# grid = GridSearchCV(model, param_grid=grid, cv=KFold(3), refit=True)

from sklearn.model_selection import RandomizedSearchCV, KFold
grid = RandomizedSearchCV(estimator=model, cv=KFold(3), param_distributions=grid, verbose=20,  n_iter=100, n_jobs=-1)

rCV = grid.fit(normed_train_data, train_labels)
best_model = rCV.best_estimator_
best_para = best_model.get_params()
print(best_para)
nn1       = best_para['nn1'    ]
nn2       = best_para['nn2'    ]
nn3       = best_para['nn3'    ]
lr        = best_para['lr'     ]
decay     = best_para['decay'  ]
l1        = best_para['l1'     ]
l2        = best_para['l2'     ]
dropout   = best_para['dropout']

activation = 'relu'
model = ML.build_model(nn1=2, nn2=56, nn3 = 2, lr=0.01, decay=0., l1=0.01, l2=0.01, activation ='relu', dropout=0)

EPOCHS = 1000
verbose = 0
monitor = 'mse'
patience= 5
model, model_history, model_early_history = ML.training(EPOCHS, model, normed_train_data, train_labels, verbose, monitor, patience)

score = model.evaluate( normed_test__data, test__labels      )
print(f'score = {score}')


# Save Keras training result
file = r"trainingdata.h5"
_h5_ = os.path.join(wdir,file)
model.save(_h5_)    


#In[6]

# Step 4: Fit a Random Forest model, " compared to "Decision Tree model, accuracy go up by 5%
clf = RandomForestClassifier(n_estimators=100, max_features="auto",random_state=0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy_score(y_test, y_pred)


#In[7]
# Step 5: Fit a AdaBoost model, " compared to "Decision Tree model, accuracy go up by 10%
clf = AdaBoostClassifier(n_estimators=100)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy_score(y_test, y_pred)


#In[8]

# Step 6: Fit a Gradient Boosting model, " compared to "Decision Tree model, accuracy go up by 10%
clf = GradientBoostingClassifier(n_estimators=100)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy_score(y_test, y_pred)



# %%
