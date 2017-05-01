
# Optimized Approach using Machine Learning on Credit Card Fraud Dataset
**Via performance metrics and paralell processing**

*Author: Jie Hu,  jie.hu.ds@gmail.com*

------------

## 0. Abstract

This is a project in which I use performance metrics to optimize machine learning algorithm to predict fraud transactions in [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/dalpozz/creditcardfraud/kernels).

The goal of this research is to find out most significant features to predict whether a transaction in the dataset is committed to fraud. The structure of this article is:
- Data Wrangling, in which I modify NA values and remove outliers
- Feature Selecting, in which I create some features I think important to predict fraud
- Training and tuning machine learning, in which I use sklearn to train 4 different models and compare their performance matrices, including precision, recall and f1 score, and plot ROC curves to compare the models
- Final part, in which I select Naive Bayes as my best model
- Conclusion

## 1. Data Wrangling

Firstly, load the dataset:


```python
# Packages
import pandas as pd
import numpy as np
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True) 

```


<script>requirejs.config({paths: { 'plotly': ['https://cdn.plot.ly/plotly-latest.min']},});if(!window.Plotly) {{require(['plotly'],function(plotly) {window.Plotly=plotly;});}}</script>



```python
df = pd.read_csv("creditcard.csv")
df.shape
```




    (284807, 31)




```python
df.columns
```




    Index([u'Time', u'V1', u'V2', u'V3', u'V4', u'V5', u'V6', u'V7', u'V8', u'V9',
           u'V10', u'V11', u'V12', u'V13', u'V14', u'V15', u'V16', u'V17', u'V18',
           u'V19', u'V20', u'V21', u'V22', u'V23', u'V24', u'V25', u'V26', u'V27',
           u'V28', u'Amount', u'Class'],
          dtype='object')




```python
sum(df.Class), round(sum(df.Class)*1.0/df.shape[0],5)
```




    (492, 0.00173)



This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.173% of all transactions.

Since the dataset is processed and now there's no NA values, so no need to clean the dataset.

On the other hand, the dataset is processed by PCA, so I would assume all outliers, if any, would not be meaningless.

## 2. Feature Selecting

Then, to begin with, I use all features and, under each algorithm I tune, use KBest to select features by their scores and compare the recall/precision rate.


```python
features_list = np.array(df.columns)[:-1]
lable_column = np.array(df.columns)[-1]

features, labels = df[features_list], df[lable_column]
```

One more step before start fitting is re-scale the data by sklearn.MinMaxScaler, because some of the algorithms I will implement might require re-scale features to avoid skewed distance and biased result.


```python
from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler()
features = min_max_scaler.fit_transform(features)
```


```python
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

data_all = SelectKBest(f_classif, k='all').fit(features,labels)
```


```python
# fit_transform(X, y)
import operator

scores = data_all.scores_
score_dict = {}
for ii in range(len(features_list) - 1):
    score_dict[features_list[ii+1]] = round(scores[ii],2)

sorted_score_dict = sorted(score_dict.items(), key=operator.itemgetter(1), reverse=False)

```

Then I plot the features sorted by their scores:


```python
features_lst = [item[0] for item in sorted_score_dict]
scores = [item[1] for item in sorted_score_dict]

layout = go.Layout(
    title='Feature Scores Rank',
    yaxis=dict(
        title='Features'
    ),
    xaxis=dict(
        title='Scores'
    ),
    height=600, width=600
)



trace = go.Bar(
            x=scores,
            y=features_lst,
            orientation = 'h',
            marker=dict(
            color=['#006775']*25+['#ff3f49'] *4)
        )


fig = go.Figure(data=[trace], layout=layout)
iplot(fig, show_link=False)

```


<div id="259ae191-3055-4c27-b30e-bee86b99ad23" style="height: 600px; width: 600px;" class="plotly-graph-div"></div><script type="text/javascript">require(["plotly"], function(Plotly) { window.PLOTLYENV=window.PLOTLYENV || {};window.PLOTLYENV.BASE_URL="https://plot.ly";Plotly.newPlot("259ae191-3055-4c27-b30e-bee86b99ad23", [{"marker": {"color": ["#006775", "#006775", "#006775", "#006775", "#006775", "#006775", "#006775", "#006775", "#006775", "#006775", "#006775", "#006775", "#006775", "#006775", "#006775", "#006775", "#006775", "#006775", "#006775", "#006775", "#006775", "#006775", "#006775", "#006775", "#006775", "#ff3f49", "#ff3f49", "#ff3f49", "#ff3f49"]}, "y": ["V23", "V24", "V26", "V16", "V27", "V14", "V25", "Amount", "V1", "V28", "V9", "V21", "V20", "V22", "V7", "V3", "V6", "V10", "V2", "V19", "V5", "V12", "V8", "V4", "V17", "V11", "V13", "V15", "V18"], "type": "bar", "orientation": "h", "x": [0.18, 2.05, 3.12, 5.08, 5.65, 5.95, 14.85, 25.9, 43.25, 88.05, 112.55, 115.0, 344.99, 465.92, 543.51, 2393.4, 2592.36, 2746.6, 2955.67, 3584.38, 5163.83, 6999.36, 10349.61, 11014.51, 11443.35, 14057.98, 20749.82, 28695.55, 33979.17]}], {"width": 600, "yaxis": {"title": "Features"}, "title": "Feature Scores Rank", "xaxis": {"title": "Scores"}, "height": 600}, {"linkText": "Export to plot.ly", "showLink": false})});</script>



```python
features_lst.reverse() # will use this later
```

From the plot, it seems 7 is a great cut-off to select features because more than 7 features will not have high score. However, here I will keep this plot as a reference, I will explore more about how the recall rate and precision rate will be changed and then compare all these plot to select best combination.

## 3. Training and Tuning Machine Learning

Before training, I will use validation to see how algorithms generalized to the overall data outside training data. Without validation, there's pretty high risk of overfitting.

After that, I run each model independently and tune for the best performance of each type of model.

### 3.1 About Tuning Models

It's very important to tune the model, not only because each model, there might be a lot of combination of parameters and hence make big influence on performance of our model, but also, we can have better chance to see how model can better match the data in an optimization process.

Without tuning the model, we might save a lot of time and get a good result, but there's always a question "can we do better" over the head. And if the model get bad prediction when there's new data coming, we even don't know where the problem is. So tune the model might cost sometime, but it did save time in future for further exploration and better improved the performance and robustness of model.


### 3.2 Naive Bayes

Similarly, I tune the model for each K from 1 to 25, and plot the performance matrices:


```python
from sklearn.model_selection import train_test_split
labels = df['Class']
features = df.drop('Class',1)
features_train, features_test, labels_train, labels_test = train_test_split(features, 
                                                                            labels, 
                                                                            random_state=1,
                                                                            stratify = labels,
                                                                            test_size=0.8)
```


```python
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
import numpy as np

# classifier test function
# extract data with k best features
# use stratified sampling iteratively to get precision / recall / f1 / accuracy
# return the 4 average scores

def classifer_tester(classifier, features_lst, parameters, k, iterations=100):
    
    precision = []
    recall = []
    accuracy = []
    f1score = []
    
    ### Extract data with k best features
    
    k_best_features_list = features_lst[0:k]
    # print k_best_features_list
    ### Extract features and labels from dataset
    features = features_train[k_best_features_list]
    labels = labels_train
    
    
    ### Iteratively to get average scores
    for ii in range(iterations):
        
        
        X_train, X_test, Y_train, Y_test = train_test_split(features, 
                                                            labels, 
                                                            random_state=ii*10,
                                                            stratify = labels)
        grid_search = GridSearchCV(classifier, parameters)
        grid_search.fit(X_train, Y_train)
        predictions = grid_search.predict(X_test)
        precision.append(precision_score(Y_test, predictions))
        recall.append(recall_score(Y_test, predictions))
        accuracy.append(accuracy_score(Y_test, predictions))
        f1score.append(f1_score(Y_test, predictions))
    
    precision_mean = np.array(precision).mean()
    recall_mean = np.array(recall).mean()
    accuracy_mean = np.array(accuracy).mean()
    f1score_mean = np.array(f1score).mean()
    
    return precision_mean, recall_mean, accuracy_mean, f1score_mean

```


```python
import datetime
from multiprocessing import Pool
from sklearn.naive_bayes import GaussianNB

def GB_classifer_tester(k):
    return classifer_tester(GaussianNB(), 
                            features_lst=features_lst, 
                            parameters={}, 
                            k=k,
                            iterations = 30)
if __name__ == '__main__':
    pool = Pool()
    Ks = range(1,30)
    t1 = datetime.datetime.now()
    results = pool.map(GB_classifer_tester, Ks)
    precisions =[]
    recalls = []
    accuracies = []
    f1_scores = []
    for i in range(len(results)):
        precisions.append(results[i][0])
        recalls.append(results[i][1])
        accuracies.append(results[i][2])
        f1_scores.append(results[i][3])
    print(datetime.datetime.now() - t1)
```

    0:00:53.056705



```python
Ks = list(range(1,29))

trace_precision = go.Scatter(
    x = Ks,
    y = precisions,
    mode = 'lines',
    name = 'precisions'
)

trace_recalls = go.Scatter(
    x = Ks,
    y = recalls,
    mode = 'lines',
    name = 'recalls'
)

trace_accuracies = go.Scatter(
    x = Ks,
    y = accuracies,
    mode = 'lines',
    name = 'accuracies'
)

trace_f1scores = go.Scatter(
    x = Ks,
    y = f1_scores,
    mode = 'lines',
    name = 'f1scores'
)


layout = dict(title = 'Perform Metrics by Number of Best Features (Naive Bayes)',
              xaxis = dict(title = 'K, Number of Best Features'),
              yaxis = dict(title = 'Scores'),
              )

fig = go.Figure(data=[trace_precision, trace_recalls, trace_accuracies, trace_f1scores], layout=layout)
iplot(fig, show_link=False)


```


<div id="30a8cc16-687f-4040-a32c-04bcc512be3e" style="height: 525px; width: 100%;" class="plotly-graph-div"></div><script type="text/javascript">require(["plotly"], function(Plotly) { window.PLOTLYENV=window.PLOTLYENV || {};window.PLOTLYENV.BASE_URL="https://plot.ly";Plotly.newPlot("30a8cc16-687f-4040-a32c-04bcc512be3e", [{"y": [0.3797324448956567, 0.3640021748503711, 0.3914408279772058, 0.506024839719198, 0.2263514633945374, 0.2413818480006803, 0.1307615375830041, 0.13944494956290843, 0.11658300818547987, 0.11468025765869154, 0.09554526786230746, 0.0812830217368235, 0.08114423734678831, 0.08013750834327497, 0.07412720915900098, 0.07389437177148249, 0.06480492431611645, 0.06174791599777229, 0.061943584748477436, 0.05977530404560494, 0.057432133201211814, 0.05794245778371908, 0.057714718358998565, 0.06235313728332881, 0.058534969571345394, 0.059070441437273295, 0.059016787561318555, 0.05932987891958314, 0.056767419008935956], "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28], "type": "scatter", "name": "precisions", "mode": "lines"}, {"y": [0.24533333333333338, 0.24533333333333338, 0.2773333333333333, 0.4786666666666667, 0.6906666666666667, 0.7346666666666669, 0.7453333333333333, 0.7706666666666667, 0.7600000000000001, 0.7560000000000001, 0.7493333333333335, 0.7560000000000001, 0.7533333333333335, 0.7453333333333334, 0.7453333333333334, 0.7440000000000001, 0.7413333333333334, 0.7346666666666668, 0.7346666666666668, 0.7346666666666668, 0.7346666666666668, 0.7346666666666668, 0.7346666666666668, 0.784, 0.7773333333333334, 0.7666666666666668, 0.7666666666666668, 0.7680000000000001, 0.7640000000000001], "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28], "type": "scatter", "name": "recalls", "mode": "lines"}, {"y": [0.9979729887882408, 0.9979285162558811, 0.9979800107670344, 0.998244505301594, 0.9952952742082718, 0.9954520984013294, 0.9908246143763314, 0.9912295484867634, 0.9894600098307704, 0.9893172295953001, 0.9870912623177212, 0.98457037193081, 0.984589097207593, 0.9845399433560377, 0.9831987454064556, 0.9831683168316832, 0.9807504154670785, 0.9799171406502353, 0.9799779977997801, 0.9792219647496665, 0.9783418767408656, 0.9784776349975424, 0.9783957119116167, 0.9788521405332024, 0.9776092502867308, 0.9781078107810782, 0.9780867448446975, 0.9781639866114273, 0.9772487887086582], "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28], "type": "scatter", "name": "accuracies", "mode": "lines"}, {"y": [0.2958078601971339, 0.29136546674206437, 0.3232585589906918, 0.48982189179697, 0.34055863304035566, 0.36278462565545244, 0.22235970912246106, 0.23601913118370935, 0.20207003579605073, 0.19906537689869866, 0.169427228496089, 0.1467423644471194, 0.1464631860890087, 0.14467929027128046, 0.13480767793126713, 0.13440257860491453, 0.11916901126896629, 0.11390488460976225, 0.11423632356822197, 0.11053903491269192, 0.10652000998086407, 0.10737727520570359, 0.10698809006385544, 0.1154809520362179, 0.10884393783925238, 0.10966126869784266, 0.10956893580490598, 0.11011995510408926, 0.10565591973756414], "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28], "type": "scatter", "name": "f1scores", "mode": "lines"}], {"yaxis": {"title": "Scores"}, "xaxis": {"title": "K, Number of Best Features"}, "title": "Perform Metrics by Number of Best Features (Naive Bayes)"}, {"linkText": "Export to plot.ly", "showLink": false})});</script>


The accuracy is an average score to show how much percentage we get the right prediction. As it's not suitable for skewed features, here I add precision and recall matrices to evaluate.


Precision is how much probability we get a sample with true if it's tested positive. By bayes prior and post probability:

$$Precision = P(T | +) = \frac{P(+|T)}{P(+|T)+P(+|F)}$$

In other words, precision is a measure of result relevancy, while recall is a measure of how many truly relevant results are returned:

$$Recall = \frac{P(+|T)}{P(+|T)+P(-|T)}$$

- Higher precission means, with transaction identified as fraud by this model, we have higher correct rate
- Higher recall means, if the transaction is fraud, then we have higher chance to identify correctly

Here we can see:

The higher recall might make precision lower, which causes a lot of non-fraud transactions to be commited fraud. So I will keep a balance here and select K = 4


Now test in test set:


```python
k_best_features_list = features_lst[0:4]
# print k_best_features_list
### Extract features and labels from dataset
features_nb_train = features_train[k_best_features_list]
labels_nb_train = labels_train
features_nb_test = features_test[k_best_features_list]
labels_nb_test = labels_test

clf = GaussianNB()

clf.fit(features_nb_train, labels_nb_train)
predictions = clf.predict(features_nb_test)

precision = precision_score(labels_nb_test, predictions)
recall = recall_score(labels_nb_test, predictions)
accuracy = accuracy_score(labels_nb_test, predictions)
f1score = f1_score(labels_nb_test, predictions)

print ('Accuracy: %s' % "{:,.2f}".format(round(accuracy, 2)) )
print ('Precision: %s' % "{:,.2f}".format(round(precision, 2)))
print ('Recall   : %s' % "{:,.2f}".format(round(recall, 2)))
print ('F1 score:  %s' % "{:,.2f}".format(round(f1score, 2)))
```

    Accuracy: 1.00
    Precision: 0.57
    Recall   : 0.58
    F1 score:  0.57


ROC Curve


```python
from sklearn import metrics
y_score = clf.fit(features_nb_train, labels_nb_train).predict_proba(features_nb_test)[:, 1]
fpr_rf, tpr_rf, _ = metrics.roc_curve(labels_nb_test, y_score)

```


```python
import matplotlib.pyplot as plt
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_rf, tpr_rf, label='RF + LR')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.show()
```


![png](output_30_0.png)


Problems of models:
Assumption of independence between transactions is not necessarily meet.

### 3.3 Decision Tree

The process is similar, but here we should set the parameters of Decision Tree and use GridSearch and visualization to get the best performance.


```python
from multiprocessing import Pool
from sklearn import tree

# due to speed, I only check several combinations
parameters = {'criterion': ['entropy','gini'],
               'max_depth': [3, 5, 10],
               'min_samples_leaf': [1, 10]}

def DT_classifer_tester(k):
    print(k)
    return classifer_tester(tree.DecisionTreeClassifier(), 
                            features_lst=features_lst, 
                            parameters=parameters, 
                            k=k,
                            iterations = 30)
if __name__ == '__main__':
    
    t1 = datetime.datetime.now()
    pool = Pool()
    Ks = range(1,30)
    results = pool.map(DT_classifer_tester, Ks)
    precisions =[]
    recalls = []
    accuracies = []
    f1scores = []
    for i in range(len(results)):
        precisions.append(results[i][0])
        recalls.append(results[i][1])
        accuracies.append(results[i][2])
        f1scores.append(results[i][3])
    print(datetime.datetime.now() - t1)
```

    1
    2
    4
    6
    5
    3
    7
    8
    9
    10
    11
    12
    13
    14
    15
    16
    17
    18
    19
    20
    21
    22
    23
    24
    25
    26
    27
    28
    29
    0:44:23.212240



```python
Ks = list(range(1,29))

trace_precision = go.Scatter(
    x = Ks,
    y = precisions,
    mode = 'lines',
    name = 'precisions'
)

trace_recalls = go.Scatter(
    x = Ks,
    y = recalls,
    mode = 'lines',
    name = 'recalls'
)

trace_accuracies = go.Scatter(
    x = Ks,
    y = accuracies,
    mode = 'lines',
    name = 'accuracies'
)

trace_f1scores = go.Scatter(
    x = Ks,
    y = f1scores,
    mode = 'lines',
    name = 'f1scores'
)


layout = dict(title = 'Perform Metrics by Number of Best Features (Decision Tree)',
              xaxis = dict(title = 'K, Number of Best Features'),
              yaxis = dict(title = 'Scores'),
              )

fig = go.Figure(data=[trace_accuracies, trace_precision, trace_f1scores, trace_recalls], layout=layout)
iplot(fig, show_link=False)
```


<div id="4bac8964-ecaa-46b6-87c8-c5923524e637" style="height: 525px; width: 100%;" class="plotly-graph-div"></div><script type="text/javascript">require(["plotly"], function(Plotly) { window.PLOTLYENV=window.PLOTLYENV || {};window.PLOTLYENV.BASE_URL="https://plot.ly";Plotly.newPlot("4bac8964-ecaa-46b6-87c8-c5923524e637", [{"y": [0.9984387800482172, 0.9984458020270108, 0.9984130327926406, 0.9986353954544387, 0.998782857009105, 0.9988179669030731, 0.9988249888818667, 0.9988998899889986, 0.9988858460314114, 0.9989069119677922, 0.9988952086698029, 0.9988788240526177, 0.9988905273506071, 0.998874142733422, 0.9989069119677924, 0.9989092526273902, 0.9988952086698029, 0.9988718020738242, 0.998874142733422, 0.9988905273506072, 0.9988718020738242, 0.998857758116237, 0.9988694614142264, 0.9990637361608499, 0.9990403295648711, 0.9990848020972308, 0.9990543735224584, 0.9990730987992414, 0.9990590548416541], "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28], "type": "scatter", "name": "accuracies", "mode": "lines"}, {"y": [0.6766121841121843, 0.7155891330891333, 0.6786050061050063, 0.6946433802122658, 0.7213060180323608, 0.7173753642248729, 0.7206598589723842, 0.7462752757799759, 0.7434991275999719, 0.7456563320518093, 0.746336073981955, 0.7328658525952128, 0.7370730952847211, 0.7404903906930314, 0.7383703983714976, 0.7382613061708836, 0.7405694301149612, 0.7336559006199246, 0.7365135337011094, 0.7584505396601575, 0.7523179073858842, 0.7477314790883577, 0.74141284915834, 0.7631648310117416, 0.7704064179567496, 0.7800366306469572, 0.7570108710707869, 0.771426239250648, 0.7696286845725416], "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28], "type": "scatter", "name": "precisions", "mode": "lines"}, {"y": [0.31959031708989233, 0.3080489170354214, 0.3051042979399321, 0.5054110151201516, 0.5883249912884096, 0.615897346822193, 0.6212864265184261, 0.6420473549441414, 0.6332895776116272, 0.6453243946130449, 0.6393992306513208, 0.636416140651186, 0.643614360773047, 0.6312192340809976, 0.6529375773518091, 0.6555549496061351, 0.6465090867015991, 0.6395517862094262, 0.6381166644046036, 0.6319430981041677, 0.626012958518806, 0.6206301030343468, 0.629300799452792, 0.7180360460230336, 0.7051041199080345, 0.717847815348821, 0.7175068637200179, 0.7202713401704756, 0.7127434969270374], "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28], "type": "scatter", "name": "f1scores", "mode": "lines"}, {"y": [0.21600000000000003, 0.20133333333333336, 0.2013333333333334, 0.40399999999999997, 0.5079999999999999, 0.548, 0.5546666666666668, 0.5733333333333334, 0.5626666666666668, 0.5786666666666666, 0.5693333333333332, 0.5733333333333334, 0.5786666666666667, 0.5640000000000001, 0.5960000000000001, 0.5986666666666667, 0.5826666666666667, 0.5773333333333334, 0.5733333333333334, 0.5533333333333333, 0.5466666666666667, 0.5426666666666667, 0.5573333333333335, 0.6880000000000001, 0.664, 0.6786666666666666, 0.6920000000000001, 0.6853333333333332, 0.6746666666666667], "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28], "type": "scatter", "name": "recalls", "mode": "lines"}], {"yaxis": {"title": "Scores"}, "xaxis": {"title": "K, Number of Best Features"}, "title": "Perform Metrics by Number of Best Features (Decision Tree)"}, {"linkText": "Export to plot.ly", "showLink": false})});</script>


From the plot we can see the model reach optimal when K = 27. 
Here I use this to train and tune the Decision Tree model again to find the best combination of parameters:


```python
k_best_features_list = features_lst[0:26]

features_dt_train = features_train[k_best_features_list]
labels_dt_train = labels_train
features_dt_test = features_test[k_best_features_list]
labels_dt_test = labels_test

clf = tree.DecisionTreeClassifier()

# parameters to tune the best Decision Tree model
parameters = {'criterion': ['entropy','gini'],
               'max_depth': [3, 5, 10],
               'min_samples_leaf': [1, 10]}

grid_search = GridSearchCV(clf, parameters, n_jobs = -1)
grid_search.fit(features_dt_train, labels_dt_train)
predictions = grid_search.predict(features_dt_test)

precision = precision_score(labels_nb_test, predictions)
recall = recall_score(labels_nb_test, predictions)
accuracy = accuracy_score(labels_nb_test, predictions)
f1score = f1_score(labels_nb_test, predictions)

print ('Accuracy: %s' % "{:,.2f}".format(round(accuracy, 2)) )
print ('Precision: %s' % "{:,.2f}".format(round(precision, 2)))
print ('Recall   : %s' % "{:,.2f}".format(round(recall, 2)))
print ('F1 score:  %s' % "{:,.2f}".format(round(f1score, 2)))
```

    Accuracy: 1.00
    Precision: 0.85
    Recall   : 0.69
    F1 score:  0.76


And the best parameters are:


```python
grid_search.best_estimator_
```




    DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=3,
                max_features=None, max_leaf_nodes=None,
                min_impurity_split=1e-07, min_samples_leaf=10,
                min_samples_split=2, min_weight_fraction_leaf=0.0,
                presort=False, random_state=None, splitter='best')



Here we care Recall and Precision the most because we want our model to increase probability to correct identify fraud with the transactions which are truely fraud and probability to get right fraud identification when we have positive test result. Thus, K = 27 will be a better choice.

And ROC curve:


```python
clf = tree.DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=3,
            max_features=None, max_leaf_nodes=None,
            min_impurity_split=1e-07, min_samples_leaf=10,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            presort=False, random_state=None, splitter='best')
y_score = clf.fit(features_dt_train, labels_dt_train).predict_proba(features_dt_test)[:, 1]
fpr_rf, tpr_rf, _ = metrics.roc_curve(labels_dt_test, y_score)

plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_rf, tpr_rf, label='RF + LR')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.show()
```


![png](output_40_0.png)


### 3.4 Support Vector Machine

The 3rd type of model I want to use is SVM, because we have multi-dimensional dataset and big dataset. SVM's hyper-plane have its advantages when we have enough features to train the model.


```python
import warnings
warnings.filterwarnings('ignore')

from sklearn import svm
parameters = {'C': [1, 10, 100, 1000],
               'gamma': [0.001, 0.0001],
               'kernel': ['rbf']}

def svm_classifer_tester(k):
    # print(k)
    # print(datetime.datetime.now())
    return classifer_tester(svm.SVC(), 
                            features_lst=features_lst, 
                            parameters=parameters, 
                            k=k,
                            iterations = 30)

if __name__ == '__main__':
    
    t1 = datetime.datetime.now()
    pool = Pool()
    Ks = list(range(1,30))
    results = pool.map(svm_classifer_tester, Ks)
    precisions =[]
    recalls = []
    accuracies = []
    f1scores = []
    for i in range(len(results)):
        precisions.append(results[i][0])
        recalls.append(results[i][1])
        accuracies.append(results[i][2])
        f1scores.append(results[i][3])
    print("Total Time:")
    print(datetime.datetime.now() - t1)
```

    Total Time:
    0:57:31.499286



```python
Ks = list(range(1,29))

trace_precision = go.Scatter(
    x = Ks,
    y = precisions,
    mode = 'lines',
    name = 'precisions'
)

trace_recalls = go.Scatter(
    x = Ks,
    y = recalls,
    mode = 'lines',
    name = 'recalls'
)

trace_accuracies = go.Scatter(
    x = Ks,
    y = accuracies,
    mode = 'lines',
    name = 'accuracies'
)

trace_f1scores = go.Scatter(
    x = Ks,
    y = f1scores,
    mode = 'lines',
    name = 'f1scores'
)


layout = dict(title = 'Perform Metrics by Number of Best Features (SVM)',
              xaxis = dict(title = 'K, Number of Best Features'),
              yaxis = dict(title = 'Scores'),
              )

fig = go.Figure(data=[trace_accuracies, trace_precision, trace_f1scores, trace_recalls], layout=layout)
iplot(fig, show_link=False)


```


<div id="07ccc349-0567-46c2-abe5-6bf3eef1941c" style="height: 525px; width: 100%;" class="plotly-graph-div"></div><script type="text/javascript">require(["plotly"], function(Plotly) { window.PLOTLYENV=window.PLOTLYENV || {};window.PLOTLYENV.BASE_URL="https://plot.ly";Plotly.newPlot("07ccc349-0567-46c2-abe5-6bf3eef1941c", [{"type": "scatter", "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28], "y": [0.9982702525571703, 0.9982702525571703, 0.9982749338763661, 0.9984153734522384, 0.9988109449242795, 0.9989911757133159, 0.9989935163729136, 0.9990660768204478, 0.9990684174800457, 0.9990590548416541, 0.9990684174800457, 0.9991807691407436, 0.9991994944175266, 0.9991948130983309, 0.999192472438733, 0.9991901317791352, 0.9991948130983309, 0.9992018350771246, 0.9991901317791352, 0.999192472438733, 0.9991737471619501, 0.9991175713316011, 0.9991035273740139, 0.9991620438639608, 0.9991503405659714, 0.999157362544765, 0.9991597032043629, 0.9991737471619502, 0.9991643845235586], "mode": "lines", "name": "accuracies"}, {"type": "scatter", "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28], "y": [0.36666666666666664, 0.36666666666666664, 0.43333333333333335, 0.7159271284271284, 0.7487324992975147, 0.7998995867792139, 0.7891779395878199, 0.8164074514996577, 0.8077262763559709, 0.8027021136733078, 0.7973588832708499, 0.8132345539400999, 0.8165676539064949, 0.8095719408937765, 0.8114239682198021, 0.8080853121094069, 0.8079766075035627, 0.8107983876811781, 0.8072142894608569, 0.8068097524773725, 0.8021729826872849, 0.8244611662548128, 0.8297044284118589, 0.8368361472498974, 0.8356985899178321, 0.8371318349792571, 0.8395634569635109, 0.8432480802476764, 0.8403746501276451], "mode": "lines", "name": "precisions"}, {"type": "scatter", "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28], "y": [0.02820512820512821, 0.02820512820512821, 0.03333333333333333, 0.2677632910174787, 0.5909969917620569, 0.6635334329739732, 0.6693045241986322, 0.6922514313562538, 0.6986311118873874, 0.6966423855701893, 0.7030255821652982, 0.7471041262115123, 0.7538327219393538, 0.755314885486565, 0.7535346251365542, 0.7537106079417497, 0.7559388851336838, 0.7574718538804757, 0.7543824200112628, 0.7552167370494527, 0.7494165699311471, 0.7157620033540137, 0.7078276594155446, 0.7302859825553034, 0.7262800149640454, 0.729014569578014, 0.7289112888346387, 0.733543287319313, 0.7310948995914781], "mode": "lines", "name": "f1scores"}, {"type": "scatter", "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28], "y": [0.014666666666666666, 0.014666666666666666, 0.017333333333333333, 0.1706666666666667, 0.49333333333333335, 0.5733333333333334, 0.5866666666666667, 0.6066666666666667, 0.6199999999999999, 0.6199999999999999, 0.6333333333333333, 0.7, 0.7066666666666667, 0.7146666666666668, 0.7093333333333334, 0.7120000000000002, 0.7160000000000001, 0.7160000000000001, 0.7133333333333334, 0.7146666666666667, 0.708, 0.64, 0.6240000000000001, 0.6533333333333334, 0.648, 0.6506666666666666, 0.6493333333333333, 0.6546666666666667, 0.6519999999999999], "mode": "lines", "name": "recalls"}], {"title": "Perform Metrics by Number of Best Features (SVM)", "xaxis": {"title": "K, Number of Best Features"}, "yaxis": {"title": "Scores"}}, {"showLink": false, "linkText": "Export to plot.ly"})});</script>


For SVM, we reach optimal at K = 18. Under the balance between precision and recall, we care more about recall because it's dangerous to miss fraud transactions. And though precision rate might be little lower than K > 18, more innocent transaction will be flagged fraud, we still can apply other technics to identify them. However, if we miss more, we might have less chance to catch them. Now let's train the SVM again with K=18 and find the best model:


```python
k_best_features_list = features_lst[0:18]

features_svm_train = features_train[k_best_features_list]
labels_svm_train = labels_train
features_svm_test = features_test[k_best_features_list]
labels_svm_test = labels_test

clf = svm.SVC()

# parameters to tune the best SVM model
parameters = {'C': [1, 10, 100, 1000],
               'gamma': [0.001, 0.0001],
               'kernel': ['rbf']}

grid_search = GridSearchCV(clf, parameters, n_jobs=-1)
grid_search.fit(features_svm_train, labels_svm_train)
predictions = grid_search.predict(features_svm_test)

precision = precision_score(labels_svm_test, predictions)
recall = recall_score(labels_svm_test, predictions)
accuracy = accuracy_score(labels_svm_test, predictions)
f1score = f1_score(labels_svm_test, predictions)

print ('Accuracy: %s' % "{:,.2f}".format(round(accuracy, 2)) )
print ('Precision: %s' % "{:,.2f}".format(round(precision, 2)))
print ('Recall   : %s' % "{:,.2f}".format(round(recall, 2)))
print ('F1 score:  %s' % "{:,.2f}".format(round(f1score, 2)))
```

    Accuracy: 1.00
    Precision: 0.84
    Recall   : 0.80
    F1 score:  0.82


The best parameters are:


```python
grid_search.best_estimator_
```




    SVC(C=100, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape=None, degree=3, gamma=0.0001, kernel='rbf',
      max_iter=-1, probability=False, random_state=None, shrinking=True,
      tol=0.001, verbose=False)




```python
# ROC Curve, probability in SVC should be set to True to enable proba method
clf = svm.SVC(C=100, gamma=0.0001, kernel='rbf', probability=True)
y_score = clf.fit(features_svm_train, labels_svm_train).predict_proba(features_svm_test)[:, 1]
fpr_rf, tpr_rf, _ = metrics.roc_curve(labels_svm_test, y_score)

plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_rf, tpr_rf, label='RF + LR')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.show()
```


![png](output_48_0.png)


SVM model does have the best ROC curve among the 3 types of models we tried.

### 3.5 Adaboost (Based on Decision Tree)


```python
from sklearn.ensemble import AdaBoostClassifier


def adb_classifer_tester(k):

    return classifer_tester(AdaBoostClassifier(), 
                            features_lst=features_lst, 
                            parameters={}, 
                            k=k,
                            iterations = 30)
if __name__ == '__main__':
    
    t1 = datetime.datetime.now()
    pool = Pool()
    Ks = list(range(1,30))
    results = pool.map(adb_classifer_tester, Ks)
    precisions =[]
    recalls = []
    accuracies = []
    f1scores = []
    for i in range(len(results)):
        precisions.append(results[i][0])
        recalls.append(results[i][1])
        accuracies.append(results[i][2])
        f1scores.append(results[i][3])
    print(datetime.datetime.now() - t1)
```

    0:47:59.325914



```python
Ks = list(range(1,29))

trace_precision = go.Scatter(
    x = Ks,
    y = precisions,
    mode = 'lines',
    name = 'precisions'
)

trace_recalls = go.Scatter(
    x = Ks,
    y = recalls,
    mode = 'lines',
    name = 'recalls'
)

trace_accuracies = go.Scatter(
    x = Ks,
    y = accuracies,
    mode = 'lines',
    name = 'accuracies'
)

trace_f1scores = go.Scatter(
    x = Ks,
    y = f1scores,
    mode = 'lines',
    name = 'f1scores'
)


layout = dict(title = 'Perform Metrics by Number of Best Features (Adaboost)',
              xaxis = dict(title = 'K, Number of Best Features'),
              yaxis = dict(title = 'Scores'),
              )

fig = go.Figure(data=[trace_accuracies, trace_precision, trace_f1scores, trace_recalls], layout=layout)
iplot(fig, show_link=False)


```


<div id="bd33d0d3-b4f7-41ae-a034-9e4e80e803de" style="height: 525px; width: 100%;" class="plotly-graph-div"></div><script type="text/javascript">require(["plotly"], function(Plotly) { window.PLOTLYENV=window.PLOTLYENV || {};window.PLOTLYENV.BASE_URL="https://plot.ly";Plotly.newPlot("bd33d0d3-b4f7-41ae-a034-9e4e80e803de", [{"type": "scatter", "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28], "y": [0.9984481426866088, 0.9984130327926406, 0.9983919668562596, 0.9986868899655921, 0.9987711537111156, 0.9988554174566391, 0.9988483954778454, 0.998836692179856, 0.998857758116237, 0.9988507361374434, 0.998839032839454, 0.9989467031809561, 0.9989630877981414, 0.9989513845001519, 0.9989560658193476, 0.9989092526273903, 0.9989349998829666, 0.9989209559253794, 0.9989115932869882, 0.9988928680102052, 0.9989560658193476, 0.9989747910961306, 0.9989607471385433, 0.9989701097769348, 0.9989747910961306, 0.9989888350537178, 0.9990145823092945, 0.9990099009900988, 0.9990169229688923], "mode": "lines", "name": "accuracies"}, {"type": "scatter", "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28], "y": [0.6780611980611982, 0.6488655788655789, 0.6342772042772042, 0.6976221493218397, 0.7231908585499245, 0.7658003298444476, 0.7645044744270753, 0.7458952648658531, 0.7577161997913445, 0.7567235935756262, 0.7523164526105703, 0.7832810627313589, 0.7832137984109984, 0.7817616966771632, 0.776448106286409, 0.7657883490404689, 0.7838581944346517, 0.7701189991614007, 0.7799594822202172, 0.7739270236302139, 0.7846095784447179, 0.8016877476668498, 0.7903162300002387, 0.7968853087082325, 0.8028326097134147, 0.8044663556544364, 0.8109429191666035, 0.807062263501892, 0.8118591938677078], "mode": "lines", "name": "precisions"}, {"type": "scatter", "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28], "y": [0.34040671251832944, 0.3248118338805446, 0.3102719784143804, 0.5511239697371814, 0.5830282867849146, 0.6047009723402891, 0.6024711964261363, 0.6074654682653597, 0.6148924503542906, 0.6115510598874604, 0.6056485599005599, 0.6499594707893127, 0.6564818376339401, 0.6523523559556507, 0.6586325871807287, 0.6349467809948148, 0.6419932096915686, 0.6391965291809004, 0.628613216687091, 0.6242520597910374, 0.6553520679705228, 0.654542475830077, 0.651991023397159, 0.6527401141286957, 0.6503848966758858, 0.658038749754125, 0.670261936725404, 0.670810984813303, 0.6732099955086668], "mode": "lines", "name": "f1scores"}, {"type": "scatter", "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28], "y": [0.23200000000000004, 0.22000000000000006, 0.20933333333333337, 0.464, 0.49600000000000005, 0.5053333333333334, 0.504, 0.5173333333333333, 0.5226666666666666, 0.5186666666666667, 0.5133333333333333, 0.5626666666666666, 0.572, 0.5666666666666667, 0.5760000000000001, 0.548, 0.552, 0.5533333333333333, 0.5333333333333333, 0.5306666666666666, 0.5679999999999998, 0.5626666666666666, 0.5626666666666666, 0.56, 0.5533333333333333, 0.5613333333333334, 0.5760000000000001, 0.5773333333333333, 0.58], "mode": "lines", "name": "recalls"}], {"title": "Perform Metrics by Number of Best Features (Adaboost)", "xaxis": {"title": "K, Number of Best Features"}, "yaxis": {"title": "Scores"}}, {"showLink": false, "linkText": "Export to plot.ly"})});</script>


It seems the more features, the better adaboost will perform. Here we can see when K= 28, we reach optimal. 


```python
k_best_features_list = features_lst[0:28]

features_adb_train = features_train[k_best_features_list]
labels_adb_train = labels_train
features_adb_test = features_test[k_best_features_list]
labels_adb_test = labels_test

clf = AdaBoostClassifier()

# parameters to tune the best SVM model
parameters = {}

grid_search = GridSearchCV(clf, parameters, n_jobs=-1)
grid_search.fit(features_adb_train, labels_adb_train)
predictions = grid_search.predict(features_adb_test)

precision = precision_score(labels_adb_test, predictions)
recall = recall_score(labels_adb_test, predictions)
accuracy = accuracy_score(labels_adb_test, predictions)
f1score = f1_score(labels_adb_test, predictions)

print ('Accuracy: %s' % "{:,.2f}".format(round(accuracy, 2)) )
print ('Precision: %s' % "{:,.2f}".format(round(precision, 2)))
print ('Recall   : %s' % "{:,.2f}".format(round(recall, 2)))
print ('F1 score:  %s' % "{:,.2f}".format(round(f1score, 2)))
```

    Accuracy: 1.00
    Precision: 0.80
    Recall   : 0.67
    F1 score:  0.73



```python
grid_search.best_estimator_
```




    AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None,
              learning_rate=1.0, n_estimators=50, random_state=None)




```python
# ROC Curve, probability in SVC should be set to True to enable proba method
clf = AdaBoostClassifier() # the parameters are the same as default
y_score = clf.fit(features_adb_train, labels_adb_train).predict_proba(features_adb_test)[:, 1]
fpr_rf, tpr_rf, _ = metrics.roc_curve(labels_adb_test, y_score)

plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_rf, tpr_rf, label='RF + LR')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.show()
```


![png](output_56_0.png)


## 4. Final Discussion

Now let's put together and select the best model and feature list.

Score Type|Naive Bayes| Decision Tree | SVM|Adaboost
------------|------------|------------|------------|------------
Accuracy|1|1|1|1
Precision|0.57|0.85|0.84|0.80
Recall|0.58|0.69|0.80|0.67
F1 score|0.57|0.79|0.81|0.73

I notice best model for out of sample test here is when K = 21, we use SVM with parameters:

SVC(C=100, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape=None, degree=3, gamma=0.0001, kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)

The best performance in testing set is:

Precision: 0.841823
Recall: 0.796954
Accuracy: 0.999390
F1 Score: 0.818774


## 5. Conclusion

In this report I firstly summarize the dataset, use visualization to find out interesting correlations among fraud and other features. Then I train 3 different models, and finally find SVM as my best model and number of features K = 21.

This is a quantative analysis and can only be a reference for commitment. The real procedure of fraud commitment is quite complex.

In future, to improve the accuracy of the model, I think there're some ways we can try:
- Given more detailed dataset, more features might have risk of overfitting, but more data can possibly provide more informaiton we need
- Given more data with fraud so it will be easier to catch up with more significant predictors.


```python

```
