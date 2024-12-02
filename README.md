import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import *
import string
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
import seaborn
#from textstat.textstat import *
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer as VS
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
%matplotlib inline
from google.colab import drive
drive.mount('/content/drive')
hasoc = pd.read_csv('/content/drive/MyDrive/Bhumika/hasoc192021.csv')
election = pd.read_csv('/content/drive/MyDrive/Bhumika/11th_hour_political_tweets.csv',delimiter='|', on_bad_lines='skip')
elec_hate = pd.read_csv('/content/drive/MyDrive/Bhumika/negativesampann.csv')
elec_hate.head()
elec_hate.groupby('date_lu').Hate.agg(['sum'])
from matplotlib import pyplot as plt
_df_1['sum'].plot(kind='line', figsize=(8, 4), title='sum')
plt.gca().spines[['top', 'right']].set_visible(False)
elec_hate.groupby('date_lu').Hate.agg(['mean'])
from matplotlib import pyplot as plt
_df_3['mean'].plot(kind='line', figsize=(8, 4), title='mean')
plt.gca().spines[['top', 'right']].set_visible(False)
import nltk
import subprocess
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
# import the NLTK resources as usual
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import re
import string

punctuation = re.compile("[" + re.escape(string.punctuation) + "]")

# define the list of stopwords
stopwords = stopwords.words('english')

lemmatizer = WordNetLemmatizer()

def textProcess(text):
    res = []
    text_punc_remove = re.sub(punctuation,'',text)
    tokens = word_tokenize(text_punc_remove)
    for token in tokens:
        if token.lower() not in stopwords:
            lemmatized_word = lemmatizer.lemmatize(token)
            res.append(lemmatized_word)
    return ' '.join(res)
    # insert a new column which contains the processed text from column 'Comment'
elec_hate['processed_text'] = elec_hate['full_text'].apply(textProcess)
election['full_text']=election['full_text'].apply(str)
election['processed_text'] = election['full_text'].apply(textProcess)
# group the training dataset into non-hateful and hateful
groups = elec_hate.groupby('Hate')
non_hateful = groups.get_group(0)
hateful = groups.get_group(1)
# visualize the common vocabulary in non-hateful comments
from matplotlib import pyplot as plt
from wordcloud import WordCloud

combined_title = ' '.join(non_hateful['processed_text'])


# create a word cloud using the combined text
wordcloud_title = WordCloud(width = 1000, height = 1000,
                            background_color ='white',
                            min_font_size = 10).generate(combined_title)

# plot the WordCloud image
plt.figure(figsize=(10,10))
plt.imshow(wordcloud_title)
plt.axis('off')
plt.title('Frequent Vocabulary in Non-Hateful Comments', fontsize=20)
plt.tight_layout(pad=2)
plt.show()
 visualize the common vocabulary in hateful comments

from wordcloud import WordCloud

combined_title = ' '.join(hateful['processed_text'])


# create a word cloud using the combined text
wordcloud_title = WordCloud(width = 1000, height = 1000,
                            background_color ='white',
                            min_font_size = 10).generate(combined_title)

# plot the WordCloud image
plt.figure(figsize=(10,10))
plt.imshow(wordcloud_title)
plt.axis('off')
plt.title('Frequent Vocabulary in Hateful Comments', fontsize=20)
plt.tight_layout(pad=2)
plt.show()
# install pywaffle library to the working environment
# documentation: https://pywaffle.readthedocs.io/en/latest/index.html
!pip install pywaffle
from pywaffle import Waffle

# check the distribution of the non-hateful and hateful comments in the dataset
comments = elec_hate.Hate.value_counts()
percentage = (comments.values / elec_hate.shape[0]) * 100

figure = plt.figure(
    FigureClass=Waffle,
    rows=5,
    values=percentage,
    colors=['skyblue', 'red'],
    icons=['face-smile','face-angry'],
    icon_legend=False,
    legend={
        'labels': ['No-hateful', 'Hateful'],
        'loc': 'upper left',
        'bbox_to_anchor': (1, 1)
    },
    font_size=20,
    figsize=(10,6)
)

# add a title to the figure
plt.title('Distribution of Hateful and Non-Hateful Comments in the Annotated Sample of Election Data')
plt.show()
#Developing ML models using HASOC Data to identify Hate and Offensive speech.
#Four different methodologies are tried - Multinomial Naive Bayes, Logistic Rgression, Decision Tree and Random Forest
#make a copy of the original data for performing text processing using TF-IDF approach
hasoc_tfidf = hasoc.copy()
# insert a new column which contains the processed text from column 'Comment'
hasoc_tfidf['processed_text'] = hasoc_tfidf['Content'].apply(textProcess)
election['full_text']
elec_hate['processed_text'] = elec_hate['full_text'].apply(textProcess)
# split the dataset into training and test datasets
train, test = train_test_split(hasoc_tfidf, test_size=0.2, random_state=122)

# check the dimension of the training and test datasets
print('Dimension of training dataset: ', train.shape)
print('Dimension of test dataset: ', test.shape)
# segregate the feature and label of the train and test data
Xtrain = train['processed_text']
ytrain = train['Label']

Xtest = test['processed_text']
ytest = test['Label']
# feature transformation - TF-IDF to the processed text feature
from sklearn.feature_extraction.text import TfidfVectorizer

# initialize the TfidfVectorizer  object
vectorizer = TfidfVectorizer()

# vectorize and perform TF-IDF to the training data
Xtrain_vectorized = vectorizer.fit_transform(Xtrain)

# vectorize and perform TF*IDF to the test data
Xtest_vectorized = vectorizer.transform(Xtest)
election_vectorized = vectorizer.transform(election['processed_text'])
elec_hate_vectorized = vectorizer.transform(elec_hate['processed_text'])
#Multinomial Naive Bayes
from sklearn.naive_bayes import MultinomialNB
# instantiate a MultinomialNB() class
clf = MultinomialNB()

# train the model with training data processed using TF-IDF
mNB2 = clf.fit(Xtrain_vectorized, ytrain)
import seaborn as sns
ypred2 = mNB2.predict(Xtest_vectorized)

# confusion matrix
confusionMatrix = confusion_matrix(ytest,ypred2, normalize='true')

# visualize the confusion matrix in heatmap
plt.figure()
sns.heatmap(confusionMatrix, annot=True, cmap='BuPu', fmt='.4g')
plt.show()

# classification report
report = classification_report(ytest, ypred2)
print(report)
cm = confusion_matrix(ytest, ypred2)
print(cm)
ypred = mNB2.predict(Xtrain_vectorized)
sum(ypred)
sum(ytrain)
#train split and fit models
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

#model selection
from sklearn.metrics import confusion_matrix, accuracy_score

#Decision Tree
classifier_dt = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier_dt.fit(Xtrain_vectorized, ytrain)
y_pred_dt = classifier_dt.predict(Xtest_vectorized)
cm = confusion_matrix(ytest, y_pred_dt)
print(cm)
# confusion matrix
confusionMatrix = confusion_matrix(ytest,y_pred_dt, normalize='true')

# visualize the confusion matrix in heatmap
plt.figure()
sns.heatmap(confusionMatrix, annot=True, cmap='BuPu', fmt='.4g')
plt.show()

# classification report
report = classification_report(ytest, y_pred_dt)
print(report)

#Logistic Regression
classifier_lr = LogisticRegression(random_state = 0)
classifier_lr.fit(Xtrain_vectorized, ytrain)
y_pred_lr = classifier_lr.predict(Xtest_vectorized)
cm = confusion_matrix(ytest, y_pred_lr)
print(cm)
# confusion matrix
confusionMatrix = confusion_matrix(ytest,y_pred_lr, normalize='true')

# visualize the confusion matrix in heatmap
plt.figure()
sns.heatmap(confusionMatrix, annot=True, cmap='BuPu', fmt='.4g')
plt.show()

# classification report
report = classification_report(ytest, y_pred_lr)
print(report)

#Random Forest
classifier_rf = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier_rf.fit(Xtrain_vectorized, ytrain)
#Random Florest
y_pred_rf = classifier_rf.predict(Xtest_vectorized)
cm = confusion_matrix(ytest, y_pred_rf)
print(cm)
# confusion matrix
confusionMatrix = confusion_matrix(ytest,y_pred_rf, normalize='true')

# visualize the confusion matrix in heatmap
plt.figure()
sns.heatmap(confusionMatrix, annot=True, cmap='BuPu', fmt='.4g')
plt.show()

# classification report
report = classification_report(ytest, y_pred_rf)
print(report)

#Comparison of Model Accuracy on Test Sample
rf_score = accuracy_score(ytest, y_pred_rf)
lr_score = accuracy_score(ytest, y_pred_lr)
dt_score = accuracy_score(ytest, y_pred_dt)
np_score = accuracy_score(ytest, ypred2)

print('Random Forest Accuracy: ', str(rf_score))
print('Logistic Regression Accuracy: ', str(lr_score))
print('Decision Tree Accuracy: ', str(dt_score))
print('Naive Bayes Accuracy: ', str(np_score))

#Scoring Logistic Model on Annotated Sample of Election Data
yelechate = elec_hate['Hate']
#Random Florest
yelechate_lr = classifier_lr.predict(elec_hate_vectorized)
cm = confusion_matrix(yelechate, yelechate_lr)
print(cm)
# confusion matrix
confusionMatrix = confusion_matrix(yelechate, yelechate_lr, normalize='true')

# visualize the confusion matrix in heatmap
plt.figure()
sns.heatmap(confusionMatrix, annot=True, cmap='BuPu', fmt='.4g')
plt.show()

# classification report
report = classification_report(yelechate, yelechate_lr)
print(report)
sum(yelechate)
sum(yelechate_lr)

#Scoring Random Forest Model on Annotated Sample of Election Data
yelechate_rf = classifier_rf.predict(elec_hate_vectorized)
# confusion matrix
confusionMatrix = confusion_matrix(yelechate,yelechate_rf, normalize='true')

# visualize the confusion matrix in heatmap
plt.figure()
sns.heatmap(confusionMatrix, annot=True, cmap='BuPu', fmt='.4g')
plt.show()

# classification report
report = classification_report(yelechate,yelechate_rf)
print(report)
sum(yelechate_rf)
#Finding word patterns based on Annotation in sample election data
# group the training dataset into non-hateful and hateful
groups = elec_hate.groupby('Hate')
non_hateful = groups.get_group(0)
hateful = groups.get_group(1)
# visualize the common vocabulary in non-hateful comments
from matplotlib import pyplot as plt
from wordcloud import WordCloud

combined_title = ' '.join(non_hateful['processed_text'])


# create a word cloud using the combined text
wordcloud_title = WordCloud(width = 1000, height = 1000,
                            background_color ='white',
                            min_font_size = 10).generate(combined_title)

# plot the WordCloud image
plt.figure(figsize=(10,10))
plt.imshow(wordcloud_title)
plt.axis('off')
plt.title('Frequent Vocabulary in Non-Hateful Comments in Scored Sample of Election Data', fontsize=20)
plt.tight_layout(pad=2)
plt.show()

# visualize the common vocabulary in hateful comments

from wordcloud import WordCloud

combined_title = ' '.join(hateful['processed_text'])


# create a word cloud using the combined text
wordcloud_title = WordCloud(width = 1000, height = 1000,
                            background_color ='white',
                            min_font_size = 10).generate(combined_title)

# plot the WordCloud image
plt.figure(figsize=(10,10))
plt.imshow(wordcloud_title)
plt.axis('off')
plt.title('Frequent Vocabulary in Hateful Comments in Scored Sample of Election Data', fontsize=20)
plt.tight_layout(pad=2)
plt.show()

#Using Logistic Regression Model to identify word patterns in sample election data
elec_hate['pred'] = yelechate_lr
# group the training dataset into non-hateful and hateful
groups = elec_hate.groupby('pred')
non_hateful = groups.get_group(0)
hateful = groups.get_group(1)
# visualize the common vocabulary in non-hateful comments
from matplotlib import pyplot as plt
from wordcloud import WordCloud

combined_title = ' '.join(non_hateful['processed_text'])


# create a word cloud using the combined text
wordcloud_title = WordCloud(width = 1000, height = 1000,
                            background_color ='white',
                            min_font_size = 10).generate(combined_title)

# plot the WordCloud image
plt.figure(figsize=(10,10))
plt.imshow(wordcloud_title)
plt.axis('off')
plt.title('Frequent Vocabulary in Non-Hateful Comments in Scored Sample of Election Data', fontsize=20)
plt.tight_layout(pad=2)
plt.show()
# visualize the common vocabulary in hateful comments

from wordcloud import WordCloud

combined_title = ' '.join(hateful['processed_text'])


# create a word cloud using the combined text
wordcloud_title = WordCloud(width = 1000, height = 1000,
                            background_color ='white',
                            min_font_size = 10).generate(combined_title)

# plot the WordCloud image
plt.figure(figsize=(10,10))
plt.imshow(wordcloud_title)
plt.axis('off')
plt.title('Frequent Vocabulary in Hateful Comments in Scored Sample of Election Data', fontsize=20)
plt.tight_layout(pad=2)
plt.show()

#Scoring the full election data using the Logistic Regression Model; f:inding word patterns based on model estimates.
yelection_lr = classifier_lr.predict(election_vectorized)
election['pred'] = yelection_lr
election.head()
election1 = election[['last_updated', 'pred', 'neg']].copy()
election1.to_csv('/content/drive/MyDrive/Bhumika/election_scored2.csv')
# group the training dataset into non-hateful and hateful
groups = election.groupby('pred')
non_hateful = groups.get_group(0)
hateful = groups.get_group(1)
# visualize the common vocabulary in non-hateful comments
from matplotlib import pyplot as plt
from wordcloud import WordCloud

combined_title = ' '.join(non_hateful['processed_text'])


# create a word cloud using the combined text
wordcloud_title = WordCloud(width = 1000, height = 1000,
                            background_color ='white',
                            min_font_size = 10).generate(combined_title)

# plot the WordCloud image
plt.figure(figsize=(10,10))
plt.imshow(wordcloud_title)
plt.axis('off')
plt.title('Frequent Vocabulary in Non-Hateful Comments in Scored Election Data', fontsize=20)
plt.tight_layout(pad=2)
plt.show()
# visualize the common vocabulary in hateful comments

from wordcloud import WordCloud

combined_title = ' '.join(hateful['processed_text'])


# create a word cloud using the combined text
wordcloud_title = WordCloud(width = 1000, height = 1000,
                            background_color ='white',
                            min_font_size = 10).generate(combined_title)

# plot the WordCloud image
plt.figure(figsize=(10,10))
plt.imshow(wordcloud_title)
plt.axis('off')
plt.title('Frequent Vocabulary in Hateful Comments in Scored Election Data', fontsize=20)
plt.tight_layout(pad=2)
plt.show()
