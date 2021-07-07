# based on keywords
# find the best number of topics
# make some modifications on LDA_1

from pprint import pprint
import pandas as pd
import os

from nltk import PorterStemmer

import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel

import matplotlib.pyplot as plt

import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

path = os.path.abspath(os.path.dirname(os.getcwd())) + '\\data'

info_article = pd.read_csv(path+'\\info\\info_article_o_8_n.csv')
info_issue = pd.read_csv(path+'\\info\\info_issue_o_8.csv')
info_year = pd.read_csv(path+'\\info\\info_year_o_8.csv')


# keywords

keywords_list = info_article['keyword'].values
keywords_list_r = []
porter = PorterStemmer()

for i_keyword in keywords_list:

    i_keyword = eval(i_keyword)

    i_keyword_r = []
    for i_i_keyword in i_keyword:

        if i_i_keyword != None:
            i_i_keyword = porter.stem(i_i_keyword)
            i_keyword_r.append(i_i_keyword)

    keywords_list_r.append(i_keyword_r)

keywords_dict = corpora.Dictionary(keywords_list_r)

corpus = [keywords_dict.doc2bow(i_keyword) for i_keyword in keywords_list_r]



# find the best number of topics
def compute_coherence_values(corpus_r, list_r, dictionary,
                             start = 10, end=20, step=2):

    coherence_values = []
    model_list = []

    for num_topics in range(start,end,step):
        model = gensim.models.LdaMulticore(corpus=corpus_r,
                                           id2word=dictionary,
                                           num_topics=num_topics,
                                           random_state=0)
        model_list.append(model)

        coherence_model = CoherenceModel(model=model,
                                         texts=list_r,
                                         dictionary=dictionary,
                                         coherence='c_v')
        coherence_values.append(coherence_model.get_coherence())

    return model_list, coherence_values


model_list, coherence_values = compute_coherence_values(corpus_r=corpus,
                                                        list_r= keywords_list_r,
                                                        dictionary=keywords_dict,
                                                        start=2, end=50, step=1)

plt.close()
x = range(2,50,1)
plt.plot(x, coherence_values)
plt.xlabel("Number of Topics")
plt.ylabel("Coherence score")
plt.savefig(path+'\\picture\\coherence_score_3.png')

# 0.6854245467586704
# number of topics : 23
# coherence_values.index(max(coherence_values)) 21

lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                       id2word=keywords_dict,
                                       num_topics=23,
                                       random_state=0)

pprint(lda_model.print_topics())

my_pic = gensimvis.prepare(lda_model, corpus, keywords_dict)

pyLDAvis.save_html(my_pic, path+'\\picture\\lda_3.html')
