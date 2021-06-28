
from pprint import pprint
import pandas as pd
import os

import gensim
import gensim.corpora as corpora

import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

import urllib.request

path = os.path.abspath(os.path.dirname(os.getcwd())) + '\\data'

info_article = pd.read_csv(path+'\\info\\info_article_o_8_n.csv')
info_issue = pd.read_csv(path+'\\info\\info_issue_o_8.csv')
info_year = pd.read_csv(path+'\\info\\info_year_o_8.csv')

# keywords

keywords_list = info_article['keyword'].values
keywords_list_r = []
for i_keyword in keywords_list:

    i_keyword = eval(i_keyword)

    i_keyword_r = []
    for i_i_keyword in i_keyword:
        if i_i_keyword != None:
            i_keyword_r.append(i_i_keyword)

    keywords_list_r.append(i_keyword_r)

keywords_dict = corpora.Dictionary(keywords_list_r)

corpus = [keywords_dict.doc2bow(i_keyword) for i_keyword in keywords_list_r]
# print(corpus[:4])

# must run with python console, cannot use jupyter notebook
lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                       id2word=keywords_dict,
                                       num_topics=10,
                                       random_state=0)

pprint(lda_model.print_topics())

my_pic = gensimvis.prepare(lda_model, corpus, keywords_dict)

pyLDAvis.save_html(my_pic, path+'\\picture\\example.html')


