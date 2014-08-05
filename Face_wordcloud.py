# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

from pytagcloud import create_tag_image, make_tags
from pytagcloud.lang.counter import get_tag_counts
import os
from nltk.corpus import stopwords
import nltk
import pandas as pd

# <codecell>


# <codecell>

accidents=pd.read_csv('Accidents.txt',sep='|')
accidents.columns=[x.lower() for x in accidents.columns]

# <codecell>

Face_accidents=accidents[accidents['ug_location']=='FACE']

# <codecell>

Face_narrative=' '.join([w for w in Face_accidents['narrative']]).lower()

# <codecell>

Face_tokenized=nltk.word_tokenize(Face_narrative)

# <codecell>

remove_list=['employee'] + stopwords.words('english')

# <codecell>

Face_cleanup=[w for w in Face_tokenized if w not in remove_list and  w.isalpha()]

# <codecell>

Face_tagged=nltk.pos_tag(Face_cleanup)

# <codecell>

Face_noun=[w for (w,tag) in Face_tagged if tag == 'NN']
Face_presentverb=[w for (w,tag) in Face_tagged if tag == 'VBG']
Face_verb_noun=[w for (w,tag) in Face_tagged if tag in ['NN','VBD','VBN','VBG']]

# <codecell>

fdist=nltk.FreqDist(Face_cleanup)
Nfdist=nltk.FreqDist(Face_noun)
Vfdist=nltk.FreqDist(Face_presentverb)
VNfdist=nltk.FreqDist(Face_verb_noun)

# <codecell>

Face_noun_Top_elements=Nfdist.keys()[:50]
Face_verb_Top_elements=Nfdist.keys()[:50]
Face_nounverb_Top_elements=VNfdist.keys()[:50]

Face_verb_sorted=' '.join([w for w in Face_presentverb if w in Face_verb_Top_elements])
Face_noun_sorted=' '.join([w for w in Face_noun if w in Face_noun_Top_elements])
Face_verbnoun_sorted=' '.join([w for w in Face_verb_noun if w in Face_nounverb_Top_elements])

# <codecell>

Face_Top_elements=fdist.keys()[:50]
Face_sorted=' '.join([w for w in Face_cleanup if w in Face_Top_elements])

# <codecell>

tags = make_tags(get_tag_counts(Face_sorted), maxsize=120)

create_tag_image(tags, 'Face_all.png', size=(900, 600), fontname='Lobster' )

#import webbrowser
#webbrowser.open('MRH_noun.png') # see results

# <codecell>


