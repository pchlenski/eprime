import os
from subprocess import _args_from_interpreter_flags
import nltk
import re
import sys
import gensim
import platform

from nltk.tree import Tree, AbstractParentedTree, ParentedTree
from nltk.tag import map_tag
import nltk_tgrep

from nltk.corpus.reader.util import *
from nltk.corpus.reader.api import *
from BracketParseCorpusReader import BracketParseCorpusReader

import numpy as np

import cPickle as pickle

from pattern.en import conjugate

be = ['am','are','is',
     'was','were','was',
     'be',
     'being',
     'been']

taste  = ['taste', 'tastes', 'tasted',  'tasting' ]
smell = ['smell', 'smells', 'smelled', 'smelling']
sound = ['sound', 'sounds', 'sounded', 'sounding']
feel  = ['feel',  'feels',  'felt',    'feeling' ]
look  = ['look',  'looks',  'looked',  'looking' ]
seem  = ['seem',  'seems',  'seemed',  'seeming' ]
allv = taste + smell + sound + feel + look + seem + be

taste2 = []
smell2 = []
sound2 = []
feel2 = []
look2 = []
seem2 = []
be2 = []
#Making parsed text into nltk.Tree
reader = BracketParseCorpusReader('','Gold2.txt')

w2vec_path = '/Users/sarahgross/Desktop/Fall 2017/Language and Computation I/Final Project/GoogleNews-vectors-negative300.bin'
model = gensim.models.KeyedVectors.load_word2vec_format(w2vec_path, binary=True, limit=1000000)

for sent in reader.parsed_sents():
    sent = ParentedTree.convert(sent)
    #see if words we care about are in sentence
    if list(set(sent.leaves()) & set(allv)) != []:
        copWords = list(set(sent.leaves()) & set(allv))
        for word in copWords:
            nouns = []
            #End goal: get [[nouns], verb, [adjectives]] for each verb in list
            #Do so by first finding the verb of interest and its position in the tree
            position = sent.leaves().index(word)
            treeposition = sent.leaf_treeposition(position)
            #Want the VP to find the Adjective Predicate which is its child
            newTree = sent[treeposition[:-2]]
            #Search for Adjective(s) below Adjective Predicate
            adj = nltk_tgrep.tgrep_nodes(newTree, 'JJ|VBN|VBG|JJR > ADJP-PRD')
            vb = sent[treeposition[:-1]]
            # To find the relevant Noun Phrase, we go up the tree until reaching the lowest sentence node, then back down to NP-SBJ.*
            s = sent[treeposition[:-1]].parent()
            while ('S' not in s.label()):
                s = s.parent()
                try: s.label()
                except AttributeError:
                    break
            #Move one level above VP to find Subject of Verb
            Ns = nltk_tgrep.tgrep_nodes(s, 'NP-SBJ|NP-SBJ-1|NP-SBJ-2')
            for N in Ns:
                nouns = nouns + nltk_tgrep.tgrep_nodes(N, 'NN|PRP|NNS|EX|WDT')
            #Moving from lists of parented trees to lists of words
            noun = [x.leaves() for x in nouns]
            noun = [single for [single] in noun]
            noun = list(set(noun))
            adj = [i.leaves() for i in adj]
            adj = [single for [single] in adj]
            adj = list(set(adj))
            #Appending to appropriate verb list
            if word in smell:
                smell2.append([noun, vb, adj])
            elif word in taste:
                taste2.append([noun, vb, adj])
            elif word in sound:
                sound2.append([noun, vb, adj])
            elif word in feel:
                feel2.append([noun, vb, adj])
            elif word in look:
                look2.append([noun, vb, adj])
            elif word in seem:
                seem2.append([noun, vb, adj])
            else:
                be2.append([noun, vb, adj])

# Now, parsing the test sentences--similar to the process above
def test_sentences():
    sentences = []
    original = []
    reader = BracketParseCorpusReader('','test sentences parsed 2.txt')
    for sent in reader.parsed_sents():
        sent = ParentedTree.convert(sent)
        #see if words we care about are in sentence
        if list(set(sent.leaves()) & set(be)) != []:
            copWords = list(set(sent.leaves()) & set(be))
            for word in copWords:
                nouns = []
                negative = False
                #End goal: get [[nouns], verb, [adjectives], isNegated] for each verb in list
                #Do so by first finding the verb of interest and its position in the tree
                position = sent.leaves().index(word)
                treeposition = sent.leaf_treeposition(position)
                #Want the VP to find the Adjective Predicate which is its child
                newTree = sent[treeposition[:-2]]
                #Search for Adjective(s) below Adjective Predicate
                adj = nltk_tgrep.tgrep_nodes(newTree, 'JJ|VBN|VBG > ADJP')
                if 'not' in newTree.leaves() or 'n\'t' in newTree.leaves():
                    negative = True
                vb = sent[treeposition[:-1]]
                # To find the relevant Noun Phrase, we go up the tree until reaching the lowest sentence node, then back down to NP-SBJ.*
                s = sent[treeposition[:-1]].parent()
                while ('S' not in s.label()):
                    s = s.parent()
                    try: s.label()
                    except AttributeError:
                        break
                #Move one level above VP to find Subject of Verb
                Ns = nltk_tgrep.tgrep_nodes(s, 'NP')
                for N in Ns:
                    nouns = nouns + nltk_tgrep.tgrep_nodes(N, 'NN|PRP|NNS|EX|WD|NNP')
                #Moving from lists of parented trees to lists of words
                noun = [x.leaves() for x in nouns]
                noun = [single for [single] in noun]
                noun = list(set(noun))
                adj = [i.leaves() for i in adj]
                adj = [single for [single] in adj]
                adj = list(set(adj))
                #Because our test sentences are all simple, can do this 
                adjp = nltk_tgrep.tgrep_nodes(sent, 'ADJP')[0].leaves()
                np = nltk_tgrep.tgrep_nodes(sent, 'NP')[0].leaves()
                sentences.append([noun, vb, adj, negative, np, adjp])
                original.append(" ".join(sent.leaves()))
    return rewrite_sentences(sentences), original

# A devilishly useful function. Give it a list of words and it will
# return a single vector that is the average of their respective
# vectors. 
# If it can't find a word, or if it finds an empty list, it will use
# a vector of zeroes as a placeholder.
def get_average_vector(word_list):
	# Initialize output vector
	output_vector = []

	if word_list == []:
		# Inevitable...
		output_vector = np.zeros(300)

	else:
		# Initialize vector list
		all_vectors = []

		for word in word_list:
			try: 
				all_vectors += [model[word]]
			except KeyError:
				# Also inevitable...
				all_vectors += [np.zeros(300)]

		output_array  = np.array(all_vectors)
		output_vector = np.mean(output_array, axis=0)

	return output_vector

def get_characteristic_vector(corpus):
	mean_noun_vectors = []
	mean_adj_vectors  = []

	for element in corpus:
		mean_noun_vectors += [get_average_vector(element[0])]
		mean_adj_vectors  += [get_average_vector(element[2])]

	final_noun_array = np.array(mean_noun_vectors)
	final_noun_mean  = np.mean(final_noun_array, axis=0)
		# Constrain to np.array
	final_adj_array  = np.array(mean_adj_vectors)
	final_adj_mean   = np.mean(final_adj_array,  axis=0)
		# Constrain to np.array

	return final_noun_mean, final_adj_mean

# Get vectors from relevant corpora:
be_vector_n, be_vector_a    = get_characteristic_vector(be2)
taste_vector_n, taste_vector_a = get_characteristic_vector(taste2)
smell_vector_n, smell_vector_a = get_characteristic_vector(smell2)
sound_vector_n, sound_vector_a = get_characteristic_vector(sound2)
feel_vector_n, feel_vector_a  = get_characteristic_vector(feel2)
look_vector_n, look_vector_a  = get_characteristic_vector(look2)
seem_vector_n, seem_vector_a  = get_characteristic_vector(seem2)


all_vectors_n  = [taste_vector_n, smell_vector_n, sound_vector_n,
				  feel_vector_n,  look_vector_n,  seem_vector_n, be_vector_n]

all_vectors_a  = [taste_vector_a, smell_vector_a, sound_vector_a,
				  feel_vector_a,  look_vector_a,  seem_vector_a, be_vector_a]


sub_keys = ['taste','smell','sound','feel','look','seem','be']
## To get word to vec vectors from verbs, average through every case of the verb (i.e. smell, smelled, smelling, smells)
w2v = [get_average_vector(taste), get_average_vector(smell), get_average_vector(sound), get_average_vector(feel), get_average_vector(look), get_average_vector(seem), get_average_vector(be)]

# A function to find the best verb to substitute for the copula
def find_best_sub(sentence):

	# Get mean vectors for nouns and adjectives in the sentence
	noun_vector = get_average_vector(sentence[0])
	adj_vector  = get_average_vector(sentence[2])

	# Compute cosine similiarities with canonical vectors
	noun_similarities = model.wv.cosine_similarities(noun_vector, all_vectors_n)
	adj_similarities  = model.wv.cosine_similarities(adj_vector, all_vectors_a)
	noun_w2v = model.wv.cosine_similarities(noun_vector, w2v)
	adj_w2v = model.wv.cosine_similarities(adj_vector, w2v)
	# Find the most similar vectors
	best_noun     = np.argmax(noun_similarities)
	best_noun_sim = noun_similarities[best_noun]
	best_adj      = np.argmax(adj_similarities)
	best_adj_sim  = adj_similarities[best_adj]
	w2v_noun      = np.argmax(noun_w2v)
	w2v_noun_sim  = noun_w2v[w2v_noun]
	w2v_adj       = np.argmax(adj_w2v)
	w2v_adj_sim   = adj_w2v[w2v_adj]

	# Pick the most similar one:
	# Check that one of the values is better than the threshold
        sims = [best_noun_sim, best_adj_sim, w2v_noun_sim, w2v_adj_sim]
        indexes = [best_noun, best_adj, w2v_noun, w2v_adj]
        highest = np.argmax(sims)
        best_sub = sub_keys[indexes[highest]]

	return best_sub


# The conjugator won't handle negations 
negations = {'taste'  : 'does not taste', 'tastes'  : 'does not taste',
			 'tasted' : 'did not taste',  'tasting' : 'not tasting',
			 # Can't avoid the copula for negating present
			 # participles, it looks like... something like the
			 # sentence 'he is being good' has two copulas, and
			 # only 'being' can be replaced in our scheme.
			 'smell'  : 'does not smell', 'smells'  : 'does not smell',
			 'smelled': 'did not smell',  'smelling': 'not smelling',
			 'sound'  : 'does not sound', 'sounds'  : 'does not sound',
			 'sounded': 'did not sound',  'sounding': 'not sounding',
			 'feel'   : 'does not feel',  'feels'   : 'does not feel',
			 'felt'   : 'did not feel',   'feeling' : 'not feeling',
			 'look'   : 'does not look',  'looks'   : 'does not look',
			 'looked' : 'did not look',   'looking' : 'not looking',
			 'seem'   : 'does not seem',  'seems'   : 'does not seem',
			 'seemed' : 'did not seem',   'seeming' : 'not seeming'}



# This function replaces the copula with the preferred sub
def replace_verb(sentence, sub, negated=False):
	if sub == 'be':
		output = sentence
                verb = sentence[1].leaves()
    
	else:
		tense = sentence[1].label()
		conj_verb = conjugate(sub, tense)

		if negated == True:
			output = (sentence[0], (negations[conj_verb], tense), sentence[2])

		else: 
			output = (sentence[0], (conj_verb, tense), sentence[2])
		verb = [output[1][0]]

        sentence2 = " ".join(sentence[4] + verb + sentence[5])
	return sentence2



# This function calls everything else
def rewrite_sentences(sentences):
        new_sentences = []
	for sentence in sentences:
                #print sentence
		sub = find_best_sub(sentence)
		new_sentences.append(replace_verb(sentence, sub, sentence[3]))
	
        return new_sentences


sentences, olds = test_sentences()
for i in range(len(sentences)):
    print olds[i] + " -> " + sentences[i]
