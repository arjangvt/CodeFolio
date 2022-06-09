"""
Here I used Naive Bayes model to classify documents.
To try non conventional data I chose use Chauser's 
The Canterberry Tales and Shakespeare's seleceted 
works. The data were downloaded from 
https://www.gutenberg.org/.

The processed data also can be found in this
repository "library.zip"

Written by: Arjang Fahim
Date: 5/20/2022 

If you have any questions or suggestions, please 
email me at "arjangvt at gmail dot com"
"""

import os
import random
import nltk
import re

from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords


def get_classes(library_folder):
	classes_list = os.listdir(os.path.join(library_folder))
	return classes_list


def get_book_from_file(filename, utf=False):
	text = ""
	if utf:
		f = open(filename,encoding="utf8")
	else:
		f = open(filename)

	for line in f:
		line = line.strip()
		text += line + "\n"
	return text


def sentences_tokenize(dataset):
	sentences = sent_tokenize(text=dataset, language='english')
	return sentences

def words_tokenize(dataset, remove_stop_words=False, add_custom_stop_words = {}):
	word_list = word_tokenize(text=dataset, language='english')

	# Remove Stop words from dataset
	filtered_words = []

	if remove_stop_words:
		stop_words = set(stopwords.words('english'))
		
		stop_words.update(add_custom_stop_words)
		#print(stop_words)

		for w in word_list:
			if w not in stop_words:
				filtered_words.append(w)

	return word_list, filtered_words

def find_features(document, word_features):
    words = set(document)

    features = {}
    for w in word_features:
        features[w] = (w in words)
        
    return features

def main():
	library_folder = r"your\path\Library"


	update_stop_words = {',', ';', 'The', '.', '?', 'I', 'And', '!', "'d", "'s", 
                    	 ':', '\'', '’', '”', '[', ']', '\'\'', '``', '*', '<', '>', ')', '(', 
                    	 '*', "_", "-", "'S", "To", "to", "thou", "what", "But", "but", "Thy",
                    	 "thy", "that","That", "What", 'Shall', 'shall', "May", 'may', "of", "Of", "when",
                    	 "When", "for", "For","Thee", "thee", "well", "Well", "As", "as", "You", "you", "never",
                    	   "Never", "Also", "also", "must", "Must", "--", "by", "By", "'ll", "if", "If",
                    	    "or", "Or", "it", "It"}
	documents = []
	all_words = []
	categories = get_classes(library_folder)
	
	# Documents here are the sentences that are labeled
	print("Creating documents ...")
	for category in categories:
		for book in os.listdir(os.path.join(library_folder, category)):
			book_path = os.path.join(library_folder, category, book)
			book_content = get_book_from_file(book_path, utf=True)
			book_content = re.sub('[\\s][*][a-z]*', "", book_content)
			book_content = re.sub("[0-9*]", "", book_content)

			wordlist, filtered_whole_book_words = words_tokenize(book_content, True, update_stop_words)
			all_words += [word.lower() for word in filtered_whole_book_words]

			sentences = sentences_tokenize(book_content)
			for sentence in sentences:
				wordlist, filtered_sentence_words = words_tokenize(sentence, True, update_stop_words)
				documents.append((list(filtered_sentence_words),category))	

	random.shuffle(documents)

	all_words = nltk.FreqDist(all_words)
	word_features = list(all_words.keys())[:4000]

	print("Creating featuresets ...")
	featuresets = [(find_features(rev, word_features), category) for (rev, category) in documents]
	
	train_set =  featuresets[:2900]
	test_set =  featuresets[2900:]	

	print("Creating NaiveBayes model ...")
	classifier = nltk.NaiveBayesClassifier.train(train_set)

	# test the accuray

	print('Accuracy:', nltk.classify.accuracy(classifier=classifier, gold=test_set) * 100)

	classifier.show_most_informative_features(30)


	who_wrote_this_sentence =  "Yea, and a case to put it into."
	who_wrote_this_sentence += "But speak you this with a sad brow, or do you play the flouting Jack, to tell us Cupid is a good hare-finder,"
	who_wrote_this_sentence += "and Vulcan a rare carpenter? Come, in what key shall a man take you, to go in the song?" 


	guess = classifier.classify(find_features(who_wrote_this_sentence, word_features))

	if (guess == "shakespeare"):
		print("I think Shakespeare wrote this! I know I am right!")
	else:
		print("I did my best my guess was wrong ")


	who_wrote_this_sentence = "The Nine Worthies, who at our day survive in the"
	who_wrote_this_sentence += "Seven Champions of Christendom. The Worthies"
	who_wrote_this_sentence += "were favourite subjects for representation at popular"
	who_wrote_this_sentence += "festivals or in masquerades."

	if (guess == "chauser"):
		print("I think Chauser wrote this! I know I am right!")
	else:
		print("I did my best my guess was wrong ")

main()