"""
A simple code can be used for the processing of any text file.
I used this code for the preparation of the data downloaded 
from https://www.gutenberg.org/ . 

I named function like get_book_from etc. Beacuse I wrote 
this code for analysis of the books from https://www.gutenberg.org/.
But generally it can be used for any text file.
Feel free to customize it as much as you want to meet your
needs. 


Written by: Arjang Fahim
Date: 5/20/2022

Note: This is just a hobby project. Use the code 
with your own responibility. 
If you have any questions or suggestions, please 
email me at "arjangvt at gmail dot com" 
"""


import os
import nltk

from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords

base_booklib_dir = r"your\path\Library"

def get_book_from_file(filename, utf=False):
	text = ""
	if utf:
		f = open(filename,encoding="utf8")
	else:
		f = open(filename)

	for line in f:
		text += line
	return text

def word_count(words, most_common_words=0):
	all_words = nltk.FreqDist(words)
	#print(all_words.N)
	#word_features = list(all_words.keys())[:3000]
	#print(word_features)
	if most_common_words > 0:
		print(all_words.most_common(most_common_words))
		print(all_words['love'])


def most_n_frequent_words():
	pass


def words_tokenize(remove_stop_words=False, add_custom_stop_words = {}):
	word_list = word_tokenize(text=dataset, language='english')

	# Remove Stop words from dataset
	filtered_words = []

	if remove_stop_words:
		stop_words = set(stopwords.words('english'))
		
		stop_words.update(add_custom_stop_words)

		for w in word_list:
			if w not in stop_words:
				filtered_words.append(w)

	return word_list, filtered_words

def sentences_tokenize(dataset):
	sentences = sent_tokenize(text=dataset, language='english')
	return sentences

update_stop_words = {',', ';', 'The', '.', '?', 'I', 'And', '!', "'d", "'s", 
                    	 ':', '\'', '’', '”', '[', ']', '\'\'', '``', '*', '<', '>', ')', '(', 
                    	 '*', "_", "-", "'S", "To", "to", "thou", "what", "But", "but", "Thy",
                    	 "thy", "that","That", "What", 'Shall', 'shall', "May", 'may', "of", "Of", "when",
                    	 "When", "for", "For","Thee", "thee", "well", "Well", "As", "as", "You", "you", "never",
                    	   "Never", "Also", "also", "must", "Must", "--", "by", "By", "'ll", "if", "If",
                    	    "or", "Or", "it", "It"}


book_path = os.path.join(base_booklib_dir, 'shakespeare' ,'allShakespeare.txt')

dataset = get_book_from_file(book_path, utf=True)
sentences = sentences_tokenize(dataset)


words, filterd_words = words_tokenize(True, update_stop_words)
word_count(filterd_words, most_common_words= 100)


book_path = os.path.join(base_booklib_dir, 'chauser' , 'Chauser.txt')

dataset = get_book_from_file(book_path, utf=True)
sentences = sentences_tokenize(dataset)


words, filterd_words = words_tokenize(True, update_stop_words)
word_count(filterd_words, most_common_words= 100)
