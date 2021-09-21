import os
import collections
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

def make_features() :
	direc = "enron1/ham/"
	hamfiles = os.listdir(direc)

	emails = [direc + email for email in hamfiles]

	direc = "enron1/spam/"
	spamfiles = os.listdir(direc)

	spamemails = [direc + email for email in spamfiles]

	emails += spamemails

	words = []
	for email in emails :
		temp = []
		flp = open(email , encoding="utf8" , errors='ignore')
		r = flp.read()
		temp += r.split(' ')
		for el in temp :
			if el.isalpha() :
				words.append(el)

	print(len(words))
	main_words = collections.Counter(words)

	return main_words.most_common(3000)

def make_feature_vector(main_words) :
	direc = "enron1/ham/"
	hamfiles = os.listdir(direc)

	emails = [direc + email for email in hamfiles]

	direc = "enron1/spam/"
	spamfiles = os.listdir(direc)

	spamemails = [direc + email for email in spamfiles]

	emails += spamemails

	feature_vec = []
	labels = []
	count = len(emails)
	for email in emails :
		data = []
		flp = open(email , encoding="utf8" , errors='ignore')
		temp = flp.read().split(' ')
		for word in main_words :
			data.append(temp.count(word[0]))
		feature_vec.append(data)
		if "ham" in email :
			labels.append(0)
		if "spam" in email :
			labels.append(1)
		print(count)
		count -= 1

	return feature_vec , labels


d = make_features()
# print(d)
features , labels = make_feature_vector(d)

x_train , x_test , y_train , y_test = train_test_split(features , labels , test_size  = 0.3)

clf = svm.SVC(kernel='rbf')
clf.fit(x_train , y_train)

preds = clf.predict(x_test)

print(accuracy_score(y_test , preds))
