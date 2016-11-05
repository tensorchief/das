#! /Users/silviar/anaconda/bin/python

from sklearn.cross_validation import train_test_split

with open('datlist.txt','r') as infile:
	runs = sorted(infile.readlines())

train, test = train_test_split(runs, test_size = 0.2)

deleted = list()
for index,item in enumerate(sorted(runs)):
	if item not in train and item not in deleted:
		if runs[index-1] in train:
			previous = train.index(runs[index-1])
			deleted.append(train[previous])
			del train[previous]
		if index < len(runs) -1:
			if runs[index+1] in train:
				following = train.index(runs[index+1])
				deleted.append(train[following])
				del train[following]

with open('training_set.txt', 'w') as outfile:
	for item in sorted(train):
		outfile.write(item)

with open('test_set.txt','w') as outfile:
    for item in sorted(test):
        outfile.write(item)
