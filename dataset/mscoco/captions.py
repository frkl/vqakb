import json

def loadjson(filename):
	f=open(filename,'r');
	data=json.load(f);
	f.close();
	return data;


def savejson(data,filename):
	f=open(filename,'w');
	json.dump(data,f);
	f.close();


train_captions=loadjson('./annotations/captions_train2014.json');
train_images=loadjson('./annotations/instances_train2014.json');
val_captions=loadjson('./annotations/captions_val2014.json');
val_images=loadjson('./annotations/instances_val2014.json');

ims_id=dict();
for i in train_images['images']:
	ims_id[i['id']]='train2014/'+i['file_name'];

for i in val_images['images']:
	ims_id[i['id']]='val2014/'+i['file_name'];

#captions are first come, first served
captions_id=dict();
for i in train_captions['annotations']:
	if not (i['image_id'] in captions_id):
		captions_id[i['image_id']]=list();
	
	captions_id[i['image_id']].append(i['caption']);

for i in val_captions['annotations']:
	if not (i['image_id'] in captions_id):
		captions_id[i['image_id']]=list();
	
	captions_id[i['image_id']].append(i['caption']);

ims=list();
for i in ims_id:
	ims.append(ims_id[i]);

captions=list();
for i in ims_id:
	captions.append(captions_id[i]);

#tokenize

import nltk

words=dict();
for i in captions:
	for j in i:
		for k in nltk.word_tokenize(j):
			if not (k in words):
				words[k]=1;
			else:
				words[k]=words[k]+1;

words_sorted=sorted(words.items(),key=lambda x:x[1],reverse=True);

words=[i[0] for i in words_sorted];
lookup=dict(zip(words,range(0,len(words))));
caption_tokens=list();
for i in captions:
	tmp1=list();
	for j in i:
		tmp2=[lookup[k] for k in nltk.word_tokenize(j)];
		tmp1.append(tmp2);
	
	caption_tokens.append(tmp1);

savejson({'captions':captions,'caption_tokens':caption_tokens,'words':words,'lookup':lookup,"ims":ims},'captions.json');
