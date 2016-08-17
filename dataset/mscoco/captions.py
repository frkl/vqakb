#generate dictionary;
#generate question and answer string encodings;
#generate multiple choice label;
import json;
def loadJson(fname):
	f=open(fname,'r');
	data=json.load(f);
	f.close();
	return data;

def saveJson(fname,data):
	f=open(fname,'w');
	json.dump(data,f);
	f.close();


train_captions=loadJson('./annotations/captions_train2014.json');
train_captions['dataset']='train2014';
val_captions=loadJson('./annotations/captions_val2014.json');
val_captions['dataset']='val2014';

import re
import nltk
def tokenize_nltk(sentence):
	return nltk.word_tokenize(sentence);
	#return [i for i in re.split(r"([-.\"',:? !\$#@~()*&\^%;\[\]/\\\+<>\n=])", sentence) if i!='' and i!=' ' and i!='\n'];



def dataset(annotations,tokenizer):
	d=dict();
	for cap in annotations['annotations']:
		imname='%s/COCO_%s_%012d.jpg'%(annotations['dataset'],annotations['dataset'],cap['image_id']);
		if not (imname in d):
			d[imname]=list();
		
		d[imname].append((tokenizer(cap['caption']),cap['id']));
	
	imnames=list();
	caption=list();
	caption_id=list();
	for k in d:
		imnames.append(k);
		caption.append([c[0] for c in d[k]]);
		caption_id.append([c[1] for c in d[k]]);
	
	return {'imname':imnames,'caption':caption,'caption_id':caption_id};

def merge(dataset1,dataset2):
	d=dict();
	for k in dataset1:
		d[k]=dataset1[k]+dataset2[k];
	
	return d;

dataset_train=dataset(train_captions,tokenize_nltk);
dataset_val=dataset(val_captions,tokenize_nltk);
saveJson('captions.json',merge(dataset_train,dataset_val));
