#generate dictionary;
#generate question and answer string encodings;
#generate multiple choice label;
import json;
def load_json(fname):
	f=open(fname,'r');
	data=json.load(f);
	f.close();
	return data;

def save_json(fname,data):
	f=open(fname,'w');
	json.dump(data,f);
	f.close();

import re
import nltk
def tokenize(sentence):
	return nltk.word_tokenize(sentence);
	#return [i for i in re.split(r"([-.\"',:? !\$#@~()*&\^%;\[\]/\\\+<>\n=])", sentence) if i!='' and i!=' ' and i!='\n'];


questions_train=load_json('VQA/Questions/OpenEnded_mscoco_train2014_questions.json');
questions_val=load_json('VQA/Questions/OpenEnded_mscoco_val2014_questions.json');
annotations_train=load_json('VQA/Annotations/mscoco_train2014_annotations.json');
annotations_val=load_json('VQA/Annotations/mscoco_val2014_annotations.json');
#fake_responses=load_json('VQA/Results/OpenEnded_mscoco_train2014_fake_results.json');

#compute common answers
answer_count=dict();
for i in annotations_train['annotations']:
	for j in i['answers']:
		if j['answer'] in answer_count:
			answer_count[j['answer']]=answer_count[j['answer']]+1;
		else:
			answer_count[j['answer']]=1;

answers=answer_count.items();
answers=sorted(answers, key=lambda a: a[1],reverse=True) 
#summarize images
image_id=dict();
subtype=questions_train['data_subtype'];
imdir='%s/COCO_%s_%012d.jpg'
for i in questions_train['questions']:
	image_id[i['question_id']]=imdir%(subtype,subtype,i['image_id']);

subtype=questions_val['data_subtype'];
for i in questions_val['questions']:
	image_id[i['question_id']]=imdir%(subtype,subtype,i['image_id']);

#tokenize questions
word_stats=dict();
for i in questions_train['questions']:
	for j in tokenize(i['question']):
		if j in word_stats:
			word_stats[j]=word_stats[j]+1;
		else:
			word_stats[j]=1;

for i in questions_val['questions']:
	for j in tokenize(i['question']):
		if j in word_stats:
			word_stats[j]=word_stats[j]+1;
		else:
			word_stats[j]=1;

word_stats=sorted(word_stats.items(), key=lambda a: a[1],reverse=True) 
word_lookup=dict(zip([i[0] for i in word_stats],range(1,len(word_stats)+1)));
encoding=dict(zip(range(1,len(word_stats)+1),[i[0] for i in word_stats]));
words=[encoding[i] for i in encoding];

questions_train_tokens=list();
questions_train_images=list();
for i in questions_train['questions']:
	tokens=[j for j in tokenize(i['question']) if j in word_lookup];
	questions_train_tokens.append((i['question_id'],[word_lookup[j] for j in tokens]));
	questions_train_images.append((i['question_id'],image_id[i['question_id']]));

questions_train_tokens=dict(questions_train_tokens);
questions_train_images=dict(questions_train_images);

questions_val_tokens=list();
questions_val_images=list();
for i in questions_val['questions']:
	tokens=[j for j in tokenize(i['question']) if j in word_lookup];
	questions_val_tokens.append((i['question_id'],[word_lookup[j] for j in tokens]));
	questions_val_images.append((i['question_id'],image_id[i['question_id']]));

questions_val_tokens=dict(questions_val_tokens);
questions_val_images=dict(questions_val_images);

#produce ground truth labels with top N answers
thresh=1000;
answers_thresh=[i[0] for i in answers[0:thresh]];
answer_lookup=dict(zip(answers_thresh,range(1,thresh+1)));
image_val=list();
label_val=list();
qt_val=list();
asdf=0;
for i in annotations_val['annotations']:
	d=dict();
	for j in i['answers']:
		if j['answer'] in answers_thresh:
			if j['answer'] in d:
				d[j['answer']]=d[j['answer']]+1;
			else:
				d[j['answer']]=1;
	
	answer='';
	if len(d)>0:
		d=sorted(d.items(), key=lambda a: a[1],reverse=True);
		
		if d[0][1]>=3:
			asdf=asdf+1;
		else:
			asdf=asdf+d[0][1]/3.0;
		
		answer=d[0][0];
		label_val.append(answer_lookup[answer]);
		qt_val.append(questions_val_tokens[i['question_id']])
		image_val.append(image_id[i['question_id']]);
	
	
	if i['question_id']%1000==0:
		print(i['question_id']);

image_train=list();
label_train=list();
qt_train=list();
asdf2=0;
for i in annotations_train['annotations']:
	d=dict();
	for j in i['answers']:
		if j['answer'] in answers_thresh:
			if j['answer'] in d:
				d[j['answer']]=d[j['answer']]+1;
			else:
				d[j['answer']]=1;
	
	answer='';
	if len(d)>0:
		d=sorted(d.items(), key=lambda a: a[1],reverse=True);
		answer=d[0][0];
		if d[0][1]>=3:
			asdf2=asdf2+1;
		else:
			asdf2=asdf2+d[0][1]/3.0;
		
		label_train.append(answer_lookup[answer]);
		qt_train.append(questions_train_tokens[i['question_id']]);
		image_train.append(image_id[i['question_id']]);
	
		
	if i['question_id']%1000==0:
		print(i['question_id']);


responses=list();
for i in annotations_val['annotations']:
	responses.append({"question_id":i['question_id'],"answer":"yes"});

save_json('data.json',{"encoding":words,"label_train":label_train,"qt_train":qt_train,"label_val":label_val,"qt_val":qt_val,"questions_val_tokens":questions_val_tokens,"questions_train_tokens":questions_train_tokens,"answer_lookup":answer_lookup,"answers_thresh":answers_thresh,"image_train":image_train,"image_val":image_val,'questions_train_images':questions_train_images,"questions_val_images":questions_val_images,"responses":responses});

save_json('question_dictionary.json',encoding);
save_json('answer_dictionary.json',answers_thresh);

print(asdf)
print(asdf2)