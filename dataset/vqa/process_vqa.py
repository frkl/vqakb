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


questions_train=loadJson('VQA/Questions/OpenEnded_mscoco_train2014_questions.json');
questions_val=loadJson('VQA/Questions/OpenEnded_mscoco_val2014_questions.json');
annotations_train=loadJson('VQA/Annotations/mscoco_train2014_annotations.json');
annotations_val=loadJson('VQA/Annotations/mscoco_val2014_annotations.json');

questions_test=loadJson('VQA/Questions/OpenEnded_mscoco_test2015_questions.json');
questions_test_dev=loadJson('VQA/Questions/OpenEnded_mscoco_test-dev2015_questions.json');
questions_test_dev['data_subtype']='test2015';




import re
import nltk
def tokenize_nltk(sentence):
	return nltk.word_tokenize(sentence);
	#return [i for i in re.split(r"([-.\"',:? !\$#@~()*&\^%;\[\]/\\\+<>\n=])", sentence) if i!='' and i!=' ' and i!='\n'];



def dataset(questions,annotations,tokenizer):
	qid=[q['question_id'] for q in questions['questions']];
	imname=['%s/COCO_%s_%012d.jpg'%(questions['data_subtype'],questions['data_subtype'],q['image_id']) for q in questions['questions']];
	question=[tokenizer(q['question']) for q in questions['questions']];
	answer=[[a['answer'] for a in ans['answers']] for ans in annotations['annotations']];
	qtype=[ans['question_type'] for ans in annotations['annotations']];
	atype=[ans['answer_type'] for ans in annotations['annotations']];
	return {'question_id':qid,'imname':imname,'question':question,'answer':answer,'question_type':qtype,'answer_type':atype};

def dataset_eval(questions,tokenizer):
	qid=[q['question_id'] for q in questions['questions']];
	imname=['%s/COCO_%s_%012d.jpg'%(questions['data_subtype'],questions['data_subtype'],q['image_id']) for q in questions['questions']];
	question=[tokenizer(q['question']) for q in questions['questions']];
	return {'question_id':qid,'imname':imname,'question':question};

def merge(dataset1,dataset2):
	d=dict();
	for k in dataset1:
		d[k]=dataset1[k]+dataset2[k];
	
	return d;

dataset_train=dataset(questions_train,annotations_train,tokenize_nltk);
dataset_val=dataset(questions_val,annotations_val,tokenize_nltk);
saveJson('train.json',dataset_train);
saveJson('val.json',dataset_val);

dataset_trainval=merge(dataset_train,dataset_val);
dataset_test_eval=dataset_eval(questions_test,tokenize_nltk);
dataset_test_dev_eval=dataset_eval(questions_test_dev,tokenize_nltk);
saveJson('trainval.json',dataset_trainval);
saveJson('test_eval.json',dataset_test_eval);
saveJson('test_dev_eval.json',dataset_test_dev_eval);
