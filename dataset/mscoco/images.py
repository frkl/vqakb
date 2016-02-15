import os
import time
import numpy
import json

def loadjson(fname):
	f=open(fname,'r');
	data=json.load(f);
	f.close();
	return data;


def reads(fname):
	f=open(fname,'r');
	d=list();
	for line in f:
		d.append(line.rstrip('\n'));
	
	f.close();
	return d;

instance_train=loadjson('./annotations/instances_train2014.json')
instance_val=loadjson('./annotations/instances_val2014.json')
ims_train=list();
ims_val=list();
for i in instance_train['images']:
	ims_train.append(i['file_name']);

for i in instance_val['images']:
	ims_val.append(i['file_name']);


ims=list();
for i in ims_train:
	ims.append('train2014/'+i);

for i in ims_val:
	ims.append('val2014/'+i);


print('Saving image names to imnames.json');
f=open('imnames.json','w');
json.dump(ims,f);
f.close();
