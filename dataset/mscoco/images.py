#Generate a list of image names from the MSCOCO dataset for extracting features
#Save to imnames,json

import json

def loadJson(fname):
	f=open(fname,'r');
	data=json.load(f);
	f.close();
	return data;

instance_train=loadJson('./annotations/instances_train2014.json')
instance_val=loadJson('./annotations/instances_val2014.json')
imageinfo_test2014=loadJson('./annotations/image_info_test2014.json')
imageinfo_test_dev2015=loadJson('./annotations/image_info_test-dev2015.json')
imageinfo_test2015=loadJson('./annotations/image_info_test2015.json')


ims=list();
for i in instance_train['images']:
	ims.append('train2014/'+i['file_name']);

for i in instance_val['images']:
	ims.append('val2014/'+i['file_name']);

for i in imageinfo_test2014['images']:
	ims.append('test2014/'+i['file_name']);

for i in imageinfo_test2015['images']:
	ims.append('test2015/'+i['file_name']);

for i in imageinfo_test_dev2015['images']:
	ims.append('test2015/'+i['file_name']);


print('Saving image names to imnames.json');
f=open('imnames.json','w');
json.dump(ims,f);
f.close();
