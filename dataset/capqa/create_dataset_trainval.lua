
function readAll(file)
    local f = io.open(file, "r")
    local content = f:read("*all")
    f:close()
    return content
end
cjson=require('cjson');
function loadJson(fname)
	local t=readAll(fname)
	return cjson.decode(t)
end

dataset=loadJson('../vqa/data.json');

question_dictionary=dataset['encoding'];
answer_labels_train=torch.LongTensor(dataset['label_train']);

question_max_length=0;
ntest=0;
ntrain=#dataset['qt_train'];
for qid,i in pairs(dataset['questions_train_tokens']) do
	question_max_length=math.max(question_max_length,#i);
end
for qid,i in pairs(dataset['questions_val_tokens']) do
	question_max_length=math.max(question_max_length,#i);
	ntest=ntest+1;
end
print(string.format('Question max length %d.',question_max_length));

question_tokens_train=torch.LongTensor(ntrain,question_max_length);
question_lengths_train=torch.LongTensor(ntrain);
question_ims_train={};
for i=1,ntrain do
	local tmp=torch.LongTensor(dataset['qt_train'][i]);
	question_lengths_train[i]=tmp:size(1);
	question_tokens_train[i][{{1,tmp:size(1)}}]=tmp;
	table.insert(question_ims_train,dataset['image_train'][i]);
end

question_tokens_test=torch.LongTensor(ntest,question_max_length);
question_lengths_test=torch.LongTensor(ntest);
question_ids_test=torch.LongTensor(ntest);
question_ims_test={};
cnt=0;
for qid,i in pairs(dataset['questions_val_tokens']) do
	cnt=cnt+1;
	local tmp=torch.LongTensor(i);
	question_lengths_test[cnt]=tmp:size(1);
	question_tokens_test[cnt][{{1,tmp:size(1)}}]=tmp;
	question_ids_test[cnt]=qid;
	table.insert(question_ims_test,dataset['questions_val_images'][qid]);
end


train_ims_nqs={};
train_ims_ind={};
cnt=0;
for i=1,#question_ims_train do
	if train_ims_nqs[question_ims_train[i]]==nil then
		train_ims_nqs[question_ims_train[i]]=1;
		cnt=cnt+1;
		train_ims_ind[question_ims_train[i]]=cnt;
	else
		train_ims_nqs[question_ims_train[i]]=train_ims_nqs[question_ims_train[i]]+1;
	end
end

test_ims_nqs={};
test_ims_ind={};
cnt=0;
for i=1,#question_ims_test do
	if test_ims_nqs[question_ims_test[i]]==nil then
		test_ims_nqs[question_ims_test[i]]=1;
		cnt=cnt+1;
		test_ims_ind[question_ims_test[i]]=cnt;
	else
		test_ims_nqs[question_ims_test[i]]=test_ims_nqs[question_ims_test[i]]+1;
	end
end

cnt=0;
train_ims={};
for imname,id in pairs(train_ims_ind) do
	train_ims[id]=imname;
end
test_ims={};
for imname,id in pairs(test_ims_ind) do
	test_ims[id]=imname;
end

question_imids_train=torch.LongTensor(ntrain);
for i=1,ntrain do
	question_imids_train[i]=train_ims_ind[question_ims_train[i]];
end
question_imids_test=torch.LongTensor(ntest);
for i=1,ntest do
	question_imids_test[i]=test_ims_ind[question_ims_test[i]];
end

answer_dictionary={};
for answer,id in pairs(dataset['answer_lookup']) do
	answer_dictionary[id]=answer;
end



function reverse_table(tbl)
	local t={};
	for i,j in pairs(tbl) do
		t[j]=i;
	end
	return t;
end

caps=torch.load('../mscoco/mscoco_captions.t7');
caps['ims_id']=reverse_table(caps['ims']);
N=5;
ind5_train=torch.LongTensor(#train_ims*N);
for i=1,#train_ims do
	for j=1,N do
		ind5_train[N*(i-1)+j]=N*(caps['ims_id'][train_ims[i]]-1)+j;
	end
end
ind5_test=torch.LongTensor(#test_ims*N);
for i=1,#test_ims do
	for j=1,N do
		ind5_test[N*(i-1)+j]=N*(caps['ims_id'][test_ims[i]]-1)+j;
	end
end


data_train={};
data_train['ims']=train_ims;
data_train['ncaptions_per_im']=N;

data_train['caption_dictionary']=caps['dictionary'];
data_train['caption_tokens']=caps['tokens']:index(1,ind5_train);

data_train['question_dictionary']=question_dictionary;
data_train['question_tokens']=question_tokens_train;
data_train['question_imids']=question_imids_train;

data_train['answer_labels']=answer_labels_train;
data_train['answer_dictionary']=answer_dictionary;

data_test={};
data_test['ims']=test_ims;
data_test['ncaptions_per_im']=N;

data_test['caption_dictionary']=caps['dictionary'];
data_test['caption_tokens']=caps['tokens']:index(1,ind5_test);

data_test['question_dictionary']=question_dictionary;
data_test['question_tokens']=question_tokens_test;
data_test['question_imids']=question_imids_test;
data_test['question_ids']=question_ids_test;

data_test['answer_dictionary']=answer_dictionary;


torch.save('dataset_train.t7',data_train);
torch.save('dataset_test.t7',data_test);
