
function readlines(fname)
	local data={};
	local f=torch.DiskFile(fname,'r');
	f:quiet();
	local i=0;
	while true do
		local line=f:readString('*l');
		if line=='' then break end
		i=i+1;
		data[i]=line;
	end
	f:close();
	return data;
end
function reverse_table(tbl)
	local t={};
	for i,j in pairs(tbl) do
		t[j]=i;
	end
	return t;
end


N=5;
train_ims=readlines('./splits/coco_train.txt');
for i=1,#train_ims do
	train_ims[i]='train2014/'..train_ims[i];
end

val_ims=readlines('./splits/coco_val.txt');
for i=1,#val_ims do
	val_ims[i]='val2014/'..val_ims[i];
end

test_ims=readlines('./splits/coco_test.txt');
for i=1,#test_ims do
	test_ims[i]='val2014/'..test_ims[i];
end

ntrain=#train_ims;
nval=#val_ims;
ntest=#test_ims;
print(string.format('Train:%d',ntrain));
print(string.format('Val:%d',nval));
print(string.format('Test:%d',ntest));

caps=torch.load('../mscoco/mscoco_captions.t7');
ims=torch.load('../mscoco/mscoco_vgg_center_v2.t7')


caps['ims_id']=reverse_table(caps['ims']);
ind5_train=torch.LongTensor(ntrain*N);
for i=1,ntrain do
	for j=1,N do
		ind5_train[N*(i-1)+j]=N*(caps['ims_id'][train_ims[i]]-1)+j;
	end
end

ind5_val=torch.LongTensor(nval*N);
for i=1,nval do
	for j=1,N do
		ind5_val[N*(i-1)+j]=N*(caps['ims_id'][val_ims[i]]-1)+j;
	end
end

ind5_test=torch.LongTensor(ntest*N);
for i=1,ntest do
	for j=1,N do
		ind5_test[N*(i-1)+j]=N*(caps['ims_id'][test_ims[i]]-1)+j;
	end
end

ims['ims_id']=reverse_table(ims['ims']);
ind_train=torch.LongTensor(ntrain);
for i=1,ntrain do
	ind_train[i]=ims['ims_id'][train_ims[i]];
end

ind_val=torch.LongTensor(nval);
for i=1,nval do
	ind_val[i]=ims['ims_id'][val_ims[i]];
end

ind_test=torch.LongTensor(ntest);
for i=1,ntest do
	ind_test[i]=ims['ims_id'][test_ims[i]];
end

data_train={};
data_train['ims']=train_ims;
data_train['caption_tokens']=caps['tokens']:index(1,ind5_train);
data_train['image_fvs']=ims['fvs']:index(1,ind_train);
data_train['caption_dictionary']=caps['dictionary'];
data_train['ncaptions_per_im']=N;


data_val={};
data_val['ims']=val_ims;
data_val['caption_tokens']=caps['tokens']:index(1,ind5_val);
data_val['image_fvs']=ims['fvs']:index(1,ind_val);
data_val['caption_dictionary']=caps['dictionary'];
data_val['ncaptions_per_im']=N;


data_test={};
data_test['ims']=test_ims;
data_test['caption_tokens']=caps['tokens']:index(1,ind5_test);
data_test['image_fvs']=ims['fvs']:index(1,ind_test);
data_test['caption_dictionary']=caps['dictionary'];
data_test['ncaptions_per_im']=N;



torch.save('dataset_train.t7',data_train);
torch.save('dataset_val.t7',data_val);
torch.save('dataset_test.t7',data_test);