--Join images with captions to make train val test datasets
cmd = torch.CmdLine();
cmd:text('Combine images with captions to create image-captioning datasets: 82783/5000/5000');
cmd:text('Options')
cmd:option('-image','../mscoco/mscoco_vgg19_center.t7','t7 of image features');
cmd:option('-caption','../mscoco/captions.json','JSON of tokenized captions');
params=cmd:parse(arg);
--print(params)

cjson=require('cjson');
function readAll(file)
    local f = io.open(file, "r")
    local content = f:read("*all")
    f:close()
    return content
end
function loadJson(fname)
	local t=readAll(fname)
	return cjson.decode(t)
end


--figure out splits
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
function index(data,dictionary)
	local ind=torch.LongTensor(#data);
	for i=1,#data do
		ind[i]=dictionary[data[i]];
	end
	return ind;
end
function slice_cell(data,ind)
	local new_data={};
	for i=1,ind:size(1) do
		table.insert(new_data,data[ind[i]]);
	end
	return new_data;
end

--Load json files
images=torch.load(params.image);
images.lookup=reverse_table(images.imname);
captions=loadJson(params.caption);
captions.lookup=reverse_table(captions.imname);


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

--Save t7s
train_ind=index(train_ims,images.lookup);
train_ind_caption=index(train_ims,captions.lookup);
torch.save('dataset_train.t7',{image={fvs=images.fvs:index(1,train_ind):clone(),imname=train_ims,lookup=reverse_table(train_ims)},caption={caption=slice_cell(captions.caption,train_ind_caption),imname=train_ims,lookup=reverse_table(train_ims)}});


val_ind=index(val_ims,images.lookup);
val_ind_caption=index(val_ims,captions.lookup);
torch.save('dataset_val.t7',{image={fvs=images.fvs:index(1,val_ind):clone(),imname=val_ims,lookup=reverse_table(val_ims)},caption={caption=slice_cell(captions.caption,val_ind_caption),imname=val_ims,lookup=reverse_table(val_ims)}});


test_ind=index(test_ims,images.lookup);
test_ind_caption=index(test_ims,captions.lookup);
torch.save('dataset_test.t7',{image={fvs=images.fvs:index(1,test_ind):clone(),imname=test_ims,lookup=reverse_table(test_ims)},caption={caption=slice_cell(captions.caption,test_ind_caption),imname=test_ims,lookup=reverse_table(test_ims)}});
