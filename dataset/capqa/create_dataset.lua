--Join QAs with captions for train/val
cmd = torch.CmdLine();
cmd:text('Combine QA with captions to create a dataset');
cmd:text('using the 19-layer VGG net in caffe model zoo');
cmd:text('Options')
cmd:option('-question','../vqa/train.json','JSON of tokenized questions with answers');
cmd:option('-caption','../mscoco/captions.json','JSON of tokenized captions');
cmd:option('-output','dataset_train.t7','Output VQA-Caption dataset');
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

--Load json files
questions=loadJson(params.question);
captions=loadJson(params.caption);

--Add a lookup table for captions
captions.lookup={};
for i=1,#captions.imname do
	captions.lookup[captions.imname[i]]=i;
end

--Save t7
torch.save(params.output,{question=questions,caption=captions});