cmd = torch.CmdLine();
cmd:text('Extract image and caption fv7 from a image-caption ranking model');
cmd:text('Dataset')
cmd:option('-caption','../dataset/mscoco/captions.json','captions for evaluation');
cmd:option('-image','../dataset/mscoco/mscoco_vgg19_center.t7','images for evaluation');
cmd:option('-session','session_rf6Ly4oTGKA0','Session folder name');
cmd:option('-output','imcap_fvs.t7','Session folder name');
cmd:option('-batch',200,'Batch size (Adjust base on GRAM and dataset size)');
params=cmd:parse(arg);

require 'nn'
require 'cutorch'
require 'cunn' 
require 'nngraph'
RNN=require('../utils/word_RNN');
require '../utils/utils'



print('Initializing session');
local cjson=require('cjson');
local function readAll(file)
    local f = io.open(file, "r")
	if f==nil then
		error(string.format('Failed to open file %s',file));
	end
    local content = f:read("*all");
    f:close()
    return content;
end
local function loadJson(fname)
	local t=readAll(fname);
	return cjson.decode(t);
end
function writeAll(file,data)
    local f = io.open(file, "w")
    f:write(data)
    f:close() 
end
function saveJson(fname,t)
	return writeAll(fname,cjson.encode(t))
end
basedir=paths.concat('./sessions',params.session);
paths.mkdir(paths.concat(basedir,'pred'));
params_train=loadJson(paths.concat(basedir,'_session_config.json'));

log_file=paths.concat(basedir,'log_test.txt');
function Log(msg)
	local f = io.open(log_file, "a")
	print(msg);
	f:write(msg..'\n');
	f:close()
end


print('Loading dataset');
dataset_im=torch.load(params.image);
dataset_cap=loadJson(params.caption);
dataset_cap.tokens,_=encode_sents3(dataset_cap.caption,params_train.caption_dictionary);
collectgarbage();


--Network definitions
Log('Loading models');
function wrap_net(net,gpu)
	local d={};
	if gpu then
		d.net=net:cuda();
	else
		d.net=net;
	end
	d.w,d.dw=d.net:getParameters();
	d.deploy=d.net:clone('weight','bias','gradWeight','gradBias','running_mean','running_std','running_var');
	return d;
end
tmp=torch.load(paths.concat(basedir,'model',string.format('model_epoch%d.t7',params_train.epochs)));

C_embedding_net=wrap_net(tmp.C_embedding_net,true);
C_encoder_net=RNN:new(tmp.C_encoder_net,dataset_cap.tokens:size(3),true);
multimodal_net=wrap_net(tmp.multimodal_net,true);
C_encoder_net:evaluate();
C_embedding_net.deploy:evaluate();
multimodal_net.deploy:evaluate();
--Create dummy states and gradients
dummy_state=torch.DoubleTensor(params_train.nhsent):fill(0):cuda();
dummy_output=torch.DoubleTensor(params_train.nhoutput):fill(0):cuda();

--Compute scores
Log('Computing caption fc7')
nim=dataset_cap.tokens:size(1);
ncaps_per_im=dataset_cap.tokens:size(2);
capfvs=torch.DoubleTensor(nim,ncaps_per_im,params_train.nhsent):fill(0);
dummy_image=dataset_im.fvs[1];
for i=1,nim,params.batch do
	local r=math.min(nim,i+params.batch-1);
	local fv_C=sort_by_length_left_aligned(dataset_cap.tokens[{{i,r}}]:view(ncaps_per_im*(r-i+1),-1),true);
	local embedding_C=C_embedding_net.deploy:forward(fv_C.words);
	local state_C,_=C_encoder_net:forward(torch.repeatTensor(dummy_state:fill(0),fv_C.map_to_sequence:size(1),1),embedding_C,fv_C.batch_sizes);
	state_C=state_C:index(1,fv_C.map_to_sequence);
	capfvs[{{i,r}}]=state_C:view(r-i+1,ncaps_per_im,-1):double();
	print(string.format('%d/%d',r,nim))
end

Log('Computing image fc7')
nim=dataset_im.fvs:size(1);
imfvs=torch.DoubleTensor(nim,params_train.nhsent):fill(0);
timer = torch.Timer();
dummy_caption=capfvs[{1,{1}}];
for i=1,nim,params.batch do
	local r=math.min(nim,i+params.batch-1);
	multimodal_net.deploy:forward({torch.repeatTensor(dummy_caption,r-i+1,1):cuda(),dataset_im.fvs[{{i,r}}]:cuda()});
	imfvs[{{i,r}}]=multimodal_net.deploy.modules[6].output:double();
	print(string.format('%d/%d',r,nim))
end

torch.save(paths.concat(basedir,'pred',params.output),{imfvs=imfvs,capfvs=capfvs});