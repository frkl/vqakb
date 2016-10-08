cmd = torch.CmdLine();
cmd:text('Evaluate the image-caption ranking model');
cmd:text('Dataset')
cmd:option('-data_test','../dataset/imcap/dataset_test.t7','Dataset for testing');
cmd:option('-K',1000,'Using first K images for testing');

cmd:option('-session','','Select which model to test')
cmd:option('-batch',1000,'Batch size (Adjust base on GRAM and dataset size)');
params=cmd:parse(arg);

require 'nn'
require 'cutorch'
require 'cunn' 
require 'nngraph'
RNN=require('../utils/word_RNN');
require '../utils/utils'
require '../utils/metric'



print('Initializing session');
local cjson=require('cjson');
local function readAll(file)
    local f = io.open(file, "r")
	if f==nil then
		error({msg='Failed to open file',file=file});
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
function sequence_length(seq)
	local v=seq:gt(0):long():sum(2):view(-1):long();
	return v;
end
dataset_test=torch.load(params.data_test);
dataset_test.caption.tokens,_=encode_sents3(dataset_test.caption.caption,params_train.caption_dictionary);
collectgarbage();


--Network definitions
print('Loading models');
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
C_encoder_net=RNN:new(tmp.C_encoder_net,dataset_test.caption.tokens:size(3),true);
multimodal_net=wrap_net(tmp.multimodal_net,true);
C_encoder_net:evaluate();
C_embedding_net.deploy:evaluate();
multimodal_net.deploy:evaluate();
--Create dummy states and gradients
dummy_state=torch.DoubleTensor(params_train.nhsent):fill(0):cuda();
dummy_output=torch.DoubleTensor(params_train.nhoutput):fill(0):cuda();


--Using the first 5 captions for evaluation.
function dataset_test:batch_eval(sc,ec,si,ei)
	local timer = torch.Timer();
	local tokens_cap=torch.LongTensor(params_train.ncaps_per_im*(ec-sc+1),self.caption.tokens:size(3)):fill(0); --select one caption per image
	for i=1,ec-sc+1 do
		local imname=self.image.imname[i+sc-1];
		local ind=self.caption.lookup[imname];
		tokens_cap[{{params_train.ncaps_per_im*(i-1)+1,params_train.ncaps_per_im*i}}]=self.caption.tokens[ind][{{1,params_train.ncaps_per_im}}];
	end
	local fv_im=self.image.fvs[{{si,ei}}];
	local fv_sorted_c=sort_by_length_left_aligned(tokens_cap,true);
	return fv_sorted_c,fv_im:cuda();
end
function Forward_test(sc,ec,si,ei)
	local fv_C,fv_im=dataset_test:batch_eval(sc,ec,si,ei);
	--Forward
	local embedding_C=C_embedding_net.deploy:forward(fv_C.words);
	local state_C,_=C_encoder_net:forward(torch.repeatTensor(dummy_state:fill(0),fv_C.map_to_sequence:size(1),1),embedding_C,fv_C.batch_sizes);
	local tv_C=state_C:index(1,fv_C.map_to_sequence);
	local scores=multimodal_net.deploy:forward({tv_C,fv_im});
	return scores:double();
end

print('Computing scores')
npts_test=math.min(#dataset_test.image.imname,params.K);
scores_test=torch.zeros(npts_test*params_train.ncaps_per_im,npts_test):long();
for i=1,npts_test,params.batch do
	for j=1,npts_test,params.batch do
		--print(string.format('\ttesting %d/%d %f',i,npts_test,timer:time().real));
		local ri=math.min(i+params.batch-1,npts_test);
		local rj=math.min(j+params.batch-1,npts_test);
		scores_test[{{5*(i-1)+1,5*ri},{j,rj}}]=Forward_test(i,ri,j,rj);
	end
end
--figure out the ground truth
gt_im=torch.zeros(npts_test,params_train.ncaps_per_im);
gt_text=torch.zeros(npts_test,params_train.ncaps_per_im);
for i=1,npts_test do
	gt_im[i]:fill(i);
	gt_text[i]=torch.range(1,params_train.ncaps_per_im)+(i-1)*params_train.ncaps_per_im;
end
gt_im=gt_im:view(-1):long();
gt_text=gt_text:long();
im_r_1=metric.accuracy_N(scores_test,gt_im,1);
im_r_5=metric.accuracy_N(scores_test,gt_im,5);
im_r_10=metric.accuracy_N(scores_test,gt_im,10);
cap_r_1=metric.accuracy_NM(scores_test:t(),gt_text,1);
cap_r_5=metric.accuracy_NM(scores_test:t(),gt_text,5);
cap_r_10=metric.accuracy_NM(scores_test:t(),gt_text,10);
Log(string.format('test acc\t im: %f %f %f\t cap: %f %f %f',im_r_1,im_r_5,im_r_10,cap_r_1,cap_r_5,cap_r_10));