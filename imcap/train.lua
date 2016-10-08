cmd = torch.CmdLine();
cmd:text('Train an image-caption ranking model');
cmd:text('Dataset')
cmd:option('-data','../dataset/imcap/dataset_train.t7','Dataset for training');
cmd:option('-data_test','../dataset/imcap/dataset_val.t7','Dataset for validation, it\'s nice to directly evaluate accuracy on the fly');
cmd:option('-K',1000,'Using first K images for testing');

cmd:text('Model parameters');
cmd:option('-nhword',256,'Word embedding size');
cmd:option('-nh',512,'RNN size');
cmd:option('-nlayers',1,'RNN layers');

cmd:text('Optimization parameters');
cmd:option('-batch',1000,'Batch size (Adjust base on GRAM and dataset size)');
cmd:option('-lr',1e-3,'Learning rate');
cmd:option('-decay',100,'Learning rate decay in epochs');
cmd:option('-epochs',200,'Epochs');
params=cmd:parse(arg);

require 'nn'
require 'cutorch'
require 'cunn' 
require 'nngraph'
require '../utils/optim_updates'
require '../utils/metric'
RNN=require('../utils/word_RNN');
require '../utils/utils'


print('Loading dataset');
function sequence_length(seq)
	local v=seq:gt(0):long():sum(2):view(-1):long();
	return v;
end
dataset=torch.load(params.data);
dataset.caption.tokens,params.caption_dictionary=encode_sents3(dataset.caption.caption);
collectgarbage();

params.nhsent=params.nh*params.nlayers*2; --Again using both cell and hidden of LSTM
params.nhoutput=1;
params.nhimage=dataset.image.fvs:size(2);

dataset_test=torch.load(params.data_test);
dataset_test.caption.tokens,_=encode_sents3(dataset_test.caption.caption,params.caption_dictionary);
collectgarbage();
params.ncaps_per_im=5;



print('Initializing session');
paths.mkdir('sessions')
Session=require('../utils/session_manager');
session=Session:init('./sessions');
basedir=session:new(params);
paths.mkdir(paths.concat(basedir,'model'));
log_file=paths.concat(basedir,string.format('log.txt',1));
function Log(msg)
	local f = io.open(log_file, "a")
	print(msg);
	f:write(msg..'\n');
	f:close()
end


--Network definitions
Log('Initializing models');

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
function imcap_CxI(nhC,nhI)
	local c=nn.Identity()();
	local i=nn.Identity()();
	local cc=nn.Dropout(0.3)(c);
	local ic=nn.Linear(nhI,nhC)(nn.Dropout(0.3)(nn.Normalize(2)(i)));
	local output=nn.MM(false,true)({cc,ic});
	return nn.gModule({c,i},{output});
end
C_embedding_net=wrap_net(nn.Sequential():add(nn.LookupTable(table.getn(params.caption_dictionary)+1,params.nhword)):add(nn.Dropout(0.5)),true);
C_embedding_net.w:uniform(-0.0001,0.0001); --initialize with sth small, so that UNK is small.
C_encoder_net=RNN:new(RNN.unit.lstm(params.nhword,params.nh,params.nhoutput,params.nlayers,0.5),math.max(dataset.caption.tokens:size(3),dataset_test.caption.tokens:size(3)),true);
multimodal_net=wrap_net(imcap_CxI(params.nhsent,params.nhimage),true);
--Criterion
criterion_im=nn.CrossEntropyCriterion():cuda();
criterion_cap=nn.CrossEntropyCriterion():cuda();
--Create dummy states and gradients
dummy_state=torch.DoubleTensor(params.nhsent):fill(0):cuda();
dummy_output=torch.DoubleTensor(params.nhoutput):fill(0):cuda();


--Optimization
Log('Setting up optimization');
niter_per_epoch=math.ceil(dataset.image.fvs:size(1)/params.batch);
max_iter=params.epochs*niter_per_epoch;
Log(string.format('%d iter per epoch.',niter_per_epoch));
decay=math.exp(math.log(0.1)/params.decay/niter_per_epoch);
opt_encoder_C={learningRate=params.lr,decay=decay};
opt_embedding_C={learningRate=params.lr,decay=decay};
opt_multimodal={learningRate=params.lr,decay=decay};
C_encoder_net:training();
C_embedding_net.deploy:training();
multimodal_net.deploy:training();

--Batch function. Sample 1 caption per image for N unique images, and set labels for images and captions that match 
function dataset:batch_train(batch_size)
	local timer = torch.Timer();
	local nim=#self.image.imname;
	local iminds=torch.randperm(nim)[{{1,batch_size}}]:long(); --sample images without repeat
	local tokens_cap=torch.LongTensor(batch_size,self.caption.tokens:size(3)):fill(0); --select one caption per image
	local labels=torch.LongTensor(batch_size);
	for i=1,batch_size do
		local imname=self.image.imname[iminds[i]];
		local ind=self.caption.lookup[imname];
		local ncaps=#self.caption.caption[ind];
		tokens_cap[i]=self.caption.tokens[ind][torch.random(ncaps)];
		labels[i]=i;
	end
	local fv_im=self.image.fvs:index(1,iminds);
	local fv_sorted_c=sort_by_length_left_aligned(tokens_cap,true);
	return fv_sorted_c,fv_im:cuda(),labels:cuda();
end
--Using the first 5 captions for evaluation.
function dataset_test:batch_eval(sc,ec,si,ei)
	local timer = torch.Timer();
	local tokens_cap=torch.LongTensor(params.ncaps_per_im*(ec-sc+1),self.caption.tokens:size(3)):fill(0); --select one caption per image
	for i=1,ec-sc+1 do
		local imname=self.image.imname[i+sc-1];
		local ind=self.caption.lookup[imname];
		tokens_cap[{{params.ncaps_per_im*(i-1)+1,params.ncaps_per_im*i}}]=self.caption.tokens[ind][{{1,params.ncaps_per_im}}];
	end
	local fv_im=self.image.fvs[{{si,ei}}];
	local fv_sorted_c=sort_by_length_left_aligned(tokens_cap,true);
	return fv_sorted_c,fv_im:cuda();
end
--Objective function
running_avg=nil;
function ForwardBackward()
	local timer = torch.Timer();
	--clear gradients--
	C_embedding_net.dw:zero();
	C_encoder_net.dw:zero();
	multimodal_net.dw:zero();
	--Grab a batch--
	local fv_C,fv_im,labels=dataset:batch_train(params.batch);
	--Forward/backward
	local embedding_C=C_embedding_net.deploy:forward(fv_C.words);
	local state_C,_=C_encoder_net:forward(torch.repeatTensor(dummy_state:fill(0),fv_C.map_to_sequence:size(1),1),embedding_C,fv_C.batch_sizes);
	local tv_C=state_C:index(1,fv_C.map_to_sequence);
	local scores=multimodal_net.deploy:forward({tv_C,fv_im});
	local f=criterion_im:forward(scores,labels)+criterion_cap:forward(scores:t(),labels); --image retrieval, caption retrieval
	
	local dscores=criterion_im:backward(scores,labels)+criterion_cap:backward(scores:t(),labels):t();
	local tmp=multimodal_net.deploy:backward({tv_C,fv_im},dscores);
	local dstate_C=tmp[1]:index(1,fv_C.map_to_rnn);
	local _,dembedding_C=C_encoder_net:backward(torch.repeatTensor(dummy_state:fill(0),params.batch,1),embedding_C,fv_C.batch_sizes,dstate_C,dummy_output);
	C_embedding_net.deploy:backward(fv_C.words,dembedding_C);
	
	C_encoder_net.dw:clamp(-2,2);
	if running_avg then
		running_avg=running_avg*0.95+f*0.05;
	else
		running_avg=f;
	end
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

--Optimization loop
Log('Begin optimizing');
local timer = torch.Timer();
for i=1,max_iter do
	--Print statistics every 1 epoch
	if i%niter_per_epoch==0 then
		Log(string.format('epoch %d/%d, trainloss %f, learning rate %f, time %f',i/niter_per_epoch,params.epochs,running_avg,opt_encoder_C.learningRate,timer:time().real));
	end
	--Save&eval every 10 epochs
	if i%(niter_per_epoch*5)==0 then
		torch.save(paths.concat(basedir,'model',string.format('model_epoch%d.t7',i/(niter_per_epoch))),{C_encoder_net=C_encoder_net.net,C_embedding_net=C_embedding_net.net,multimodal_net=multimodal_net.net});
		--do some testing here
		C_encoder_net:evaluate();
		C_embedding_net.deploy:evaluate();
		multimodal_net.deploy:evaluate();
		local npts_test=math.min(#dataset_test.image.imname,params.K);
		local scores_test=torch.zeros(npts_test*params.ncaps_per_im,npts_test):double();
		for i=1,npts_test,params.batch do
			for j=1,npts_test,params.batch do
				--print(string.format('\ttesting %d/%d %f',i,npts_test,timer:time().real));
				local ri=math.min(i+params.batch-1,npts_test);
				local rj=math.min(j+params.batch-1,npts_test);
				scores_test[{{5*(i-1)+1,5*ri},{j,rj}}]=Forward_test(i,ri,j,rj);
			end
		end
		--figure out the ground truth
		local gt_im=torch.zeros(npts_test,params.ncaps_per_im);
		local gt_text=torch.zeros(npts_test,params.ncaps_per_im);
		for i=1,npts_test do
			gt_im[i]:fill(i);
			gt_text[i]=torch.range(1,params.ncaps_per_im)+(i-1)*params.ncaps_per_im;
		end
		gt_im=gt_im:view(-1):long();
		gt_text=gt_text:long();
		local im_r_1=metric.accuracy_N(scores_test,gt_im,1);
		local im_r_5=metric.accuracy_N(scores_test,gt_im,5);
		local im_r_10=metric.accuracy_N(scores_test,gt_im,10);
		local cap_r_1=metric.accuracy_NM(scores_test:t(),gt_text,1);
		local cap_r_5=metric.accuracy_NM(scores_test:t(),gt_text,5);
		local cap_r_10=metric.accuracy_NM(scores_test:t(),gt_text,10);
		Log(string.format('test acc\t im: %f %f %f\t cap: %f %f %f',im_r_1,im_r_5,im_r_10,cap_r_1,cap_r_5,cap_r_10));
		C_encoder_net:training();
		C_embedding_net.deploy:training();
		multimodal_net.deploy:training();
	end
	--Collect garbage every 10 iterations
	if i%10==0 then
		collectgarbage();
	end
	ForwardBackward();
	--Update parameters
	rmsprop(C_encoder_net.w,C_encoder_net.dw,opt_encoder_C);
	rmsprop(C_embedding_net.w,C_embedding_net.dw,opt_embedding_C);
	rmsprop(multimodal_net.w,multimodal_net.dw,opt_multimodal);
	--Learning rate decay
	opt_encoder_C.learningRate=opt_encoder_C.learningRate*opt_encoder_C.decay;
	opt_embedding_C.learningRate=opt_embedding_C.learningRate*opt_embedding_C.decay;
	opt_multimodal.learningRate=opt_multimodal.learningRate*opt_multimodal.decay;
	
end
