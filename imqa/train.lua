cmd = torch.CmdLine();
cmd:text('Train a VQA model');
cmd:text('Dataset')
cmd:option('-data','../dataset/imqa/dataset_train.t7','Dataset for training');
cmd:option('-data_test','../dataset/imqa/dataset_val.t7','Dataset for validation, it\'s nice to directly evaluate accuracy on the fly');

cmd:text('Model parameters');
cmd:option('-nanswers',1000,'Number of most frequent answers to use');
cmd:option('-nhword',200,'Word embedding size');
cmd:option('-nh',512,'RNN size');
cmd:option('-nlayers',2,'RNN layers');
cmd:option('-nhcommon',1024,'Common embedding size');

cmd:text('Optimization parameters');
cmd:option('-batch',500,'Batch size (Adjust base on GRAM and dataset size)');
cmd:option('-lr',3e-4,'Learning rate');
cmd:option('-decay',150,'Learning rate decay in epochs');
cmd:option('-epochs',300,'Epochs');
params=cmd:parse(arg);

require 'nn'
require 'cutorch'
require 'cunn' 
require 'nngraph'
require '../utils/optim_updates'
RNN=require('../utils/word_RNN');
require '../utils/utils'


print('Loading dataset');
function sequence_length(seq)
	local v=seq:gt(0):long():sum(2):view(-1):long();
	return v;
end
dataset=torch.load(params.data);
dataset.question.tokens,params.question_dictionary=encode_sents(dataset.question.question);
dataset.question.question=nil;
collectgarbage();
dataset.question.labels,params.answer_dictionary=encode_sents(dataset.question.answer,nil,params.nanswers);
dataset.question.answer=nil;
collectgarbage();

params.nhsent=params.nh*params.nlayers*2; --Using both cell and hidden of LSTM
params.noutput=params.nanswers;
params.nhoutput=1;
params.nhimage=dataset.image.fvs:size(2);

dataset_test=torch.load(params.data_test);
dataset_test.question.tokens,_=encode_sents(dataset_test.question.question,params.question_dictionary);
dataset_test.question.question=nil;
collectgarbage();
dataset_test.question.labels,_=encode_sents(dataset_test.question.answer,params.answer_dictionary,params.nanswers);
dataset_test.question.answer=nil;
collectgarbage();



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
function VQA_PW(nhA,nhB,nhcommon,noutput)
	local q=nn.Identity()();
	local i=nn.Identity()();
	local qc=nn.Tanh()(nn.Linear(nhA,nhcommon)(nn.Dropout(0.5)(q)));
	local ic=nn.Tanh()(nn.Linear(nhB,nhcommon)(nn.Dropout(0.5)(nn.Normalize(2)(i))));
	local output=nn.Linear(nhcommon,noutput)(nn.Dropout(0.5)(nn.CMulTable()({qc,ic})));
	return nn.gModule({q,i},{output});
end
Q_embedding_net=wrap_net(nn.Sequential():add(nn.LookupTable(table.getn(params.question_dictionary)+1,params.nhword)):add(nn.Dropout(0.5)),true);
Q_embedding_net.w:uniform(-0.0001,0.0001); --initialize with sth small, so that UNK is small.
Q_encoder_net=RNN:new(RNN.unit.lstm(params.nhword,params.nh,params.nhoutput,params.nlayers,0.5),math.max(dataset.question.tokens:size(2),dataset_test.question.tokens:size(2)),true);
multimodal_net=wrap_net(VQA_PW(params.nhsent,params.nhimage,params.nhcommon,params.noutput),true);
--Criterion
criterion=nn.CrossEntropyCriterion():cuda();
--Create dummy states and gradients
dummy_state=torch.DoubleTensor(params.nhsent):fill(0):cuda();
dummy_output=torch.DoubleTensor(params.nhoutput):fill(0):cuda();


--Optimization
Log('Setting up optimization');
niter_per_epoch=math.ceil(dataset.question.tokens:size(1)/params.batch);
max_iter=params.epochs*niter_per_epoch;
Log(string.format('%d iter per epoch.',niter_per_epoch));
decay=math.exp(math.log(0.1)/params.decay/niter_per_epoch);
opt_encoder_Q={learningRate=params.lr,decay=decay};
opt_embedding_Q={learningRate=params.lr,decay=decay};
opt_multimodal={learningRate=params.lr,decay=decay};
Q_encoder_net:training();
Q_embedding_net.deploy:training();
multimodal_net.deploy:training();

--Batch function
function dataset:batch_train(batch_size)
	local timer = torch.Timer();
	local nqs=self.question.tokens:size(1);
	local qinds=torch.LongTensor(batch_size):fill(0);
	local labels=torch.LongTensor(batch_size):fill(0);
	local iminds=torch.LongTensor(batch_size):fill(0);
	for i=1,batch_size do
		while true do
			qinds[i]=torch.random(nqs);
			local answers=self.question.labels[qinds[i]];
			local valid_answer_ind=answers:le(params.nanswers);
			if valid_answer_ind:long():sum()>0 then
				local answer_id=torch.range(1,self.question.labels:size(2))[valid_answer_ind]:long();
				iminds[i]=self.image.lookup[self.question.imname[qinds[i]]];
				labels[i]=answers[answer_id[torch.random(answer_id:size(1))]];
				break;
			end
		end
	end
	local fv_sorted_q=sort_by_length_left_aligned(self.question.tokens:index(1,qinds),true);
	local fv_im=self.image.fvs:index(1,iminds);
	return fv_sorted_q,fv_im:cuda(),labels:cuda();
end
function dataset_test:batch_eval(s,e)
	local timer = torch.Timer();
	local batch_size=e-s+1;
	local qinds=torch.LongTensor(batch_size):fill(0);
	local iminds=torch.LongTensor(batch_size):fill(0);
	for i=1,batch_size do
		qinds[i]=s+i-1;
		iminds[i]=self.image.lookup[self.question.imname[qinds[i]]];
	end
	local fv_sorted_q=sort_by_length_left_aligned(self.question.tokens:index(1,qinds),true);
	local fv_im=self.image.fvs:index(1,iminds);
	return fv_sorted_q,fv_im:cuda();
end
--Objective function
running_avg=nil;
function ForwardBackward()
	local timer = torch.Timer();
	--clear gradients--
	Q_embedding_net.dw:zero();
	Q_encoder_net.dw:zero();
	multimodal_net.dw:zero();
	--Grab a batch--
	local fv_Q,fv_I,labels=dataset:batch_train(params.batch);
	--Forward/backward
	local embedding_Q=Q_embedding_net.deploy:forward(fv_Q.words);
	local state_Q,_=Q_encoder_net:forward(torch.repeatTensor(dummy_state:fill(0),fv_Q.map_to_sequence:size(1),1),embedding_Q,fv_Q.batch_sizes);
	local tv_Q=state_Q:index(1,fv_Q.map_to_sequence);
	local scores=multimodal_net.deploy:forward({tv_Q,fv_I});
	local f=criterion:forward(scores,labels);
	
	local dscores=criterion:backward(scores,labels);
	local tmp=multimodal_net.deploy:backward({tv_Q,fv_I},dscores);
	local dstate_Q=tmp[1]:index(1,fv_Q.map_to_rnn);
	local _,dembedding_Q=Q_encoder_net:backward(torch.repeatTensor(dummy_state:fill(0),params.batch,1),embedding_Q,fv_Q.batch_sizes,dstate_Q,dummy_output);
	Q_embedding_net.deploy:backward(fv_Q.words,dembedding_Q);
	
	Q_encoder_net.dw:clamp(-5,5);
	if running_avg then
		running_avg=running_avg*0.95+f*0.05;
	else
		running_avg=f;
	end
end
function Forward_test(s,e)
	local fv_Q,fv_I=dataset_test:batch_eval(s,e);
	--Forward
	local embedding_Q=Q_embedding_net.deploy:forward(fv_Q.words);
	local state_Q,_=Q_encoder_net:forward(torch.repeatTensor(dummy_state:fill(0),fv_Q.map_to_sequence:size(1),1),embedding_Q,fv_Q.batch_sizes);
	local tv_Q=state_Q:index(1,fv_Q.map_to_sequence);
	local scores=multimodal_net.deploy:forward({tv_Q,fv_I});
	return scores;
end

--Optimization loop
Log('Begin optimizing');
local timer = torch.Timer();
for i=1,max_iter do
	--Print statistics every 1 epoch
	if i%niter_per_epoch==0 then
		Log(string.format('epoch %d/%d, trainloss %f, learning rate %f, time %f',i/niter_per_epoch,params.epochs,running_avg,opt_encoder_Q.learningRate,timer:time().real));
	end
	--Save every 10 epochs
	if i%(niter_per_epoch*10)==0 then
		torch.save(paths.concat(basedir,'model',string.format('model_epoch%d.t7',i/(niter_per_epoch))),{Q_encoder_net=Q_encoder_net.net,Q_embedding_net=Q_embedding_net.net,multimodal_net=multimodal_net.net});
		--do some testing here
		Q_encoder_net:evaluate();
		Q_embedding_net.deploy:evaluate();
		multimodal_net.deploy:evaluate();
		local npts_test=dataset_test.question.tokens:size(1);
		local pred_test=torch.zeros(npts_test):long();
		for i=1,npts_test,params.batch do
			--print(string.format('\ttesting %d/%d %f',i,npts_test,timer:time().real));
			local r=math.min(i+params.batch-1,npts_test);
			local score=Forward_test(i,r):double();
			--predict
			_,pred_test[{{i,r}}]=torch.max(score,2);
		end
		local correct_count=torch.repeatTensor(pred_test:view(-1,1),1,dataset_test.question.labels:size(2)):eq(dataset_test.question.labels):double():sum(2);
		correct_count[correct_count:gt(3)]=3;
		local acc_test=correct_count:mean()/3;
		Log(string.format('test acc:%f',acc_test));
		Q_encoder_net:training();
		Q_embedding_net.deploy:training();
		multimodal_net.deploy:training();
	end
	--Collect garbage every 10 iterations
	if i%10==0 then
		collectgarbage();
	end
	ForwardBackward();
	--Update parameters
	rmsprop(Q_encoder_net.w,Q_encoder_net.dw,opt_encoder_Q);
	rmsprop(Q_embedding_net.w,Q_embedding_net.dw,opt_embedding_Q);
	rmsprop(multimodal_net.w,multimodal_net.dw,opt_multimodal);
	--Learning rate decay
	opt_encoder_Q.learningRate=opt_encoder_Q.learningRate*opt_encoder_Q.decay;
	opt_embedding_Q.learningRate=opt_embedding_Q.learningRate*opt_embedding_Q.decay;
	opt_multimodal.learningRate=opt_multimodal.learningRate*opt_multimodal.decay;
	
end
