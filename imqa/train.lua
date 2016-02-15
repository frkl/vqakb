cmd = torch.CmdLine();
cmd:text('Train a VQA model');
cmd:text('Options')
cmd:option('-data','../dataset/imqa/dataset_train.t7','Dataset for training');
cmd:option('-model','model/lstm.t7','Model filename');
cmd:text();
cmd:option('-nhword',200,'Word embedding size');
cmd:option('-nh',512,'RNN size');
cmd:option('-nlayers',2,'RNN layers');
cmd:option('-nhcommon',1024,'Common embedding size');
cmd:text();
cmd:option('-batch',500,'Batch size (Adjust base on GRAM)');
cmd:option('-lr',3e-4,'Learning rate');
cmd:option('-epochs',300,'Epochs');
cmd:option('-lr_decay',200,'After lr_decay epochs lr reduces to 0.1*lr');
cmd:option('-vt',false,'Internal GPULock.')
params=cmd:parse(arg);
--print(params)

nhword=params.nhword;
nh=params.nh;
nlayers=params.nlayers;
nhcommon=params.nhcommon;
nhsent=nlayers*nh*2;

batch_size=params.batch;
learning_rate=params.lr;
nepochs=params.epochs;
nepochs_lr_10decay=params.lr_decay;


dataset_path=params.data;  
model_path=params.model; 
paths.mkdir('model/save');

require 'nn'
require 'cutorch'
require 'cunn'
require 'nngraph'
require 'optim'
require '../utils/optim_updates'
require '../utils/RNNUtils'
LSTM=require('../utils/LSTM');

--GPU Lock
if params.vt then
	vtutils=require('vtutils');
	id=vtutils.obtain_gpu_lock_id.get_id();
	print(id);
	cutorch.setDevice(id+1);
end

print('Loading dataset');
dataset=torch.load(dataset_path);
vocabulary_size_q=table.getn(dataset['question_dictionary']);
nhimage=dataset['image_fvs']:size(2);
noutput=table.getn(dataset['answer_dictionary']);

print('Right aligning words');
dataset['question_lengths']=sequence_length(dataset['question_tokens']);
dataset['question_tokens']=right_align(dataset['question_tokens'],dataset['question_lengths']);
collectgarbage();

print('Initializing models');
nhdummy=1;
--Network definitions
embedding_net_q=nn.Sequential():add(nn.Linear(vocabulary_size_q,nhword)):add(nn.Dropout(0.5)):add(nn.Tanh()):cuda();
encoder_net_q=LSTM.lstm(nhword,nh,nhdummy,nlayers,0.5):cuda();
function AxB(nhA,nhB,nhcommon,dropout)
	dropout = dropout or 0 
	local q=nn.Identity()();
	local i=nn.Identity()();
	local qc=nn.Tanh()(nn.Linear(nhA,nhcommon)(nn.Dropout(dropout)(q)));
	local ic=nn.Tanh()(nn.Linear(nhB,nhcommon)(nn.Dropout(dropout)(nn.Normalize(2)(i))));
	local output=nn.CMulTable()({qc,ic});
	return nn.gModule({q,i},{output});
end
multimodal_net=nn.Sequential():add(AxB(nhsent,nhimage,nhcommon,0.5)):add(nn.Dropout(0.5)):add(nn.Linear(nhcommon,noutput)):cuda();
--criterion
criterion=nn.CrossEntropyCriterion():cuda();
--weights
embedding_w_q,embedding_dw_q=embedding_net_q:getParameters();
embedding_w_q:uniform(-0.08, 0.08);
encoder_w_q,encoder_dw_q=encoder_net_q:getParameters();
encoder_w_q:uniform(-0.08, 0.08);
multimodal_w,multimodal_dw=multimodal_net:getParameters();
multimodal_w:uniform(-0.08, 0.08);
--Create dummies so the originals are not contaminated with inputs/outputs during training.
embedding_net_q_dummy=embedding_net_q:clone('weight','bias','gradWeight','gradBias');
multimodal_net_dummy=multimodal_net:clone('weight','bias','gradWeight','gradBias');
encoder_net_buffer_q=dupe_rnn(encoder_net_q,dataset['question_tokens']:size(2));
--Create dummy gradients
dummy_state_q=torch.DoubleTensor(nhsent):fill(0):cuda();
dummy_output_q=torch.DoubleTensor(nhdummy):fill(0):cuda();

--Optimization parameters
print('Setting up optimization');
niter_per_epoch=math.ceil(dataset['question_tokens']:size(1)/batch_size);
print(string.format('%d iter per epoch.',niter_per_epoch));
opt_encoder={};
opt_encoder.maxIter=nepochs*niter_per_epoch;
opt_encoder.learningRate=learning_rate;
opt_encoder.decay=math.exp(math.log(0.1)/nepochs_lr_10decay/niter_per_epoch);
opt_embedding={};
opt_embedding.maxIter=nepochs*niter_per_epoch;
opt_embedding.learningRate=learning_rate;
opt_embedding.decay=math.exp(math.log(0.1)/nepochs_lr_10decay/niter_per_epoch);
opt_multimodal={};
opt_multimodal.maxIter=nepochs*niter_per_epoch;
opt_multimodal.learningRate=learning_rate;
opt_multimodal.decay=math.exp(math.log(0.1)/nepochs_lr_10decay/niter_per_epoch);


--Batch function
function dataset:next_batch_train()
	local timer = torch.Timer();
	local nqs=dataset['question_tokens']:size(1);
	local qinds=torch.LongTensor(batch_size):fill(0);
	local iminds=torch.LongTensor(batch_size):fill(0);
	for i=1,batch_size do
		qinds[i]=torch.random(nqs);
		iminds[i]=dataset['question_imids'][qinds[i]];
	end
	local fv_sorted_q=sort_encoding_onehot_right_align(dataset['question_tokens']:index(1,qinds),dataset['question_lengths']:index(1,qinds),vocabulary_size_q);
	fv_sorted_q.onehot=fv_sorted_q.onehot:cuda();
	fv_sorted_q.map_to_rnn=fv_sorted_q.map_to_rnn:cuda();
	fv_sorted_q.map_to_sequence=fv_sorted_q.map_to_sequence:cuda();
	local fv_im=dataset['image_fvs']:index(1,iminds);
	local labels=dataset['answer_labels']:index(1,qinds);
	return fv_sorted_q,fv_im:cuda(),labels:cuda(),batch_size;
end


--Objective function
running_avg=0;
function ForwardBackward()
	local timer = torch.Timer();
	--clear gradients--
	encoder_dw_q:zero();
	embedding_dw_q:zero();
	multimodal_dw:zero();
	--grab a batch--
	local fv_sorted_q,fv_im,labels,batch_size=dataset:next_batch_train();
	local question_max_length=fv_sorted_q.batch_sizes:size(1);
	--embedding forward--
	local word_embedding_q=embedding_net_q_dummy:forward(fv_sorted_q.onehot);
	--encoder forward--
	local states_q,_=rnn_forward(encoder_net_buffer_q,torch.repeatTensor(dummy_state_q:fill(0),batch_size,1),word_embedding_q,fv_sorted_q.batch_sizes);
	--multimodal/criterion forward--
	local tv_q=states_q[question_max_length+1]:index(1,fv_sorted_q.map_to_sequence);
	local scores=multimodal_net_dummy:forward({tv_q,fv_im});
	local f=criterion:forward(scores,labels);
	--multimodal/criterion backward--
	local dscores=criterion:backward(scores,labels);
	local tmp=multimodal_net_dummy:backward({tv_q,fv_im},dscores);
	local dtv_q=tmp[1]:index(1,fv_sorted_q.map_to_rnn);
	--encoder backward
	local _,dword_embedding_q=rnn_backward(encoder_net_buffer_q,dtv_q,dummy_output_q,states_q,word_embedding_q,fv_sorted_q.batch_sizes);
	--embedding backward--
	embedding_net_q_dummy:backward(fv_sorted_q.onehot,dword_embedding_q);
	--summarize f and gradient
	encoder_dw_q:clamp(-10,10);
	running_avg=running_avg*0.95+f*0.05;
end

--Optimization loop
print('Begin optimizing');
local timer = torch.Timer();
for i=1,opt_encoder.maxIter do
	if i%(niter_per_epoch*10)==0 then
		torch.save(string.format('model/save/lstm_save_epoch%d.t7',i/(niter_per_epoch)),{encoder_net_q=encoder_net_q,embedding_net_q=embedding_net_q,multimodal_net=multimodal_net,nhword=nhword,nh=nh,nhsent=nhsent,nhcommon=nhcommon,nlayers=nlayers,nhimage=nhimage,noutput=noutput});
	end
	if i%niter_per_epoch==0 then
		print(string.format('epoch %d/%d, trainloss %f, learning rate %f, time %f',i/niter_per_epoch,nepochs,running_avg,opt_encoder.learningRate,timer:time().real));
	end
	ForwardBackward();
	rmsprop(encoder_w_q,encoder_dw_q, opt_encoder);
	rmsprop(embedding_w_q,embedding_dw_q, opt_embedding);
	rmsprop(multimodal_w,multimodal_dw, opt_multimodal);
	opt_encoder.learningRate=opt_encoder.learningRate*opt_encoder.decay;
	opt_embedding.learningRate=opt_embedding.learningRate*opt_embedding.decay;
	opt_multimodal.learningRate=opt_multimodal.learningRate*opt_multimodal.decay;
end
print('Save model');
torch.save(model_path,{encoder_net_q=encoder_net_q,embedding_net_q=embedding_net_q,multimodal_net=multimodal_net,nhword=nhword,nh=nh,nhsent=nhsent,nhcommon=nhcommon,nlayers=nlayers,nhimage=nhimage,noutput=noutput});