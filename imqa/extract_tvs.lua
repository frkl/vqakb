cmd = torch.CmdLine();
cmd:text('Extract thought vectors (fc7) from a VQA model');
cmd:text('Options')
cmd:option('-model','model/lstm.t7','Model for testing');
cmd:option('-data','../dataset/imqa/dataset_val.t7','Dataset for testing');
cmd:option('-output','tvs.t7','Output filename');
cmd:option('-mode','image','Which tv to extract? Can be "image", "question" or "answer"');
cmd:option('-batch',1000,'Batch size (Adjust base on GRAM)');
cmd:option('-vt',false,'Internal GPULock.')
params=cmd:parse(arg);
--print(params)


--parameters
dataset_path=params.data;  
model_path=params.model;
batch_size=params.batch;
mode=params.mode;

require 'nn'
require 'cutorch'
require 'cunn'
require 'nngraph'
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
buffer_size_q=dataset['question_tokens']:size(2);

if mode=='question' then
	print('Right aligning words');
	dataset['question_lengths']=sequence_length(dataset['question_tokens']);
	dataset['question_tokens']=right_align(dataset['question_tokens'],dataset['question_lengths']);
	collectgarbage();
	vocabulary_size_q=table.getn(dataset['question_dictionary']);
end

--Parse model
print('Loading models');
model=torch.load(model_path);
embedding_net_q=model['embedding_net_q'];
encoder_net_q=model['encoder_net_q'];
multimodal_net=model['multimodal_net'];

nhimage=model['nhimage'];
nhsent=model['nhsent'];
noutput=model['noutput'];
nhcommon=model['nhcommon'];

embedding_net_q:evaluate();
encoder_net_q:evaluate();
multimodal_net:evaluate();

encoder_net_buffer_q=dupe_rnn(encoder_net_q,buffer_size_q);
dummy_state=torch.DoubleTensor(nhsent):fill(0):cuda();

--Batch functions
function dataset:next_batch_question(s,e)
	local timer = torch.Timer();
	local batch_size=e-s+1;
	local qinds=torch.LongTensor(batch_size):fill(0);
	for i=1,batch_size do
		qinds[i]=s+i-1;
	end
	
	local fv_sorted_q=sort_encoding_onehot_right_align(dataset['question_tokens']:index(1,qinds),dataset['question_lengths']:index(1,qinds),vocabulary_size_q);
	fv_sorted_q.onehot=fv_sorted_q.onehot:cuda();
	fv_sorted_q.map_to_rnn=fv_sorted_q.map_to_rnn:cuda();
	fv_sorted_q.map_to_sequence=fv_sorted_q.map_to_sequence:cuda();
	
	return fv_sorted_q,batch_size;
end
function dataset:next_batch_image(s,e)
	local timer = torch.Timer();
	local batch_size=e-s+1;
	local fv_im=dataset['image_fvs'][{{s,e},{}}];
	return fv_im:cuda(),batch_size;
end

--forward
function Forward(s,e)
	local timer = torch.Timer();
	--grab a batch--
	if mode=='question' then
		local fv_sorted_q,batch_size=dataset:next_batch_question(s,e);
		local question_max_length=fv_sorted_q.batch_sizes:size(1);
		local word_embedding_q=embedding_net_q:forward(fv_sorted_q.onehot);
		local states_q,_=rnn_forward(encoder_net_buffer_q,torch.repeatTensor(dummy_state,batch_size,1),word_embedding_q,fv_sorted_q.batch_sizes);
		local tv_q=states_q[question_max_length+1]:index(1,fv_sorted_q.map_to_sequence);
		local scores=multimodal_net:forward({tv_q,torch.CudaTensor(e-s+1,nhimage):zero()});
	else
		local fv_im,batch_size=dataset:next_batch_image(s,e);
		local scores=multimodal_net:forward({torch.CudaTensor(e-s+1,nhsent):zero(),fv_im});
	end
	local tvs;
	if mode=='question' then
		tvs=multimodal_net.modules[1].modules[4].output:double();
	elseif mode=='image' then
		tvs=multimodal_net.modules[1].modules[9].output:double();
	end
	return tvs;
end

--compute scores
print('Computing tvs');
local timer = torch.Timer();
if mode=='question' then
	nqs=dataset['question_tokens']:size(1);
	tvs=torch.DoubleTensor(nqs,nhcommon);
	for i=1,nqs,batch_size do
		print(string.format('%d/%d, time %f',i,nqs,timer:time().real));
		r=math.min(i+batch_size-1,nqs);
		tvs[{{i,r},{}}]=Forward(i,r);
	end
elseif mode=='image' then
	nim=dataset['image_fvs']:size(1);
	tvs=torch.DoubleTensor(nim,nhcommon);
	for i=1,nim,batch_size do
		print(string.format('%d/%d, time %f',i,nim,timer:time().real));
		r=math.min(i+batch_size-1,nim);
		tvs[{{i,r},{}}]=Forward(i,r);
	end
elseif mode=='answer' then
	tvs=multimodal_net.modules[3].weight:double();
end
print('Saving tvs');
torch.save(params.output,tvs);
