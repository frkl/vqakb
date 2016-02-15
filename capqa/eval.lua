cmd = torch.CmdLine();
cmd:text('Compute scores from Caption-QA model');
cmd:text('Options')
cmd:option('-model','model/lstm_bow.t7','Model for testing');
cmd:option('-data','../dataset/capqa/dataset_val.t7','Dataset for testing');
cmd:option('-output','scores.t7','Output filename');
cmd:option('-softmax',false,'Compute probability instead of scores');
cmd:option('-batch',1000,'Batch size (Adjust base on GRAM)');
cmd:option('-gt',false,'Report scores only on ground truth answer.');
cmd:option('-vt',false,'Internal GPULock.')
params=cmd:parse(arg);
--print(params)


--parameters
dataset_path=params.data;  
model_path=params.model;
batch_size=params.batch;

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

print('Right aligning words');
dataset['question_lengths']=sequence_length(dataset['question_tokens']);
dataset['question_tokens']=right_align(dataset['question_tokens'],dataset['question_lengths']);
collectgarbage();
vocabulary_size_q=table.getn(dataset['question_dictionary']);

--Parse model
print('Loading models');
model=torch.load(model_path);
embedding_net_q=model['embedding_net_q'];
encoder_net_q=model['encoder_net_q'];
multimodal_net=model['multimodal_net'];

nhcap=model['nhcap'];
nhsent=model['nhsent'];
noutput=model['noutput'];

embedding_net_q:evaluate();
encoder_net_q:evaluate();
multimodal_net:evaluate();

encoder_net_buffer_q=dupe_rnn(encoder_net_q,buffer_size_q);
dummy_state=torch.DoubleTensor(nhsent):fill(0):cuda();
softmax=nn.SoftMax():cuda();

print('Computing BoW');
dataset['caption_lengths']=sequence_length(dataset['caption_tokens']);
dataset['bow_cap']=bag_of_words(dataset['caption_tokens'],dataset['caption_lengths'],nhcap);
collectgarbage();

--Batch function
function dataset:next_batch_test(s,e)
	local timer = torch.Timer();
	local batch_size=e-s+1;
	local qinds=torch.LongTensor(batch_size):fill(0);
	local capinds=torch.LongTensor(batch_size):fill(0);
	for i=1,batch_size do
		qinds[i]=s+i-1;
		capinds[i]=(dataset['question_imids'][qinds[i]]-1)*dataset['ncaptions_per_im']+1; --always use the 1st caption
	end
	
	local fv_sorted_q=sort_encoding_onehot_right_align(dataset['question_tokens']:index(1,qinds),dataset['question_lengths']:index(1,qinds),vocabulary_size_q);
	fv_sorted_q.onehot=fv_sorted_q.onehot:cuda();
	fv_sorted_q.map_to_rnn=fv_sorted_q.map_to_rnn:cuda();
	fv_sorted_q.map_to_sequence=fv_sorted_q.map_to_sequence:cuda();
	
	local fv_cap=dataset['bow_cap']:index(1,capinds);
	
	return fv_sorted_q,fv_cap:cuda(),qids,batch_size;
end

--forward
function Forward(s,e)
	local timer = torch.Timer();
	--grab a batch--
	local fv_sorted_q,fv_cap,qids,batch_size=dataset:next_batch_test(s,e);
	local question_max_length=fv_sorted_q.batch_sizes:size(1);
	local word_embedding_q=embedding_net_q:forward(fv_sorted_q.onehot);
	local states_q,_=rnn_forward(encoder_net_buffer_q,torch.repeatTensor(dummy_state,batch_size,1),word_embedding_q,fv_sorted_q.batch_sizes);
	local tv_q=states_q[question_max_length+1]:index(1,fv_sorted_q.map_to_sequence);
	local scores=multimodal_net:forward({tv_q,fv_cap});
	if params.softmax==true then
		scores=softmax:forward(scores);
	end
	return scores:double();
end

--compute scores
print('Computing scores');
if params.gt then
	nqs=dataset['question_tokens']:size(1);
	scores=torch.DoubleTensor(nqs);
	for i=1,nqs,batch_size do
		print(string.format('%d/%d',i,nqs));
		r=math.min(i+batch_size-1,nqs);
		local tmp=Forward(i,r);
		scores[{{i,r}}]=tmp:gather(2,dataset['answer_labels'][{{i,r}}]:view(-1,1)):view(-1);
	end
else
	nqs=dataset['question_tokens']:size(1);
	scores=torch.DoubleTensor(nqs,noutput);
	for i=1,nqs,batch_size do
		print(string.format('%d/%d',i,nqs));
		r=math.min(i+batch_size-1,nqs);
		scores[{{i,r},{}}]=Forward(i,r);
	end
end

--Save scores as a matrix or json for vqa evaluation
print('Saving scores');
if string.match(params.output,'json')==nil or params.gt then
	torch.save(params.output,scores);
else
	_,pred=torch.max(scores,2);
	response={};
	for i=1,nqs do
		table.insert(response,{question_id=dataset['question_ids'][i],answer=dataset['answer_dictionary'][pred[{i,1}]]});
	end
	saveJson(params.output,response);
end
