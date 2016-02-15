npy4th=require 'npy4th'
function readAll(file)
    local f = io.open(file, "r")
    local content = f:read("*all")
    f:close()
    return content
end
cjson=require('cjson');
function loadJson(fname)
	local t=readAll(fname)
	return cjson.decode(t)
end

data=loadJson('captions.json')

--savejson({'captions':captions,'caption_tokens':caption_tokens,'words':words,'lookup':lookup,"ims":ims},'captions.json');
--left align
--+1
--lengths

nim=#data['captions'];
N=5;

max_length=0;

lengths=torch.zeros(N*nim);
for i=1,nim do
	for j=1,N do
		max_length=math.max(max_length,#data['caption_tokens'][i][j]);
		lengths[(i-1)*N+j]=#data['caption_tokens'][i][j];
	end
end
print(string.format("Sentence length: %d",max_length))

tokens=torch.zeros(N*nim,max_length);
for i=1,nim do
	for j=1,N do
		tokens[(i-1)*N+j][{{1,#data['caption_tokens'][i][j]}}]=torch.LongTensor(data['caption_tokens'][i][j])+1;
	end
end


dictionary=data['words'];
torch.save('mscoco_captions.t7',{max_length=max_length,lengths=lengths,tokens=tokens,dictionary=dictionary,ims=data['ims']});