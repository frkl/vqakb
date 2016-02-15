cmd = torch.CmdLine();
cmd:text('Extract features from a list of images');
cmd:text('using the 19-layer VGG net in caffe model zoo');
cmd:text('Options')
cmd:option('-pretrain','../../pretrain/vgg_ilsvrc_19_layers/','Folder that has the pretrained model VGG_ILSVRC_19_layers.caffemodel and VGG_ILSVRC_19_layers_deploy.prototxt');
cmd:option('-images','ims.json','A json file specifying a list of images');
cmd:option('-path','./','Path prefix of images');
cmd:option('-output','image_fvs.t7','Output file');
cmd:option('-batch',8,'Batch size, adjust according to GRAM');
cmd:text('Advanced');
cmd:option('-dropout',false,'Turn on dropout');
cmd:option('-nrepeat',0,'Repeat feature extraction N times. 0=off');
cmd:option('-mode','center','"center" for center crop; "average" for average over 10 crops; "10" for reporting all 10 crops.');
cmd:option('-vt',false,'Internal GPULock.')
params=cmd:parse(arg);
--print(params)



path_to_vgg=params.pretrain;
path_to_mscoco=params.path;
batch_size=params.batch;
f_imnames=params.images;
nrepeat=params.nrepeat;

ndims=4096;

require 'nn'
require 'cunn'
require 'cutorch'
require 'loadcaffe'
require 'image'

--GPU Lock
if params.vt then
	vtutils=require('vtutils');
	id=vtutils.obtain_gpu_lock_id.get_id();
	print(id);
	cutorch.setDevice(id+1);
end


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

imnames=loadJson(f_imnames);
--Find a unique list of images, and index everything.
list_of_images={};
for qid,im in pairs(imnames) do
	list_of_images[im]=0;
end
list_im_names={};
for im,junk in pairs(list_of_images) do
	table.insert(list_im_names,im);
end
net=loadcaffe.load(path_to_vgg..'VGG_ILSVRC_19_layers_deploy.prototxt',path_to_vgg..'VGG_ILSVRC_19_layers.caffemodel','nn');

if params.dropout then
	net:training();
else
	net:evaluate();
end
net=net:cuda();

imloader={};
function imloader:load(fname)
	self.im="rip";
	if not pcall(function () self.im=image.load(fname); end) then
		if not pcall(function () self.im=image.loadPNG(fname); end) then
			if not pcall(function () self.im=image.loadJPG(fname); end) then
				print(string.format('Error loading image: %s',fname));
			end
		end
	end
end
function loadim(imname)
	imloader:load(imname);
	local im=imloader.im;
	if im:size(1)==1 then
		--Grayscale
		local im2=torch.cat(im,im,1);
		im2=torch.cat(im2,im,1);
		im=im2;
	elseif im:size(1)==4 then
		--RGBA
		im=im[{{1,3},{},{}}];
	end
	
	local h=im:size(2);
	local w=im:size(3);
	local newh,neww;
	if h<=w then
		newh=256;
		neww=math.ceil((256/h*w)/2)*2;
	else
		newh=math.ceil((256/w*h)/2)*2;
		neww=256;
	end
	
	im=image.scale(im,neww,newh);
	--Change dynamic range to 0~255
	im=im*255;
	
	--Convert to BGR and subtract mean--
	local im2=im:clone();
	im2[{{3},{},{}}]=im[{{1},{},{}}]-123.68;
	im2[{{2},{},{}}]=im[{{2},{},{}}]-116.779;
	im2[{{1},{},{}}]=im[{{3},{},{}}]-103.939;
	
	local im4;
	if params.mode=='average' or params.mode=='10' then
		--return 10 crops
		im4=torch.DoubleTensor(10,3,224,224):fill(0);
		im4[1]=im2[{{},{newh/2-111,newh/2+112},{neww/2-111,neww/2+112}}];
		im4[2]=im2[{{},{newh/2-127,newh/2+96},{neww/2-127,neww/2+96}}];
		im4[3]=im2[{{},{newh/2-127,newh/2+96},{neww/2-95,neww/2+128}}];
		im4[4]=im2[{{},{newh/2-95,newh/2+128},{neww/2-127,neww/2+96}}];
		im4[5]=im2[{{},{newh/2-95,newh/2+128},{neww/2-95,neww/2+128}}];
		
		im4[6]=image.hflip(im2[{{},{newh/2-111,newh/2+112},{neww/2-111,neww/2+112}}]);
		im4[7]=image.hflip(im2[{{},{newh/2-127,newh/2+96},{neww/2-127,neww/2+96}}]);
		im4[8]=image.hflip(im2[{{},{newh/2-127,newh/2+96},{neww/2-95,neww/2+128}}]);
		im4[9]=image.hflip(im2[{{},{newh/2-95,newh/2+128},{neww/2-127,neww/2+96}}]);
		im4[10]=image.hflip(im2[{{},{newh/2-95,newh/2+128},{neww/2-95,neww/2+128}}]);
	else
		--return center crop;
		im4=im2[{{},{newh/2-111,newh/2+112},{neww/2-111,neww/2+112}}];
	end
	
	return im4;
end


sz=#list_im_names;
print(string.format('Processing %d images...',sz));
local timer = torch.Timer();
if params.mode=='center' and nrepeat==0 then
	fvs_ims=torch.DoubleTensor(sz,ndims):fill(0);
	for i=1,sz,batch_size do
		print(string.format('%d/%d, time %f',i,sz,timer:time().real));
		r=math.min(sz,i+batch_size-1);
		ims=torch.CudaTensor(r-i+1,3,224,224);
		for j=1,r-i+1 do
			ims[j]=loadim(path_to_mscoco..list_im_names[i+j-1]):cuda();
		end
		net:forward(ims);
		--fc7 in vgg19 seems to be layer 43.
		local tmp=net.modules[43].output:double();
		fvs_ims[{{i,r},{}}]=tmp;
		collectgarbage();
	end
elseif params.mode=='center' and nrepeat>0 then
	fvs_ims=torch.DoubleTensor(sz,nrepeat,ndims):fill(0);
	for i=1,sz,batch_size do
		print(string.format('%d/%d, time %f',i,sz,timer:time().real));
		r=math.min(sz,i+batch_size-1);
		ims=torch.CudaTensor(r-i+1,3,224,224);
		for j=1,r-i+1 do
			ims[j]=loadim(path_to_mscoco..list_im_names[i+j-1]):cuda();
		end
		for j=1,nrepeat do
			net:forward(ims);
			--fc7 in vgg19 seems to be layer 43.
			local tmp=net.modules[43].output:double();
			fvs_ims[{{i,r},j,{}}]=tmp;
		end
		collectgarbage();
	end
elseif params.mode=='average' and nrepeat==0 then
	fvs_ims=torch.DoubleTensor(sz,ndims):fill(0);
	for i=1,sz,batch_size do
		print(string.format('%d/%d, time %f',i,sz,timer:time().real));
		r=math.min(sz,i+batch_size-1);
		ims=torch.CudaTensor(r-i+1,10,3,224,224);
		for j=1,r-i+1 do
			ims[j]=loadim(path_to_mscoco..list_im_names[i+j-1]):cuda();
		end
		ims=torch.reshape(ims,(r-i+1)*10,3,224,224);
		net:forward(ims);
		--fc7 in vgg19 seems to be layer 43.
		local tmp=net.modules[43].output:double();
		tmp=torch.reshape(tmp,r-i+1,10,ndims);
		tmp=torch.reshape(tmp:mean(2),r-i+1,ndims);
		fvs_ims[{{i,r},{}}]=tmp;
		collectgarbage();
	end
elseif params.mode=='average' and nrepeat>0 then
	fvs_ims=torch.DoubleTensor(sz,nrepeat,ndims):fill(0);
	for i=1,sz,batch_size do
		print(string.format('%d/%d, time %f',i,sz,timer:time().real));
		r=math.min(sz,i+batch_size-1);
		ims=torch.CudaTensor(r-i+1,10,3,224,224);
		for j=1,r-i+1 do
			ims[j]=loadim(path_to_mscoco..list_im_names[i+j-1]):cuda();
		end
		ims=torch.reshape(ims,(r-i+1)*10,3,224,224);
		for j=1,nrepeat do
			net:forward(ims);
			--fc7 in vgg19 seems to be layer 43.
			local tmp=net.modules[43].output:double();
			tmp=torch.reshape(tmp,r-i+1,10,ndims);
			tmp=torch.reshape(tmp:mean(2),r-i+1,ndims);
			fvs_ims[{{i,r},j,{}}]=tmp;
		end
		collectgarbage();
	end
elseif params.mode=='10' and nrepeat==0 then
	fvs_ims=torch.DoubleTensor(sz,10,ndims):fill(0);
	for i=1,sz,batch_size do
		print(string.format('%d/%d, time %f',i,sz,timer:time().real));
		r=math.min(sz,i+batch_size-1);
		ims=torch.CudaTensor(r-i+1,10,3,224,224);
		for j=1,r-i+1 do
			ims[j]=loadim(path_to_mscoco..list_im_names[i+j-1]):cuda();
		end
		ims=torch.reshape(ims,(r-i+1)*10,3,224,224);
		net:forward(ims);
		--fc7 in vgg19 seems to be layer 43.
		local tmp=net.modules[43].output:double();
		tmp=torch.reshape(tmp,r-i+1,10,ndims);
		fvs_ims[{{i,r},{},{}}]=tmp;
		collectgarbage();
	end
elseif params.mode=='10' and nrepeat>0 then
	fvs_ims=torch.DoubleTensor(sz,nrepeat,10,ndims):fill(0);
	for i=1,sz,batch_size do
		print(string.format('%d/%d, time %f',i,sz,timer:time().real));
		r=math.min(sz,i+batch_size-1);
		ims=torch.CudaTensor(r-i+1,10,3,224,224);
		for j=1,r-i+1 do
			ims[j]=loadim(path_to_mscoco..list_im_names[i+j-1]):cuda();
		end
		ims=torch.reshape(ims,(r-i+1)*10,3,224,224);
		for j=1,nrepeat do
			net:forward(ims);
			--fc7 in vgg19 seems to be layer 43.
			local tmp=net.modules[43].output:double();
			tmp=torch.reshape(tmp,r-i+1,10,ndims);
			fvs_ims[{{i,r},j,{},{}}]=tmp;
		end
		collectgarbage();
	end
end

for i=1,sz do
	list_of_images[list_im_names[i]]=i;
end

torch.save(params.output,{fvs=fvs_ims,list_of_images=list_of_images,ims=list_im_names,mode=params.mode,nrepeat=nrepeat,dropout=params.dropout});


