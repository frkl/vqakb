cmd = torch.CmdLine();
cmd:text('Extract features from a list of images');
cmd:text('using the 19-layer VGG net in caffe model zoo');
cmd:text('Options')
cmd:option('-pretrain','vgg19_cudnn_deploy.t7','The pretrained fool-proof VGG model with integrated preprocessing');
cmd:option('-ims','imnames.json','A json file specifying a list of images');
cmd:option('-path','./images/','Path prefix of images');
cmd:option('-output','mscoco_vgg19_center.t7','Output file');
cmd:option('-batch',100,'Batch size, adjust according to GRAM');
params=cmd:parse(arg);
--print(params)


params.nhimage=4096;

require 'nn'
require 'cutorch'
require 'cunn'
require 'cudnn'
require 'nngraph'
require 'image'

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

print('Loading image list')
imnames=loadJson(params.ims);
--Find a unique list of images, and index everything.
imlookup={};
for qid,im in pairs(imnames) do
	imlookup[im]=0;
end
imnames={};
for im,junk in pairs(imlookup) do
	table.insert(imnames,im);
end
for i=1,#imnames do
	imlookup[imnames[i]]=i;
end


print('Loading model')
net=torch.load(params.pretrain);
net=net:cuda();
net:evaluate();

imloader={};
function imloader:load(fname)
	self.im=nil;
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
		--Grayscale image, replicate to get RGB
		im=torch.repeatTensor(im,3,1,1);
	elseif im:size(1)==4 then
		--RGBA, crop for only RGB
		im=im[{{1,3},{},{}}];
	end	
	--Resize to 256x256
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
	--Return center Crop
	return im[{{},{newh/2-111,newh/2+112},{neww/2-111,neww/2+112}}]:clone();
end


nim=#imnames;
print(string.format('Processing %d images...',nim));
fvs=torch.DoubleTensor(nim,params.nhimage):fill(0);
local timer = torch.Timer();
for i=1,nim,params.batch do
	--Load a batch of images
	r=math.min(nim,i+params.batch-1);
	local ims=torch.DoubleTensor(r-i+1,3,224,224);
	for j=1,r-i+1 do
		ims[j]=loadim(params.path..imnames[i+j-1]);
	end
	local t_im=timer:time().real;
	--Do a forward
	net:forward(ims:cuda());
	--fc7 in vgg19 seems to be layer 43.
	fvs[{{i,r},{}}]=net.modules[2].modules[43].output:double();
	collectgarbage();
	print(string.format('%d/%d,\ttime %f\t%f',r,nim,t_im,timer:time().real));
end

torch.save(params.output,{fvs=fvs,imname=imnames,lookup=imlookup});

