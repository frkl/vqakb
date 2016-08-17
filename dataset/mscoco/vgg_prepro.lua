cmd = torch.CmdLine();
cmd:text('Extract features from a list of images');
cmd:text('using the 19-layer VGG net in caffe model zoo');
cmd:text('Options')
cmd:option('-pretrain','../../../pretrain/vgg_ilsvrc_19_layers/','Folder that has the pretrained model VGG_ILSVRC_19_layers.caffemodel and VGG_ILSVRC_19_layers_deploy.prototxt');
cmd:option('-lib','cudnn','Which convolution library to use nn/cudnn');
cmd:option('-out','vgg19_cudnn_deploy.t7','Output model filename');
params=cmd:parse(arg);

require 'nn'
require 'cunn'
require 'cutorch'
require 'nngraph'
require 'loadcaffe'
if params.lib=='cudnn' then
	require 'cudnn';
end

net=loadcaffe.load(params.pretrain..'VGG_ILSVRC_19_layers_deploy.prototxt',params.pretrain..'VGG_ILSVRC_19_layers.caffemodel',params.lib);

--add dimension flipping and mean subtraction
function prepro_net()
	local im=nn.Identity()();
	local im2=nn.MulConstant(255)(im);
	local R=nn.AddConstant(-123.68)(nn.Narrow(2,1,1)(im2));
	local G=nn.AddConstant(-116.779)(nn.Narrow(2,2,1)(im2));
	local B=nn.AddConstant(-103.939)(nn.Narrow(2,3,1)(im2));
	local output=nn.JoinTable(2)({B,G,R});
	return nn.gModule({im},{output});
end

net_lazy=nn.Sequential():add(prepro_net()):add(net);
net_lazy=net_lazy:cuda();

torch.save(params.out,net_lazy)