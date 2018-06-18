#! /usr/bin/env luajit

require 'sys'
require 'torch'
require 'optim'
require 'models'
require 'image'
require 'utils'
--require 'loadcaffe'

io.stdout:setvbuf('no')

cmd = torch.CmdLine()
cmd:option('-g', false, 'gpu enabled')
cmd:option('-gpu', 1, 'gpu id')
cmd:option('-seed', 42, 'random seed')
--cmd:option('-vggmodel', './pretrained_nets/VGG_ILSVRC_16_layers.caffemodel')
--cmd:option('-vggProto', './pretrained_nets/VGG_ILSVRC_16_layers_deploy.prototxt')
cmd:option('-bs', 32)
cmd:option('-patchSizeTr', 32)
cmd:option('-lr', 0.003)
cmd:option('-sceneNum', 22)
cmd:option('-beta1', 0.9)
cmd:option('-beta2', 0.999)
cmd:option('-data_dir', 'data_mb2014_dark')
cmd:option('-arch', '', 'network architecture')
cmd:option('-continueLearning', false)
cmd:option('-last_net_name', './net/a.t7')
cmd:option('-epoch_start', 1)
cmd:option('-epoch_end', 10)

opt = cmd:parse(arg)

if opt.g then
  require 'cunn'
  require 'cutorch'
  require 'cudnn'
else
  require 'nn'
end

torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(opt.seed)
if opt.g then
  cutorch.manualSeed(opt.seed)
  cutorch.setDevice(tonumber(opt.gpu))
  --vgg = loadcaffe.load(opt.vggProto, opt.vggmodel, 'cudnn')
else
  --vgg = loadcaffe.load(opt.vggProto, opt.vggmodel, 'nn')
end

-- Loading train data
X = {}
Y = {}
for n = 1, opt.sceneNum do
  local XX = {}
  light = 1
  while true do
    fname = ('%s/x_train_%d_%d.bin'):format(opt.data_dir, n, light)
    if not paths.filep(fname) then
      break
    end
    table.insert(XX, fromfile(fname))
    light = light + 1
  end
  table.insert(X, XX)
  
  fname = ('%s/y_train_%d.bin'):format(opt.data_dir, n)
  if paths.filep(fname) then
    table.insert(Y, fromfile(fname))
  end
end
print('Loaded trainig data.')

if opt.continueLearning then
  assert(string.find(opt.last_net_name, opt.arch))
  net = torch.load(opt.last_net_name, 'ascii')
  criterion = opt.g and nn.AbsCriterion():cuda() or nn.AbsCriterion()
  net_name = opt.arch
  epochS = opt.epoch_start
  epochE = opt.epoch_end
else
  if     opt.arch == 'simple_cnn' then net, criterion, net_name = simple_cnn(opt.g)
  elseif opt.arch == 'enc_dec'    then net, criterion, net_name = enc_dec(opt.g)
  else
    print('wrong architecture')
    os.exit()
  end
  epochS = 1
  epochE = opt.epoch_end
end
assert(net_name == opt.arch)
assert(opt.epoch_end > epochS)

print(net_name)
print(net)

params, gradParams = net:getParameters()
local optimState = {
                    learningRate = opt.lr,
                    learningRateDecay = 0,
                    weightDecay = 10e-6,
                    beta1 = opt.beta1,
                    beta2 = opt.beta2,
                    epsilon = 1e-8
                   }

if opt.g then
  x_batch_tr = torch.CudaTensor(2, opt.bs, 3, opt.patchSizeTr, opt.patchSizeTr)
  y_batch_tr = torch.CudaTensor(opt.bs, 3, opt.patchSizeTr, opt.patchSizeTr)
else
  x_batch_tr = torch.Tensor(2, opt.bs, 3, opt.patchSizeTr, opt.patchSizeTr)
  y_batch_tr = torch.Tensor(opt.bs, 3, opt.patchSizeTr, opt.patchSizeTr)
end
x_batch_tr_ = torch.FloatTensor(x_batch_tr:size())
y_batch_tr_ = torch.FloatTensor(y_batch_tr:size())

time = sys.clock()
err_tr = 0
err_tr_cnt = 0
for epoch=epochS, epochE do
  err_tr = 0
  err_tr_cnt = 0
  perm = torch.randperm(#X)
  for sample=1, #X do
    XX = X[perm[sample]]
    YY = Y[perm[sample]]
    for b=1, opt.bs do
      l_idx = torch.random(#XX)
      exp_idx = torch.random(XX[l_idx]:size(1))
      r = torch.random(XX[l_idx]:size(4)-opt.patchSizeTr+1) --top left pixel
      c = torch.random(XX[l_idx]:size(5)-opt.patchSizeTr+1)
      x_batch_tr_[1][b] = XX[l_idx][{{exp_idx},{1},{},{r,r+opt.patchSizeTr-1},{c,c+opt.patchSizeTr-1}}]
      x_batch_tr_[2][b] = XX[l_idx][{{exp_idx},{2},{},{r,r+opt.patchSizeTr-1},{c,c+opt.patchSizeTr-1}}]
      y_batch_tr_[b] = YY[1][{{},{r,r+opt.patchSizeTr-1},{c,c+opt.patchSizeTr-1}}]
    end
 
    x_batch_tr:copy(x_batch_tr_:mul(2/255):add(-1)) --normalization between -1,1
    y_batch_tr:copy(y_batch_tr_:mul(2/255):add(-1))

    --image.save('im0.png', x_batch_tr[1][2]:add(1):div(2))
    --image.save('im1.png', x_batch_tr[2][2]:add(1):div(2))
    --image.save('imy.png', y_batch_tr[2]:add(1):div(2))
    --os.exit()

    function feval(params)
      gradParams:zero()
      local outputs = net:forward(x_batch_tr)
      local loss = criterion:forward(outputs, y_batch_tr)
      local dloss_doutputs = criterion:backward(outputs, y_batch_tr)
      net:backward(x_batch_tr, dloss_doutputs)
  
      err_tr = err_tr + loss
      return loss, gradParams
    end
  
    optim.adam(feval, params, optimState)
    
    err_tr_cnt = err_tr_cnt + 1
  end --for sample=1, #X do

  print(epoch, err_tr / err_tr_cnt, sys.clock()-time)
  collectgarbage()
  net:clearState()
  if epoch % 100 == 0 then
    torch.save(('net/net_%s_%s_bs%d_p%d_e%d.t7'):format(opt.g and 'gpu' or 'cpu', net_name, opt.bs, opt.patchSizeTr, epoch), net, 'ascii')
  end
end -- for epoch=1, opt.epoch do
torch.save(('net/net_%s_%s_bs%d_p%d_final.t7'):format(opt.g and 'gpu' or 'cpu', net_name, opt.bs, opt.patchSizeTr), net, 'ascii')
