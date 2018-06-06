#! /usr/bin/env luajit

require 'sys'
require 'torch'
require 'optim'
--require 'loadcaffe'

io.stdout:setvbuf('no')

cmd = torch.CmdLine()
cmd:option('-g', false, 'gpu enabled')
cmd:option('-gpu', 1, 'gpu id')
cmd:option('-seed', 42, 'random seed')
cmd:option('-debug', false)
cmd:option('-vggmodel', './pretrained_nets/VGG_ILSVRC_16_layers.caffemodel')
cmd:option('-vggProto', './pretrained_nets/VGG_ILSVRC_16_layers_deploy.prototxt')
cmd:option('-bs', 32)
cmd:option('-patchSizeTr', 32)
cmd:option('-epoch', 10)
cmd:option('-lr', 0.003)
cmd:option('-sceneNum', 22)
cmd:option('-beta1', 0.9)
cmd:option('-beta2', 0.9)
opt = cmd:parse(arg)

if opt.g then
  require 'cunn'
  require 'cutorch'
  require 'cudnn'
else
  require 'nn'
end

torch.manualSeed(opt.seed)
if opt.g then
  cutorch.manualSeed(opt.seed)
  cutorch.setDevice(tonumber(opt.gpu))
  --vgg = loadcaffe.load(opt.vggProto, opt.vggmodel, 'cudnn')
else
  --vgg = loadcaffe.load(opt.vggProto, opt.vggmodel, 'nn')
end


function fromfile(fname)
   local file = io.open(fname .. '.dim')
   local dim = {}
   for line in file:lines() do
      table.insert(dim, tonumber(line))
   end
   if #dim == 1 and dim[1] == 0 then
      return torch.Tensor()
   end

   local file = io.open(fname .. '.type')
   local type = file:read('*all')

   local x
   local s
   if type == 'float32' then
      s = 1
      for i=1,#dim do
         s = s * dim[i]
      end
      --x = torch.FloatTensor(torch.FloatStorage(fname))
      x = torch.FloatTensor(torch.FloatStorage(s))
      torch.DiskFile(fname,'r'):binary():readFloat(x:storage())
   elseif type == 'int32' then
      s = 1
      for i=1,#dim do
         s = s * dim[i]
      end
      --x = torch.IntTensor(torch.IntStorage(fname))
      x = torch.IntTensor(torch.IntStorage(s))
      torch.DiskFile(fname,'r'):binary():readInt(x:storage())
   elseif type == 'int64' then
      s = 1
      for i=1,#dim do
         s = s * dim[i]
      end
      --x = torch.LongTensor(torch.LongStorage(fname))
      x = torch.LongTensor(torch.LongStorage(s))
      torch.DiskFile(fname, 'r'):binary():readLong(x:storage())
   else
      print(fname, type)
      assert(false)
   end

   x = x:reshape(torch.LongStorage(dim))
   return x
end

-- Loading train data
data_dir = 'data.mb.2014'

X = {}
Y = {}
for n = 1, opt.sceneNum do
  local XX = {}
  light = 1
  while true do
    fname = ('%s/x_train_%d_%d.bin'):format(data_dir, n, light)
    if not paths.filep(fname) then
      break
    end
    table.insert(XX, fromfile(fname))
    light = light + 1
  end
  table.insert(X, XX)
  
  fname = ('%s/y_train_%d.bin'):format(data_dir, n)
  if paths.filep(fname) then
    table.insert(Y, fromfile(fname))
  end
end
print('Loaded trainig data.')

net_l = nn.Sequential()
net_r = nn.Sequential()
net = nn.Sequential()
net_p = nn.Parallel(1,2)
if opt.g then
  net_l:add(cudnn.SpatialConvolution(3  , 16 , 3, 3, 1, 1, 1, 1))
  net_l:add(cudnn.ReLU(true))
  net_l:add(cudnn.SpatialConvolution(16 , 32 , 3, 3, 1, 1, 1, 1))
  net_l:add(cudnn.ReLU(true))
  net_l:add(cudnn.SpatialConvolution(32 , 64 , 3, 3, 1, 1, 1, 1))
  net_l:add(cudnn.ReLU(true))
  net_l:add(cudnn.SpatialConvolution(64 , 128, 3, 3, 1, 1, 1, 1))
  net_l:add(cudnn.ReLU(true))
  net_l:add(cudnn.SpatialConvolution(128, 256, 3, 3, 1, 1, 1, 1))
  net_l:add(cudnn.ReLU(true))
  net_l:add(cudnn.SpatialConvolution(256, 512, 3, 3, 1, 1, 1, 1))
  net_l:add(cudnn.ReLU(true))
  net_l:add(cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1))
  net_l:add(cudnn.ReLU(true))

  net_r:add(cudnn.SpatialConvolution(3  , 16 , 3, 3, 1, 1, 1, 1))
  net_r:add(cudnn.ReLU(true))
  net_r:add(cudnn.SpatialConvolution(16 , 32 , 3, 3, 1, 1, 1, 1))
  net_r:add(cudnn.ReLU(true))
  net_r:add(cudnn.SpatialConvolution(32 , 64 , 3, 3, 1, 1, 1, 1))
  net_r:add(cudnn.ReLU(true))
  net_r:add(cudnn.SpatialConvolution(64 , 128, 3, 3, 1, 1, 1, 1))
  net_r:add(cudnn.ReLU(true))
  net_r:add(cudnn.SpatialConvolution(128, 256, 3, 3, 1, 1, 1, 1))
  net_r:add(cudnn.ReLU(true))
  net_r:add(cudnn.SpatialConvolution(256, 512, 3, 3, 1, 1, 1, 1))
  net_r:add(cudnn.ReLU(true))
  net_r:add(cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1))
  net_r:add(cudnn.ReLU(true))
 
  net_p:add(net_l)
  net_p:add(net_r)
  
  net:add(net_p)

  net:add(cudnn.SpatialConvolution(1024, 1024, 3, 3, 1, 1, 1, 1))
  net:add(cudnn.ReLU(true))
  net:add(cudnn.SpatialConvolution(1024, 750 , 3, 3, 1, 1, 1, 1))
  net:add(cudnn.ReLU(true))
  net:add(cudnn.SpatialConvolution(750 , 512 , 3, 3, 1, 1, 1, 1))
  net:add(cudnn.ReLU(true))
  net:add(cudnn.SpatialConvolution(512, 512 , 3, 3, 1, 1, 1, 1))
  net:add(cudnn.ReLU(true))
  net:add(cudnn.SpatialConvolution(512 , 256 , 3, 3, 1, 1, 1, 1))
  net:add(cudnn.ReLU(true))
  net:add(cudnn.SpatialConvolution(256 , 128  , 3, 3, 1, 1, 1, 1))
  net:add(cudnn.ReLU(true))
  net:add(cudnn.SpatialConvolution(128 , 128  , 3, 3, 1, 1, 1, 1))
  net:add(cudnn.ReLU(true))
  net:add(cudnn.SpatialConvolution(128 , 64  , 3, 3, 1, 1, 1, 1))
  net:add(cudnn.ReLU(true))
  net:add(cudnn.SpatialConvolution(64 , 32  , 3, 3, 1, 1, 1, 1))
  net:add(cudnn.ReLU(true))
  net:add(cudnn.SpatialConvolution(32 , 16  , 3, 3, 1, 1, 1, 1))
  net:add(cudnn.ReLU(true))
  net:add(cudnn.SpatialConvolution(16 , 3  , 3, 3, 1, 1, 1, 1))

  net:cuda()
  criterion = nn.MSECriterion():cuda()
else
  net_l:add(nn.SpatialConvolution(3  , 16 , 3, 3, 1, 1, 1, 1))
  net_l:add(nn.ReLU(true))
  net_l:add(nn.SpatialConvolution(16 , 32 , 3, 3, 1, 1, 1, 1))
  net_l:add(nn.ReLU(true))

  net_r:add(nn.SpatialConvolution(3  , 16 , 3, 3, 1, 1, 1, 1))
  net_r:add(nn.ReLU(true))
  net_r:add(nn.SpatialConvolution(16 , 32 , 3, 3, 1, 1, 1, 1))
  net_r:add(nn.ReLU(true))
  
  net_p:add(net_l)
  net_p:add(net_r)
  
  net:add(net_p)

  net:add(nn.SpatialConvolution(64, 32, 3, 3, 1, 1, 1, 1))
  net:add(nn.ReLU(true))
  net:add(nn.SpatialConvolution(32, 3 , 3, 3, 1, 1, 1, 1))

  criterion = nn.MSECriterion()
end

print(net)

params, gradParams = net:getParameters()
local optimState = {
                    learningRate = opt.lr,
--                    learningRateDecay = 0,
--                    weightDecay = 0,
                    beta1 = opt.beta1,
                    beta2 = opt.beta2
--                    epsilon = 
                   }

if opt.g then
  x_batch_tr = torch.CudaTensor(2, opt.bs, 3, opt.patchSizeTr, opt.patchSizeTr)
  y_batch_tr = torch.CudaTensor(1, opt.bs, 3, opt.patchSizeTr, opt.patchSizeTr)
else
  x_batch_tr = torch.Tensor(2, opt.bs, 3, opt.patchSizeTr, opt.patchSizeTr)
  y_batch_tr = torch.Tensor(1, opt.bs, 3, opt.patchSizeTr, opt.patchSizeTr)
end
x_batch_tr_ = torch.FloatTensor(x_batch_tr:size())
y_batch_tr_ = torch.FloatTensor(y_batch_tr:size())

time = sys.clock()
err_tr = 0
err_tr_cnt = 0
for epoch=1, opt.epoch do
  err_tr = 0
  err_tr_cnt = 0
  perm = torch.randperm(#X)
  for sample=1, #X do
    --print(('sample = %d'):format(sample))
    scene_idx = perm[sample]
    XX = X[scene_idx]
    YY = Y[scene_idx]
    for b=1, opt.bs do
      --print(('#XX = %d'):format(#XX))
      --print(XX[1]:size())
      l_idx = torch.random(#XX)
      exp_idx = torch.random(XX[l_idx]:size(1))

      r = torch.random(XX[l_idx]:size(4)-opt.patchSizeTr+1)
      c = torch.random(XX[l_idx]:size(5)-opt.patchSizeTr+1)

      --print(XX[l_idx][{{exp_idx},{1},{},{r,r+opt.patchSizeTr-1},{c,c+opt.patchSizeTr-1}}]:size())
      x_batch_tr_[1][b] = XX[l_idx][{{exp_idx},{1},{},{r,r+opt.patchSizeTr-1},{c,c+opt.patchSizeTr-1}}]
      x_batch_tr_[2][b] = XX[l_idx][{{exp_idx},{2},{},{r,r+opt.patchSizeTr-1},{c,c+opt.patchSizeTr-1}}]
      --print(#YY)
      --print(YY[1]:size())
      y_batch_tr_[1][b]  = YY[1][{{},{r,r+opt.patchSizeTr-1},{c,c+opt.patchSizeTr-1}}]
      --print(YY[1][{{},{r,r+opt.patchSizeTr-1},{c,c+opt.patchSizeTr-1}}]:size())
      --print(y_batch_tr_[b]:size())
    end

    x_batch_tr:copy(x_batch_tr_)
    y_batch_tr:copy(y_batch_tr_)

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
  end

  print(epoch, err_tr / err_tr_cnt, sys.clock()-time)
  collectgarbage()
  net:clearState()
  torch.save(('net/net_%s_%d.t7'):format(opt.g and 'gpu' or 'cpu', epoch), net, 'ascii')
end -- for epoch=1, opt.epoch do

