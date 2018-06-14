#! /usr/bin/env luajit

require 'sys'
require 'torch'
require 'optim'
require 'models'
require 'image'
--require 'loadcaffe'

io.stdout:setvbuf('no')

cmd = torch.CmdLine()
cmd:option('-g', false, 'gpu enabled')
cmd:option('-gpu', 1, 'gpu id')
cmd:option('-seed', 42, 'random seed')
cmd:option('-debug', false)
--cmd:option('-vggmodel', './pretrained_nets/VGG_ILSVRC_16_layers.caffemodel')
--cmd:option('-vggProto', './pretrained_nets/VGG_ILSVRC_16_layers_deploy.prototxt')
cmd:option('-bs', 32)
cmd:option('-patchSizeTr', 32)
cmd:option('-epoch', 10)
cmd:option('-lr', 0.003)
cmd:option('-sceneNum', 22)
cmd:option('-beta1', 0.9)
cmd:option('-beta2', 0.999)
cmd:option('-data_dir', 'data_mb2014_dark')
cmd:option('-arch', '', 'network architecture')
cmd:option('-aug', false, 'Data Augmentation Enable')

cmd:option('-transX',2)
cmd:option('-transY',2)
cmd:option('-scaleR', 0.2)
cmd:option('-rotate', 28, 'degree')
cmd:option('-hflip', 0.5, 'chance of hflip')
cmd:option('-vflip', 0.5, 'chance of vflip')
cmd:option('-contrastR', 0.05, 'range of contrast')

opt = cmd:parse(arg)

if opt.aug then
  require 'libcv'
end

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

if     opt.arch == 'simple_cnn' then net, criterion, net_name = simple_cnn(opt.g)
elseif opt.arch == 'enc_dec'    then net, criterion, net_name = enc_dec(opt.g)
else
  print('wrong architecture')
  os.exit()
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

function normalize(x_b)
  -- normalization between -1,1
  if opt.aug then
    x_b_n = torch.FloatTensor(x_b:size())
    MR = x_b[1]:max(); mR = x_b[1]:min()
    MG = x_b[2]:max(); mG = x_b[2]:min()
    MB = x_b[3]:max(); mB = x_b[3]:min()
  
    x_b[1]:add(-mR):mul(2/(MR-mR)):add(-1)
    x_b[2]:add(-mG):mul(2/(MG-mG)):add(-1)
    x_b[3]:add(-mB):mul(2/(MB-mB)):add(-1)
    x_b_n:copy(x_b)
  else
    x_b_n:copy(x_b:mul(2/255):add(-1))
  end
  return x_b_n
end


time = sys.clock()
err_tr = 0
err_tr_cnt = 0
for epoch=1, opt.epoch do
  err_tr = 0
  err_tr_cnt = 0
  perm = torch.randperm(#X)
  for sample=1, #X do
    XX = X[perm[sample]]
    YY = Y[perm[sample]]
    for b=1, opt.bs do
      l_idx = torch.random(#XX)
      exp_idx = torch.random(XX[l_idx]:size(1))
      if opt.aug then
        --translation
        local transX = torch.uniform(0, opt.transX)
        local transY = torch.uniform(0, opt.transY)
        XX_trans1 = image.translate(XX[l_idx][{{exp_idx},{1},{},{},{}}]:squeeze(), transX, transY)
        XX_trans2 = image.translate(XX[l_idx][{{exp_idx},{2},{},{},{}}]:squeeze(), transX, transY)
        YY_trans  = image.translate(YY[1][{{},{},{}}]:squeeze(), transX, transY)
        --scale
        local W = torch.floor(torch.uniform(1-opt.scaleR, 1+opt.scaleR) * XX_trans1:size(2))
        local H = torch.floor(torch.uniform(1-opt.scaleR, 1+opt.scaleR) * XX_trans1:size(3))
        XX_scale1 = image.scale(XX_trans1, W, H)
        XX_scale2 = image.scale(XX_trans2, W, H)
        YY_scale  = image.scale(YY_trans , W, H)
        --rotate
        local phi = torch.uniform(-opt.rotate, opt.rotate) * math.pi / 180
        XX_rotat1 = image.rotate(XX_scale1, phi)
        XX_rotat2 = image.rotate(XX_scale2, phi)
        YY_rotat  = image.rotate(YY_scale , phi)
        --hflip
        if torch.uniform(0, 0.5) > opt.hflip then
          XX_hflip1 = image.hflip(XX_rotat1)
          XX_hflip2 = image.hflip(XX_rotat2)
          YY_hflip  = image.hflip(YY_rotat )
        else
          XX_hflip1 = XX_rotat1
          XX_hflip2 = XX_rotat2
          YY_hflip  = YY_rotat
        end
        --vflip
        if torch.uniform(0, 0.5) > opt.vflip then
          XX_vflip1 = image.vflip(XX_hflip1)
          XX_vflip2 = image.vflip(XX_hflip2)
          YY_vflip  = image.vflip(YY_hflip )
        else
          XX_vflip1 = XX_hflip1
          XX_vflip2 = XX_hflip2
          YY_vflip  = YY_hflip
        end
        --contrast
        local cont = torch.uniform(1-opt.contrastR, 1+opt.contrastR)
        XX_cont1 = torch.mul(XX_vflip1, cont)
        XX_cont2 = torch.mul(XX_vflip2, cont)
        --extracting patch
        r = torch.random(XX_cont1:size(2)-opt.patchSizeTr+1)
        c = torch.random(XX_cont1:size(3)-opt.patchSizeTr+1)
        x_batch_tr_[1][b] = normalize(XX_cont1[{{},{r,r+opt.patchSizeTr-1},{c,c+opt.patchSizeTr-1}}]:squeeze())
        x_batch_tr_[2][b] = normalize(XX_cont2[{{},{r,r+opt.patchSizeTr-1},{c,c+opt.patchSizeTr-1}}]:squeeze())
        y_batch_tr_[1][b] = normalize(YY_vflip[{{},{r,r+opt.patchSizeTr-1},{c,c+opt.patchSizeTr-1}}]:squeeze())
      else
        r = torch.random(XX[l_idx]:size(4)-opt.patchSizeTr+1) --top left pixel
        c = torch.random(XX[l_idx]:size(5)-opt.patchSizeTr+1)
        x_batch_tr_[1][b] = normalize(XX[l_idx][{{exp_idx},{1},{},{r,r+opt.patchSizeTr-1},{c,c+opt.patchSizeTr-1}}]:squeeze())
        x_batch_tr_[2][b] = normalize(XX[l_idx][{{exp_idx},{2},{},{r,r+opt.patchSizeTr-1},{c,c+opt.patchSizeTr-1}}]:squeeze())
        y_batch_tr_[1][b] = normalize(YY[1][{{},{r,r+opt.patchSizeTr-1},{c,c+opt.patchSizeTr-1}}]:squeeze())
      end
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
  end --for sample=1, #X do

  print(epoch, err_tr / err_tr_cnt, sys.clock()-time)
  collectgarbage()
  net:clearState()
  torch.save(('net/net_%s_%s_bs%d_p%d_%s_e%d.t7'):format(opt.g and 'gpu' or 'cpu', net_name, opt.bs, opt.patchSizeTr, opt.aug and 'aug' or '', epoch), net, 'ascii')
end -- for epoch=1, opt.epoch do

