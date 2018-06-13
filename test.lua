#! /usr/bin/env luajit

require 'torch'
require 'image'

io.stdout:setvbuf('no')

cmd = torch.CmdLine()
cmd:option('-g', false, 'gpu enabled')
cmd:option('-test_samples', '1')
cmd:option('-net_fname', 'net/net_cpu_10.t7')
cmd:option('-data_dir', './data_mb2014_dark')
cmd:option('-net_name', 'simple_cnn')
opt = cmd:parse(arg)

if opt.g then
  require 'cunn'
  require 'cudnn'
  require 'cutorch'
else
  require 'nn'
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

-- Loading network
net = torch.load(opt.net_fname, 'ascii')

X = {}
Y = {}
-- Loading test data & inference
for n = 1, opt.test_samples do
  local XX = {}
  light = 1
  while true do
    fname = ('%s/x_test_%d_%d.bin'):format(opt.data_dir, n, light)
    if not paths.filep(fname) then
      break
    end
    table.insert(XX, fromfile(fname))
    light = light + 1
  end
  table.insert(X, XX)
  
  fname = ('%s/y_test_%d.bin'):format(opt.data_dir, n)
  if paths.filep(fname) then
    table.insert(Y, fromfile(fname))
  end
end
print('Loaded test data.')

if opt.g then
  x_batch = torch.CudaTensor()
  y_batch = torch.CudaTensor()
  criterion = nn.MSECriterion():cuda()
else
  x_batch = torch.Tensor()
  y_batch = torch.Tensor()
  criterion = nn.MSECriterion()
end
y2 = torch.Tensor()

gpu_en = opt.g and 'gpu' or 'cpu'
print('i \t light \t exposure \t error')
for i=1, #X do
  XX = X[i]
  YY = Y[i]
  
  local st = 513
  local en = 1024
  y_batch:resize(1, 1, YY[1]:size(1), en-st+1, en-st+1)
  y2:resize(YY[1]:size(1), en-st+1, en-st+1)
  y_batch[1][1]:copy(YY[1][{{},{st,en},{st,en}}])
  y_batch:mul(2/255):add(-1)
  y2:copy(y_batch[1][1])
  for l=1, #XX do
    for e=1, XX[l]:size(1) do
      im0 = XX[l][e][1][{{},{st,en},{st,en}}]:squeeze()
      im1 = XX[l][e][2][{{},{st,en},{st,en}}]:squeeze()

      x_batch:resize(2, 1, im0:size(1), im0:size(2), im0:size(3))
      x_batch[1][1]:copy(im0:mul(2/255):add(-1))
      x_batch[2][1]:copy(im1:mul(2/255):add(-1))

      net:forward(x_batch)

      local err = criterion:forward(net.output, y_batch)

      print(i, l, e, err)

      image.save(('out/_%d_%d_%d_%s_%s_netoutput.png'):format(i, l, e, gpu_en, opt.net_name), net.output[1]:add(1):div(2))
      image.save(('out/_%d_%d_%d_%s_%s_im0.png'):format(i, l, e, gpu_en, opt.net_name), im0:add(1):div(2))
      image.save(('out/_%d_%d_%d_%s_%s_im1.png'):format(i, l, e, gpu_en, opt.net_name), im1:add(1):div(2))
    end
  end
  image.save(('out/_%d_%s_%s_gt.png'):format(i, gpu_en, opt.net_name), y2:add(1):div(2))
end

