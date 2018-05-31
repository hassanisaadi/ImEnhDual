#! /usr/bin/env luajit

require 'torch'
require 'nn'
require 'cunn'

io.stdout:setvbuf('no')

cmd = torch.CmdLine()
cmd:option('-g', 0, 'gpu enabled')
cmd:option('-gpu', 1, 'gpu id')
cmd:option('-seed', 42, 'random seed')
cmd:option('-debug', false)
cmd:option('-test_samples', '3') -- Should be removed!!!!
 
opt = cmd:parse(arg)

torch.manualSeed(opt.seed)
if opt.g then
  cutorch.manualSeed(opt.seed)
  cutorch.setDevice(tonumber(opt.gpu))
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
metadata = fromfile(('%s/metab.bin'):format(data_dir))

X = {}
Y = {}
for n = 1, metadata:size(1)-opt.test_samples do
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


