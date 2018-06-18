#! /usr/bin/env luajit

require 'torch'
require 'image'
require 'utils'

io.stdout:setvbuf('no')

cmd = torch.CmdLine()
cmd:option('-g', false, 'gpu enabled')
cmd:option('-test_samples', '1')
cmd:option('-net_fname', 'net/net_cpu_10.t7')
cmd:option('-data_dir', './data_mb2014_dark')
cmd:option('-net_name', 'simple_cnn')
cmd:option('-ws', 8)
cmd:option('-stride', 4)
opt = cmd:parse(arg)

if opt.g then
  require 'cunn'
  require 'cudnn'
  require 'cutorch'
else
  require 'nn'
end

torch.setdefaulttensortype('torch.FloatTensor')

-- Loading network
net = torch.load(opt.net_fname, 'ascii')

function feed_forward_conv(x, net, ws, s, gpuEn)
  assert(x:size(3) == 3) --x:size() = C x H x W
  C = x:size(3); H = x:size(4); W = x:size(5)
  if gpuEn then
    yt = torch.CudaTensor(1, 1, ws*ws, C, H+2*(ws-1), W+2*(ws-1)):fill(0)
    y = torch.CudaTensor(1, 1, C, H, W):fill(0)
    xp = torch.CudaTensor(2, 1, C, H+2*(ws-1), W+2*(ws-1)):fill(0)
  else
    yt = torch.Tensor(1, 1, ws*ws, C, H+2*(ws-1), W+2*(ws-1)):fill(0)
    y = torch.Tensor(1, 1, C, H, W):fill(0)
    xp = torch.Tensor(2, 1, C, H+2*(ws-1), W+2*(ws-1)):fill(0)
  end
  xp[{{},{},{},{ws,H+ws-1},{ws,W+ws-1}}]:copy(x)
  
  khc = 1; kwc = 1;
  kh = 1; kw = 1;
  while khc<=ws do
    kwc = 1
    kw = 1
    if kh > ws then
      if kh % ws == 0 then
        starth = ws
      else
        starth = kh % ws
      end
    else
      starth = kh
    end
    while kwc<=ws do
      if kw > ws then
        if kw % ws == 0 then
          startw = ws
        else
          startw = kw % ws
        end
      else
        startw = kw
      end
      for h=starth,H+ws-1,ws do
        for w=startw,W+ws-1,ws do
          yt[{{},{},{(khc-1)*ws+kwc},{},{h,h+ws-1},{w,w+ws-1}}] = net:forward(xp[{{},{},{},{h,h+ws-1},{w,w+ws-1}}])
          --yt[{{(khc-1)*ws+kwc},{},{h,h+ws-1},{w,w+ws-1}}]:cmul(m[{{1},{},{},{}}],xp[{{},{h,h+ws-1},{w,w+ws-1}}])
        end
      end
      kw = kw + s
      kwc = kwc + 1
    end
    kh = kh + s
    khc = khc + 1
  end
  for k=1,ws*ws do
    y[{{},{},{1},{},{}}] = y[{{},{},{1},{},{}}] + yt[{{},{},{k},{1},{ws,ws+H-1},{ws,ws+W-1}}]
    y[{{},{},{2},{},{}}] = y[{{},{},{2},{},{}}] + yt[{{},{},{k},{2},{ws,ws+H-1},{ws,ws+W-1}}]
    y[{{},{},{3},{},{}}] = y[{{},{},{3},{},{}}] + yt[{{},{},{k},{3},{ws,ws+H-1},{ws,ws+W-1}}]
  end
  y:div(ws*ws)
  return y
end

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
  net_out = torch.CudaTensor()
  --criterion = nn.MSECriterion():cuda()
  criterion = nn.AbsCriterion():cuda()
else
  x_batch = torch.Tensor()
  y_batch = torch.Tensor()
  net_out = torch.Tensor()
  --criterion = nn.MSECriterion()
  criterion = nn.AbsCriterion()
end
y2 = torch.Tensor()

ss = 512
gpu_en = opt.g and 'gpu' or 'cpu'
print('i \t light \t exposure \t error')
for i=1, #X do
  XX = X[i]
  YY = Y[i]
  
  --y_batch:resize(1, 1, YY[1]:size(1), YY[1]:size(2), YY[1]:size(3))
  y_batch:resize(1, 1, YY[1]:size(1), ss, ss)
  --net_out:resize(1, 1, YY[1]:size(1), YY[1]:size(2), YY[1]:size(3))
  net_out:resize(1, 1, YY[1]:size(1), ss, ss)
  --y2:resize(YY[1]:size(1), YY[1]:size(2), YY[1]:size(3))
  y2:resize(YY[1]:size(1), ss, ss)
  y_batch[1][1]:copy(YY[1][{{},{1,ss},{1,ss}}])
  y_batch:mul(2/255):add(-1)
  y2:copy(y_batch[1][1])
  for l=1, #XX do
    for e=1, XX[l]:size(1) do
      im0 = XX[l][e][1][{{},{1,ss},{1,ss}}]:squeeze()
      im1 = XX[l][e][2][{{},{1,ss},{1,ss}}]:squeeze()

      x_batch:resize(2, 1, im0:size(1), im0:size(2), im0:size(3))
      x_batch[1][1]:copy(im0:mul(2/255):add(-1))
      x_batch[2][1]:copy(im1:mul(2/255):add(-1))

      --net:forward(x_batch)
      net_out = feed_forward_conv(x_batch, net, opt.ws, opt.stride, opt.g)

      local err = criterion:forward(net_out, y_batch)

      print(i, l, e, err)

      image.save(('out/_%d_%d_%d_%s_%s_netoutput.png'):format(i, l, e, gpu_en, opt.net_name), net_out[1][1]:add(1):div(2))
      image.save(('out/_%d_%d_%d_%s_%s_im0.png'):format(i, l, e, gpu_en, opt.net_name), im0:add(1):div(2))
      --image.save(('out/_%d_%d_%d_%s_%s_im1.png'):format(i, l, e, gpu_en, opt.net_name), im1:add(1):div(2))
    end
  end
  image.save(('out/_%d_%s_%s_gt.png'):format(i, gpu_en, opt.net_name), y2:add(1):div(2))
end

