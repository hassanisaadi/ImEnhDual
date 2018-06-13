#! /usr/bin/env luajit

function weights_init(m)
  local name = torch.type(m)
  if name:find('Convolution') then
    m.weight:normal(0.0, 0.02)
    m.bias:fill(0)
  elseif name:find('BatchNormalization') then
    if m.weight then m.weight:normal(1.0, 0.02) end
    if m.bias   then m.bias:fill(0) end
  end
end

function simple_cnn(gpu_en)
  net_l = nn.Sequential()
  net_r = nn.Sequential()
  net = nn.Sequential()
  net_p = nn.Parallel(1,2)
  if gpu_en then
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
    net:add(cudnn.Tanh())
  
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
    net:add(nn.Tanh())
  
    criterion = nn.MSECriterion()
  end

  net:apply(weights_init)
  net_name = 'simple_cnn'
  return net, criterion, net_name
end

function enc_dec(gpu_en)
  e_l = nn.Sequential()
  e_r = nn.Sequential()
  d_l = nn.Sequential()
  d_r = nn.Sequential()
  net_p = nn.Parallel(1,2)
  net = nn.Sequential()
  if gpu_en then
    e_l:add(cudnn.SpatialConvolution(3  , 8  , 3, 3, 1, 1, 1, 1))
    e_r:add(cudnn.SpatialConvolution(3  , 8  , 3, 3, 1, 1, 1, 1))
    e_l:add(cudnn.ReLU(true))
    e_r:add(cudnn.ReLU(true))
    e_l:add(cudnn.SpatialBatchNormalization(8))
    e_r:add(cudnn.SpatialBatchNormalization(8))
    e_l:add(cudnn.SpatialConvolution(8  , 32 , 3, 3, 1, 1, 1, 1))
    e_r:add(cudnn.SpatialConvolution(8  , 32 , 3, 3, 1, 1, 1, 1))
    e_l:add(cudnn.ReLU(true))
    e_r:add(cudnn.ReLU(true))
    e_l:add(cudnn.SpatialBatchNormalization(32))
    e_r:add(cudnn.SpatialBatchNormalization(32))

    e_l:add(cudnn.SpatialMaxPooling(2, 2, 2, 2, 0, 0))
    e_r:add(cudnn.SpatialMaxPooling(2, 2, 2, 2, 0, 0))

    e_l:add(cudnn.SpatialConvolution(32 , 64 , 3, 3, 1, 1, 1, 1))
    e_r:add(cudnn.SpatialConvolution(32 , 64 , 3, 3, 1, 1, 1, 1))
    e_l:add(cudnn.ReLU(true))
    e_r:add(cudnn.ReLU(true))
    e_l:add(cudnn.SpatialBatchNormalization(64))
    e_r:add(cudnn.SpatialBatchNormalization(64))
    e_l:add(cudnn.SpatialConvolution(64 , 128, 3, 3, 1, 1, 1, 1))
    e_r:add(cudnn.SpatialConvolution(64 , 128, 3, 3, 1, 1, 1, 1))
    e_l:add(cudnn.ReLU(true))
    e_r:add(cudnn.ReLU(true))
    e_l:add(cudnn.SpatialBatchNormalization(128))
    e_r:add(cudnn.SpatialBatchNormalization(128))

    e_l:add(cudnn.SpatialMaxPooling(2, 2, 2, 2, 0, 0))
    e_r:add(cudnn.SpatialMaxPooling(2, 2, 2, 2, 0, 0))

    e_l:add(cudnn.SpatialConvolution(128, 256, 3, 3, 1, 1, 1, 1))
    e_r:add(cudnn.SpatialConvolution(128, 256, 3, 3, 1, 1, 1, 1))
    e_l:add(cudnn.ReLU(true))
    e_r:add(cudnn.ReLU(true))
    e_l:add(cudnn.SpatialBatchNormalization(256))
    e_r:add(cudnn.SpatialBatchNormalization(256))
    e_l:add(cudnn.SpatialConvolution(256, 512, 3, 3, 1, 1, 1, 1))
    e_r:add(cudnn.SpatialConvolution(256, 512, 3, 3, 1, 1, 1, 1))
    e_l:add(cudnn.ReLU(true))
    e_r:add(cudnn.ReLU(true))
    e_l:add(cudnn.SpatialBatchNormalization(512))
    e_r:add(cudnn.SpatialBatchNormalization(512))

    e_l:add(cudnn.SpatialMaxPooling(2, 2, 2, 2, 0, 0))
    e_r:add(cudnn.SpatialMaxPooling(2, 2, 2, 2, 0, 0))

    net_p:add(e_l)
    net_p:add(e_r)

    net:add(net_p)
    net:add(cudnn.SpatialConvolution(1024, 1024, 3, 3, 1, 1, 1, 1))
    net:add(cudnn.ReLU(true))
    net:add(cudnn.SpatialBatchNormalization(1024))
    -- #1
    net:add(cudnn.SpatialFullConvolution(1024, 512, 4, 4, 2, 2, 1, 1, 0, 0))
    net:add(cudnn.ReLU(true))
    net:add(cudnn.SpatialBatchNormalization(512))

    net:add(cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1))
    net:add(cudnn.ReLU(true))
    net:add(cudnn.SpatialBatchNormalization(512))
    -- #2
    net:add(cudnn.SpatialFullConvolution(512 , 512, 4, 4, 2, 2, 1, 1, 0, 0))
    net:add(cudnn.ReLU(true))
    net:add(cudnn.SpatialBatchNormalization(512))

    net:add(cudnn.SpatialConvolution(512, 256, 3, 3, 1, 1, 1, 1))
    net:add(cudnn.ReLU(true))
    net:add(cudnn.SpatialBatchNormalization(256))
    -- #3
    net:add(cudnn.SpatialFullConvolution(256 , 256, 4, 4, 2, 2, 1, 1, 0, 0))
    net:add(cudnn.ReLU(true))
    net:add(cudnn.SpatialBatchNormalization(256))

    net:add(cudnn.SpatialConvolution(256, 128, 3, 3, 1, 1, 1, 1))
    net:add(cudnn.ReLU(true))
    net:add(cudnn.SpatialBatchNormalization(128))

    net:add(cudnn.SpatialConvolution(128, 64, 3, 3, 1, 1, 1, 1))
    net:add(cudnn.ReLU(true))
    net:add(cudnn.SpatialBatchNormalization(64))

    net:add(cudnn.SpatialConvolution(64 , 32, 3, 3, 1, 1, 1, 1))
    net:add(cudnn.ReLU(true))
    net:add(cudnn.SpatialBatchNormalization(32))

    net:add(cudnn.SpatialConvolution(32 , 3 , 3, 3, 1, 1, 1, 1))
    net:add(cudnn.Tanh())

    net:cuda()
    criterion = nn.MSECriterion():cuda()
  else
    e_l:add(nn.SpatialConvolution(3  , 8  , 3, 3, 1, 1, 1, 1))
    e_r:add(nn.SpatialConvolution(3  , 8  , 3, 3, 1, 1, 1, 1))
    e_l:add(nn.ReLU(true))
    e_r:add(nn.ReLU(true))
    e_l:add(nn.SpatialBatchNormalization(8))
    e_r:add(nn.SpatialBatchNormalization(8))
    e_l:add(nn.SpatialConvolution(8  , 8 , 3, 3, 1, 1, 1, 1))
    e_r:add(nn.SpatialConvolution(8  , 8 , 3, 3, 1, 1, 1, 1))
    e_l:add(nn.ReLU(true))
    e_r:add(nn.ReLU(true))
    e_l:add(nn.SpatialBatchNormalization(8))
    e_r:add(nn.SpatialBatchNormalization(8))

    e_l:add(nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0))
    e_r:add(nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0))

    e_l:add(nn.SpatialConvolution(8 , 8, 3, 3, 1, 1, 1, 1))
    e_r:add(nn.SpatialConvolution(8 , 8, 3, 3, 1, 1, 1, 1))
    e_l:add(nn.ReLU(true))
    e_r:add(nn.ReLU(true))
    e_l:add(nn.SpatialBatchNormalization(8))
    e_r:add(nn.SpatialBatchNormalization(8))

    e_l:add(nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0))
    e_r:add(nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0))

    net_p:add(e_l)
    net_p:add(e_r)

    net:add(net_p)
    net:add(nn.SpatialConvolution(16, 16, 3, 3, 1, 1, 1, 1))
    net:add(nn.ReLU(true))
    net:add(nn.SpatialBatchNormalization(16))
    -- #1
    net:add(nn.SpatialFullConvolution(16, 8, 4, 4, 2, 2, 1, 1, 0, 0))
    net:add(nn.ReLU(true))
    net:add(nn.SpatialBatchNormalization(8))

    net:add(nn.SpatialConvolution(8, 4, 3, 3, 1, 1, 1, 1))
    net:add(nn.ReLU(true))
    net:add(nn.SpatialBatchNormalization(4))
    -- #2
    net:add(nn.SpatialFullConvolution(4, 4, 4, 4, 2, 2, 1, 1, 0, 0))
    net:add(nn.ReLU(true))
    net:add(nn.SpatialBatchNormalization(4))

    net:add(nn.SpatialConvolution(4, 3, 3, 3, 1, 1, 1, 1))
    net:add(nn.Tanh())

    criterion = nn.MSECriterion()
  end

  net:apply(weights_init)
  net_name = 'enc_dec'
  return net, criterion, net_name
end

function gan(gpu_en)
  if gpu_en then

  else
  end


  net:apply(weights_init)
  net_name = 'gan'
  return netG, netD, criterionG, criterionD, net_name
end
