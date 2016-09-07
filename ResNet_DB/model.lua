require 'nn'
require 'cutorch'
require 'cunn'
require 'cudnn'
require 'optim'
--require 'torch'
local nninit = require 'nninit'
require 'cifar-dataset'

require 'ResidualDrop'
   
cudnn.fastest = true
cudnn.benchmark = true

function trainHyper(params)
  --[[cutorch.setDevice(1)   -- torch uses 1-based indexing for GPU, so +1
  cutorch.manualSeed(1)
  torch.manualSeed(1)
  torch.setnumthreads(1)]]
    
  final = io.open('final.txt','a')
  local f = io.open('./conf/cpt','r')
  local i = f:read('*n')
  f:close()
  
  
  local timer = torch.Timer()
  
  if i == 0 then 
    init()
  else    
    opt = torch.load('./conf/opt.conf')
    addtables = torch.load('./conf/addtables.th')
    all_result = torch.load('./conf/add_result.th')
    model = torch.load('./conf/model.net')
    loss = torch.load('./conf/loss.net')
    sgdState = torch.load('./conf/sgd.th')
    lrSchedule = torch.load('./conf/lr.th')
    dataTrain = torch.load('./conf/train.th')
    dataValid = torch.load('./conf/valid.th')
    dataTest = torch.load('./conf/test.th')
  end
  
  -- training
  i = i+1
  opt.batchSize = params['batchSize'] 
  train(params['depth'], i, timer) 
  
  -- get back error to send to spearmint
  local ferr = io.open('TMPresult.txt', "r")
  local err = ferr:read("*n")
  local f = io.open('./conf/cpt','w')
  f:write(i)
  ferr:close()
  final:close()
  f:close()
    
  return err 
end

-- as spearmint execute the
-- saving config
function save()
  print("Saving...")
  torch.save('./conf/opt.conf',opt)
  torch.save('./conf/addtables.th', addtables)
  torch.save('./conf/add_result.th', all_result)
  torch.save('./conf/model.net', model)
  torch.save('./conf/loss.net', loss)
  torch.save('./conf/sgd.th', sgdState)
  torch.save('./conf/lr.th', lrSchedule)
  torch.save('./conf/train.th', dataTrain)
  torch.save('./conf/test.th', dataTest)
  torch.save('./conf/valid.th', dataValid)
end

-- initialisation
function init()
    print('Initialisation...\n')
    opt = torch.load('./conf/opt.conf')
    print(opt)
    --[[cutorch.setDevice(opt.device+1)   -- torch uses 1-based indexing for GPU, so +1
    cutorch.manualSeed(1)
    torch.manualSeed(1)
    torch.setnumthreads(1) ]]           -- number of OpenMP threads, 1 is enough

    ---- Loading data ----
    if opt.dataset == 'svhn' then require 'svhn-dataset' 
    elseif opt.dataset == 'mnist' then require 'mnist-dataset' 
    else require 'cifar-dataset' end
    if opt.dataset == 'svhn' or opt.dataset == 'cifar10' or opt.dataset == 'cifar100' 
        then
        all_data, all_labels = get_Data(opt.dataset, opt.dataRoot, false) -- default do shuffling
        dataTrain = Dataset.LOADER(all_data, all_labels, "train", opt.batchSize, opt.augmentation)
        dataValid = Dataset.LOADER(all_data, all_labels, "valid", opt.batchSize)
        dataTest = Dataset.LOADER(all_data, all_labels, "test", opt.batchSize)
        else 
        mnist = require 'mnist'
        train = mnist.traindataset()
        test = mnist.testdataset()
        dataTrain = Dataset.LOADER(train, test, "train")
        dataValid = Dataset.LOADER(train,test, "valid")
        dataTest = Dataset.LOADER(train, test, "test")
        end
        
    --local mean,std = dataTrain:preprocess()
    --dataValid:preprocess(mean,std)
    --dataTest:preprocess(mean,std)
    print("Training set size:\t",   dataTrain:size())
    print("Validation set size:\t", dataValid:size())
    print("Test set size:\t\t",     dataTest:size())

    ---- Optimization hyperparameters ----
    sgdState = {
       weightDecay   = 1e-4,
       momentum      = 0.9,
       dampening     = 0,
       nesterov      = true,
    }
    -- Point at which learning rate decrease by 10x
    lrSchedule = {mnist    = {0.5, 0.75},
                  svhn     = {0.6, 0.7 }, 
                  cifar10  = {0.5, 0.75},
                  cifar100 = {0.5, 0.75}}

    ---- Buidling the residual network model ----
    print('Building model...\n')

    if opt.dataset == "cifar10" or opt.dataset == "cifar100" or opt.dataset == "svhn" then
        -- Input: 3x32x32
        model = nn.Sequential()
        ------> 3, 32,32
        model:add(cudnn.SpatialConvolution(3, 16, 3,3, 1,1, 1,1)
                    :init('weight', nninit.kaiming, {gain = 'relu'})
                    :init('bias', nninit.constant, 0))
        model:add(cudnn.SpatialBatchNormalization(16))
        model:add(cudnn.ReLU(true))
        ------> 16, 32,32   First Group
        for i=1,opt.N do   addResidualDrop(model, nil, 16)   end
        ------> 32, 16,16   Second Group
        addResidualDrop(model, nil, 16, 32, 2)
        for i=1,opt.N-1 do   addResidualDrop(model, nil, 32)   end
        ------> 64, 8,8     Third Group
        addResidualDrop(model, nil, 32, 64, 2)
        for i=1,opt.N-1 do   addResidualDrop(model, nil, 64)   end
        ------> 10, 8,8     Pooling, Linear, Softmax
        model:add(nn.SpatialAveragePooling(8,8)):add(nn.Reshape(64))
        if opt.dataset == 'cifar10' or opt.dataset == 'svhn' then
          model:add(nn.Linear(64, 10))
        elseif opt.dataset == 'cifar100' then
          model:add(nn.Linear(64, 100))
        else
          print('Invalid argument for dataset!')
        end
        model:add(cudnn.LogSoftMax())
        loss = nn.ClassNLLCriterion()
        model:cuda()
        loss:cuda()
      
    else
        require 'mnist/residual_mnist'
        modMnist = require('mnist/model_mnist')
        
        model, loss = modMnist.residual(opt.N)
        --[[N = 15
        -- Input: 1x28x28
        model = nn.Sequential()
        model:add(nn.Reshape(1,28,28))
        ------> 1, 28,28
        model:add(cudnn.SpatialConvolution(1, 64, 3,3, 1,1, 1,1)
                    :init('weight', nninit.kaiming, {gain = 'relu'})
                    :init('bias', nninit.constant, 0))
        model:add(cudnn.SpatialBatchNormalization(64))
        model:add(cudnn.ReLU(true))
        ------> 64, 28,28   First Group
        for i=1,N do   addResidualDrop(model, nil, 64)   end
        ------> 128,64,64   Second Group
        addResidualDrop(model, nil, 64, 128, 2)
        for i=1,N do   addResidualDrop(model, nil, 128)   end
        ------> 64, 8,8     Third Group
        addResidualDrop(model, nil, 128, 256, 2)
        for i=1,N do   addResidualDrop(model, nil, 256)   end
        
        --cls = nn.Sequential()
        --local wid = 4
        model:add(nn.SpatialAveragePooling(6,6,4,4)):add(nn.Reshape(256))
        --cls:add(nn.Reshape(512*wid*wid))
        --cls:add(nn.Linear(512*wid*wid,10))
        --cls:add(cudnn.LogSoftMax())
        --model:add(cls)
        model:add(nn.Linear(256,10))
        model:add(cudnn.LogSoftMax())
        ]]
        
        end
        
        collectgarbage()
        --print(model)   -- if you need to see the architecture, it's going to be long!

    ---- Determines the position of all the residual blocks ----
    addtables = {}
    for i=1,model:size() do
        if tostring(model:get(i)) == 'nn.ResidualDrop' then addtables[#addtables+1] = i end
    end

    ---- Sets the deathRate (1 - survival probability) for all residual blocks  ----
    for i,block in ipairs(addtables) do
      if opt.deathMode == 'uniform' then
        model:get(block).deathRate = opt.deathRate
      elseif opt.deathMode == 'lin_decay' then
        model:get(block).deathRate = i / #addtables * opt.deathRate
      elseif opt.deathMode == 'spearmint' then
        --local HPOptim = require('/HPOptim/HPOptim.lua')
      else
        print('Invalid argument for deathMode!')
      end
    end
    
    all_results = {}
    sgdState.epochCounter  = 1
    save()
    
    if opt.dataset == 'svhn' then 
      sgdState.iterCounter = 1 
      print('Training...\nIter\t\tValid. err\tTest err\tTraining time')
    else
      print('Training...\nEpoch\tValid. err\tTest err\tTraining time')
    end
    
end

---- Resets all gates to open ----
function openAllGates()
  for i,block in ipairs(addtables) do model:get(block).gate = true end
end

---- Testing ----
function evalModel(dataset)
  model:evaluate()
  openAllGates() -- this is actually redundant, test mode never skips any layer
  local correct = 0
  local total = 0
  local batches = torch.range(1, dataset:size()):long():split(opt.batchSize)
  for i=1,#batches do
     local batch = dataset:sampleIndices(batches[i])
     local inputs, labels = batch.inputs, batch.outputs:long()
     local y = model:forward(inputs:cuda()):float()
     local _, indices = torch.sort(y, 2, true)
     -- indices is a tensor with shape (batchSize, nClasses)
     local top1 = indices:select(2, 1)
     correct = correct + torch.eq(top1, labels):sum()
     total = total + indices:size(1)
  end
  return 1-correct/total
end

-- For CIFAR, accounting is done every epoch, and for SVHN, every 200 iterations
function accounting(training_time, depth, batches)
  
  local results = {evalModel(dataValid), evalModel(dataTest)}
  if all_results == nil then print('all_results:nil') 
  else all_results[#all_results + 1] = results end
  -- Saves the errors. These get covered up by new ones every time the function is called
  torch.save(opt.resultFolder .. string.format('errors_%d_%s_%s_%.1f', 
    opt.N, opt.dataset, opt.deathMode, opt.deathRate), all_results)
  if opt.dataset == 'svhn' then 
    print(string.format('Iter %d:\t%.2f%%\t\t%.2f%%\t\t%0.0fs', 
      sgdState.iterCounter, results[1]*100, results[2]*100, training_time))
  else
    print(string.format('Epoch %d:\t%.2f%%\t\t%.2f%%\t\t%0.0fs', 
      sgdState.epochCounter, results[1]*100, results[2]*100, training_time))
    final:write(string.format('Epoch %d:\t%.2f%%\t\t%.2f%%\t\t%0.0fs\t\t%d\t\t%d\n', 
       sgdState.epochCounter, results[1]*100, results[2]*100, training_time, 
       depth,batches))
  end
end

function getError()
    return tonumber(evalModel(dataValid))
end


---- Training ----
function train(depth, counter, timer)  
  depth_suggestion = depth
  sgdState.epochCounter = counter
  
  local weights, gradients = model:getParameters()   

  local all_indices = torch.range(1, dataTrain:size())
  --local timer = torch.Timer()
  --while sgdState.epochCounter <= opt.maxEpochs do
    -- Learning rate schedule
    if sgdState.epochCounter < opt.maxEpochs*lrSchedule[opt.dataset][1] then
      sgdState.learningRate = 0.1
    elseif sgdState.epochCounter < opt.maxEpochs*lrSchedule[opt.dataset][2] then
      sgdState.learningRate = 0.01
    else
      sgdState.learningRate = 0.001
    end
    
    local shuffle = torch.randperm(dataTrain:size())
    local batches = all_indices:index(1, shuffle:long()):long():split(opt.batchSize)
    --print(#batches)
    for i=1,#batches do
        model:training()
        --print(i)
        openAllGates()    -- resets all gates to open
        -- Randomly determines the gates to close, according to their survival probabilities
        if opt.deathMode == 'uniform' or opt.deathMode == 'lin_decay' then
            for i,tb in ipairs(addtables) do
              if torch.rand(1)[1] < model:get(tb).deathRate then model:get(tb).gate = false end
            end
        elseif opt.deathMode == 'spearmint' then
            for i,tb in ipairs(addtables) do
              if i > depth_suggestion then model:get(tb).gate = false end
            end
        end
        function feval(x)
            gradients:zero()
            local batch = dataTrain:sampleIndices(batches[i])
            local inputs, labels = batch.inputs, batch.outputs:long()
            inputs = inputs:cuda()
            labels = labels:cuda()
            local y = model:forward(inputs)
            local loss_val = loss:forward(y, labels)
            local dl_df = loss:backward(y, labels)
            model:backward(inputs, dl_df)
            return loss_val, gradients
        end
        optim.sgd(feval, weights, sgdState)
        if opt.dataset == 'svhn' then
          if sgdState.iterCounter % 200 == 0 then
            accounting(timer:time().real, depth, batches)
            timer:reset()
          end
          sgdState.iterCounter = sgdState.iterCounter + 1
        end
    end
    if opt.dataset ~= 'svhn' then
      accounting(timer:time().real, depth, opt.batchSize)
      timer:reset()
    end
    if opt.deathMode == 'spearmint' then
    -- record error then reuse it in model.lua -> trainHyper()
      ferr = io.open('TMPresult.txt', "w+")
      ferr:write(getError())
      ferr:close()
    end    
    sgdState.epochCounter = sgdState.epochCounter + 1
    -- record update cpt
  --end
  print('save...')
  save()
end 

-- main
function main()

  local i = 0
  local f = io.open('./conf/cpt','w')
  f:write(i)
  f:close()
  
  while i <= 500 do
    trainHyper()
  end
end

--main()
