HPOptim = require '/HPOptim/HPOptim'

require 'cutorch'
require 'torch'

opt = lapp[[
      --maxEpochs     (default 500)         Maximum number of epochs to train the network
      --batchSize     (default 128)         Mini-batch size
      --N             (default 18)          Model has 6*N+2 convolutional layers
      --dataset       (default cifar10)     Use cifar10, cifar100 or svhn or mnist
      --deathMode     (default spearmint)   Use lin_decay or uniform or spearmint
      --deathRate     (default 0)           1-p_L for lin_decay, 1-p_l for uniform, 0 is constant depth
      --device        (default 0)           Which GPU to run on, 0-based indexing
      --augmentation  (default true)        Standard data augmentation (CIFAR only), true or false 
      --resultFolder  (default "")          Path to the folder where you'd like to save results
      --dataRoot      (default "./cifar.torch/")          Path to data (e.g. contains cifar10-train.t7)
    ]]

cutorch.setDevice(opt.device+1)   -- torch uses 1-based indexing for GPU, so +1
cutorch.manualSeed(1)
torch.manualSeed(1)
torch.setnumthreads(1)            -- number of OpenMP threads, 1 is enough

torch.save('./conf/opt.conf', opt)

f = io.open('./conf/cpt','w')
f:write(0)
f:close()

HPOptim.init()
HPOptim.findHP(4320000)
