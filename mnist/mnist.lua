-- This script downloads and loads the MNIST dataset
-- http://yann.lecun.com/exdb/mnist/
----------------------------------------------------------------------
print '==> downloading dataset'

-- Here we download dataset files. 

-- Note: files were converted from their original LUSH format
-- to Torch's internal format.

-- The SVHN dataset contains 3 files:
--    + train: training data
--    + test:  test data

tar = 'http://torch7.s3-website-us-east-1.amazonaws.com/data/mnist.t7.tgz'

if not paths.dirp('mnist.t7') then
   os.execute('wget ' .. tar)
   os.execute('tar xvf ' .. paths.basename(tar))
end

train_file = 'mnist.t7/train_32x32.t7'
test_file = 'mnist.t7/test_32x32.t7'

----------------------------------------------------------------------
print '==> loading dataset'

-- We load the dataset from disk, it's straightforward

trainData = torch.load(train_file, 'ascii')
testData = torch.load(test_file, 'ascii')

require 'cudnn'
require 'nn'

net = nn.Sequential()
net:add(nn.SpatialConvolution(1, 32, 5, 5)) -- 1 input image channels, 32 output channels, 5x5 convolution kernel
net:add(nn.ReLU())                       -- non-linearity 
net:add(nn.SpatialMaxPooling(2,2,2,2))     -- A max-pooling operation that looks at 2x2 windows and finds the max.
net:add(nn.SpatialConvolution(32, 64, 5, 5))
net:add(nn.ReLU())                       -- non-linearity 
net:add(nn.SpatialMaxPooling(2,2,2,2))
net:add(nn.View(64 * 5 * 5))                    -- reshapes from a 3D tensor of 16x5x5 into 1D tensor of 16*5*5
net:add(nn.Linear(64 * 5 * 5, 1024))             -- fully connected layer (matrix multiplication between input and weights)
net:add(nn.ReLU())                       -- non-linearity 
net:add(nn.Linear(1024, 10))                   -- 10 is the number of outputs of the network (in this case, 10 digits)
net:add(nn.LogSoftMax())                     -- converts the output to a log-probability. Useful for classification problems

criterion = nn.ClassNLLCriterion()

trainer = nn.StochasticGradient(net, criterion)
trainer.learningRate = 0.001
trainer.maxIteration = 5 -- just do 5 epochs of training.

setmetatable(trainData, 
    {__index = function(t, i) 
                    return {t.data[i], t.labels[i]} 
                end}
);
trainData.data = trainData.data:double() -- convert the data from a ByteTensor to a DoubleTensor.

function trainData:size() 
    return self.data:size(1) 
end

cudnn.convert(net, cudnn)
trainer:train(trainData)
