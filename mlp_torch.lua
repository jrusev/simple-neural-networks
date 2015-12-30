require 'torch'
require 'nn'
require 'mnist_torch'

net = nn.Sequential()
        :add(nn.Linear(28*28, 100))
        :add(nn.Sigmoid())
        :add(nn.Linear(100, 10))

function train(X, Y)
  local criterion = nn.CrossEntropyCriterion()
  criterion:forward(net:forward(X), Y)
  net:zeroGradParameters()
  net:backward(X, criterion:backward(net.output, Y))
  net:updateParameters(learn_rate) -- weights:add(-learn_rate, grads)
end

function accuracy(teX, teY)
    local _, predicted = torch.max(net:forward(teX), 2)
    return predicted:eq(teY:long()):sum() / teX:size(1)
end

num_epochs = 30
batch_size = 10
learn_rate = 0.2

trX, trY, teX, teY = mnist.load_data()
for i = 1,num_epochs do
    for t = 1,trX:size(1),batch_size do
        train(trX:sub(t, t+batch_size-1), trY:sub(t, t+batch_size-1))
    end
    print(i .. ' ' .. accuracy(teX, teY))
end
