local mnist_dataset = require 'mnist' -- https://github.com/andresy/mnist

mnist = {}

function mnist.load_data(flatten)
  flatten = flatten or true
  local train_set = mnist_dataset.traindataset()
  local test_set = mnist_dataset.testdataset()

  local trX = train_set.data[{{1,50000}}]:double()
  local trY = train_set.label[{{1,50000}}]:add(1)
  local teX = test_set.data:double()
  local teY = test_set.label:add(1)

  -- Convert images from matrix of size 28x28 to a vector of size 784
  if flatten then
    trX = trX:reshape(trX:size(1), trX:nElement() / trX:size(1))
    teX = teX:reshape(teX:size(1), teX:nElement() / teX:size(1))
  end

  -- Normalize the inputs in the range [0,1] (actually [0, 255/256]) for
  -- compatibility with http://deeplearning.net/data/mnist/mnist.pkl.gz
  trX = trX / 256
  teX = teX / 256

  return trX, trY, teX, teY
end
