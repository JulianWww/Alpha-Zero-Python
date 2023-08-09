from keras.layers import Layer, Conv2D, BatchNormalization, ReLU, Add, Flatten, Dense, Input
from keras.optimizers import SGD
from keras.models import Model
from keras.callbacks import TensorBoard, ModelCheckpoint
from loss import softmax_cross_entropy_with_logits
import numpy as np
from datetime import datetime
from config import MODEL_PATH, N_BATCHES

logdir="logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=logdir)


class ResidualLayer(Layer):
  def __init__(self, fillters: int, kernel, **kwargs):
    super(ResidualLayer, self).__init__(**kwargs)
    self.fillters = fillters
    self.kernel = kernel

    self.conv1 = Conv2D(self.fillters, self.kernel, padding="same")
    self.nomr1 = BatchNormalization()
    self.relu1 = ReLU()
    self.conv2 = Conv2D(self.fillters, self.kernel, padding="same")
    self.norm2 = BatchNormalization()
    self.addly = Add()
    self.relu2 = ReLU()

  def call(self, input):
    inner = self.conv1(input)
    inner = self.nomr1(inner)
    inner = self.relu1(inner)
    inner = self.conv2(inner)
    inner = self.norm2(inner)
    out = self.addly([inner, input])
    out = self.relu2(out)
    return out

class ConvTail(Layer):
  def __init__(self, fillters, kernel, **kwargs):
    super(ConvTail, self).__init__(**kwargs)
    self.fillters = fillters
    self.kernel = kernel

    self.conv = Conv2D(self.fillters, self.kernel, padding="same")
    self.norm = BatchNormalization()
    self.relu = ReLU()
  
  def call(self, input):
    input = self.conv(input)
    input = self.norm(input)
    input = self.relu(input)
    return input

class ValueHead(Layer):
  def __init__(self, **kwargs):
    super(ValueHead, self).__init__(**kwargs)

    self.conv1 = Conv2D(1, (1,1))
    self.norm1 = BatchNormalization()
    self.relu1 = ReLU()
    self.flatt = Flatten()
    self.dens1 = Dense(256)
    self.relu2 = ReLU()
    self.dens2 = Dense(1, activation="tanh", name="value_head")
  
  def call(self, input):
    input = self.conv1(input)
    input = self.norm1(input)
    input = self.relu1(input)
    input = self.flatt(input)
    input = self.dens1(input)
    input = self.relu2(input)
    input = self.dens2(input)
    return input

class PolicyHead(Layer):
  def __init__(self, action_space, **kwargs):
    super(PolicyHead, self).__init__(**kwargs)
    self.action_space = action_space


    self.conv = Conv2D(2, (1,1))
    self.norm = BatchNormalization()
    self.relu = ReLU()
    self.flat = Flatten()
    self.dens = Dense(self.action_space, name="policy_head")
  
  def call(self, input):
    input = self.conv(input)
    input = self.norm(input)
    input = self.relu(input)
    input = self.flat(input)
    input = self.dens(input)
    return input

class AlphaZeroModel(Model):
  def __init__(self, action_space, input_shape, res_size, lr, momentum, perform_saves=False):
    inp = Input(input_shape, name="main_input")

    self.zeroCallbacks = self.buildCallback(perform_saves)

    x = ConvTail(256, (3,3), name="convTail")(inp)
    for i in range(res_size):
      x = ResidualLayer(256, (3,3), name=f"res_{i}")(x)
    
    poly = PolicyHead(action_space)(x)
    valu = ValueHead()(x)


    super(AlphaZeroModel, self).__init__(inputs=[inp], outputs=[poly, valu])
    self.compile(
      loss={
        "value_head": "mean_squared_error",
        "policy_head": softmax_cross_entropy_with_logits
      },
      optimizer=SGD(learning_rate=lr, momentum=momentum),
      loss_weights={
        "value_head": 0.5,
        "policy_head": 0.5
      }
    )

  def fit(self, *args, **kwargs):
    super().fit(self, *args, callbacks=self.zeroCallbacks, **kwargs)
  
  def buildCallback(self, perform_saves):
    cp_callback = ModelCheckpoint(
      filepath=MODEL_PATH, 
      verbose=1, 
      save_weights_only=True,
      save_freq=5*N_BATCHES)
    
    print(cp_callback)

inp = np.random.random((1,6,7,2))
policy = np.random.random((1,42))
value = np.random.random((1,1))

modle = AlphaZeroModel(42, (6,7,2), 1, 0.1, 0.9)

print(modle.predict(inp))
modle.fit(inp, [policy, value])