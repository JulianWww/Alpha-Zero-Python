from tensorflow import where, fill, zeros_like, shape, equal, nn

def softmax_cross_entropy_with_logits(y_true, y_pred):
  zeros = zeros_like(y_true)
  locs = equal(y_true, zeros)
  negs = fill(shape(y_pred), -100.0)
  pred = where(locs, negs, y_pred)

  return nn.softmax_cross_entropy_with_logits(y_true, pred)

    