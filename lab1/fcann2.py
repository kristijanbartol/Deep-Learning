import data


def fcann2_train():
    pass


def fcann2_classify():
    pass

# eksponencirane klasifikacijske mjere
# pri računanju softmaksa obratite pažnju
# na odjeljak 4.1 udžbenika
# (Deep Learning, Goodfellow et al)!
scores = ...  # N x C
expscores = ...  # N x C

# nazivnik sofmaksa
sumexp = ...  # N x 1

# logaritmirane vjerojatnosti razreda
probs = ...  # N x C
logprobs = ...  # N x C

# gubitak
loss = ...  # scalar

# dijagnostički ispis
if i % 10 == 0:
    print("iteration {}: loss {}".format(i, loss))

# derivacije komponenata gubitka po mjerama
dL_ds = ...  # N x C

# gradijenti parametara
grad_W = ...  # C x D (ili D x C)
grad_b = ...  # C x 1 (ili 1 x C)

# poboljšani parametri
W += -param_delta * grad_W
b += -param_delta * grad_b