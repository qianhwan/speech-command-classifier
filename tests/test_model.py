import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchtest import assert_vars_change, assert_never_nan, assert_never_inf

from speech_command_classifier.model import Model


def test_model_build():
    model = Model(n_input=3, n_output=5, n_channel=16)
    print(model)


def test_model_train():
    model = Model(n_input=1, n_output=30, n_channel=32)
    inputs = Variable(torch.randn(4, 32, 32))
    targets = Variable(torch.randint(0, 30, (4,)))
    assert_vars_change(
            model=model,
            loss_fn=F.nll_loss,
            optim=torch.optim.Adam(model.parameters()),
            batch=[inputs, targets],
            device='cpu')


def test_model_params():
    model = Model(n_input=1, n_output=30, n_channel=32)
    inputs = Variable(torch.randn(4, 32, 32))
    outputs = model(inputs)

    assert_never_nan(outputs)
    assert_never_inf(outputs)
