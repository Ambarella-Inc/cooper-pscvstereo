import torch
import torch.nn as nn
import torch.nn.functional as F


class FlowHead(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=256, output_dim=2, act=nn.ReLU):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, output_dim, 3, padding=1)
        self.relu = act(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class ConvMiniGRU(nn.Module):
    """Mini-GRU with separate input/hidden convs and external cz/cq (e.g. disp) biases.

    Forward: ``(h, cz, cq, inpz, inpq, x)`` where ``inpz``/``inpq`` are split context
    and ``x`` is the motion feature.
    """

    def __init__(self, hidden_dim, input_dim, kernel_size=3):
        super(ConvMiniGRU, self).__init__()
        pad = kernel_size // 2
        self.input_convz = nn.Conv2d(input_dim, hidden_dim, kernel_size, padding=pad, bias=False)
        self.hidden_convz = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=pad, bias=True)
        self.input_convq = nn.Conv2d(input_dim, hidden_dim, kernel_size, padding=pad, bias=False)
        self.hidden_convq = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=pad, bias=True)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, h, cz, cq, inpz, inpq, x):
        z = self.sigmoid(self.input_convz(
            torch.cat([x, inpz], 1)) + self.hidden_convz(h) + cz)
        q = self.tanh(self.input_convq(
            torch.cat([x, inpq], 1)) + self.hidden_convq(h) + cq)
        return z * (q - h) + h
