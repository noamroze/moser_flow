import torch.nn as nn
import torch

class LinearGrad(nn.Linear):
    def forward(self, input, input_grad, compute_grad=False, is_first = False, pc_dim=3):
        output = super().forward(input)
        if not compute_grad:
            return output, None
        output_grad = self.weight[:,-pc_dim:] if is_first else self.weight.matmul(input_grad)
        return output, output_grad

class SoftplusGrad(nn.Softplus):
    def forward(self, input, input_grad, compute_grad=False):
        output = super().forward(input)
        if not compute_grad:
            return output, None
        output_grad = torch.sigmoid(self.beta * input).unsqueeze(-1) * input_grad
        return output, output_grad
