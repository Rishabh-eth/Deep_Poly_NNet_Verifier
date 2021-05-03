import torch
import torch.nn as nn

from transformer import *


class DeepPoly(nn.Module):
    """
    Transform the original model for deep poly
    """
    def __init__(self, model:nn.Module, x:torch.Tensor, eps:float, true_label:int):
        super().__init__()

        self.net = model
        inpt = InputTransform(input_layer=None, depth=0, input_shape=x.shape)
        self.x = inpt(None, x, eps)
        # print(self.x.ub.flatten())
        self.true_label = true_label
        self.build_model()
        self.forward()
        self.optimizer = torch.optim.Adam(self.parameters(), lr = 0.1)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.9,
            patience=10, threshold=0.01, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=False)
        
    
    def build_model(self):
        prev_transform = self.x
        depth = 1
        is_first_conv = True
        modified_layers = []
        for layer in self.net.layers:
            layer_name = layer.__class__.__name__
            if layer_name == 'Conv2d' and is_first_conv:
                transformed = modify_layer(nn.Flatten(), depth, prev_transform.output_shape)
                modified_layers.append(transformed)
                prev_transform = transformed(prev_transform)
                depth += 1
                is_first_conv = False
            transformed = modify_layer(layer, depth, prev_transform.output_shape)
            modified_layers.append(transformed)
            prev_transform = transformed(prev_transform)
            # print(layer_name, prev_transform.ub.flatten())
            depth += 1

        self.model = nn.Sequential(*modified_layers)
            

    def forward(self):
        
        output = self.model(self.x)
        
        return output

    def update_parameters(self):
        # self.optimizer = torch.optim.Adam(self.parameters(), lr = 0.1)
        loss = (-self.all_diffs[self.all_diffs < 0.]).square().sum()
        # loss = torch.exp(torch.clamp(-self.all_diffs, min=0.).sum())
        # print(loss)
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()
        self.scheduler.step(loss)
    
    def print_grads(self):
        for layer in self.model:
            if isinstance(layer, ReLUTransform) and hasattr(layer, "opt_params"):
                params = layer.opt_params.data
                print("params = ", params)
                print("grads = ", layer.opt_params.grad)

    def parameters(self):
        # A generator to allow gradient descent on `slope` of ReLUs.
        for layer in self.model:
            if isinstance(layer, ReLUTransform) and hasattr(layer, "opt_params"):
                yield layer.opt_params

    def get_diff(self, predictions, true_label):
        # weights and bias for true label
        true_lower_w = predictions.lower_weights[:, true_label, None]
        true_lower_b = predictions.lower_bias[true_label]

        # weights and bias for diff (x11 - x12)
        diff_w = true_lower_w - torch.cat((predictions.upper_weights[:,:true_label], predictions.upper_weights[:, (true_label+1):]), dim=1)
        diff_b = true_lower_b - torch.cat((predictions.upper_bias[:true_label], predictions.upper_bias[(true_label+1):]), dim=0)

        lpos_w = (diff_w >= 0.)
        lneg_w = (diff_w < 0.)

        # lower bound
        lb = predictions.init_lb @ (lpos_w * diff_w) +\
                  predictions.init_ub @ (lneg_w * diff_w) + diff_b
        
        return lb

    def set_backsub_start(self, start_depth):
        BaseTransform.backsub_start_depth = start_depth


    def verify(self, with_backsub = True):
        self.pred_poly = self.forward()

        if not with_backsub:
            true_lb = self.pred_poly.lb[self.true_label]
            true_ub = self.pred_poly.ub[self.true_label]
            verified = torch.all(self.pred_poly.ub[torch.arange(len(self.pred_poly.ub)) != self.true_label] < true_lb)
            self.all_diffs = true_lb - self.pred_poly.ub[torch.arange(len(self.pred_poly.ub)) != self.true_label]
        else:
            verified = False
            all_diffs = self.get_diff(self.pred_poly, self.true_label)
            # verified = torch.all(all_diffs[torch.arange(len(all_diffs)) != self.true_label] > 0.)
            verified = torch.all(all_diffs > 0.)
            # while not verified:
            # self.all_diffs = all_diffs[torch.arange(len(all_diffs)) != self.true_label]
            self.all_diffs = all_diffs

        return verified



## test for deeppoly
def test_deeppoly():
    inp = torch.Tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    eps = 0.05
    print(inp.shape)
    # input transform
    inpt = InputTransform(input_layer=None, depth=0, input_shape=inp.shape)
    x = inpt(None, inp, eps)
    print(f"Input: ub = {x.ub.flatten()} and lb = {x.lb.flatten()}")
    normalization_layer = Normalization(device="cpu")
    normalization_layer.mean = 1.0
    normalization_layer.sigma = 1.0
    norm = NormalizeTransform(input_layer=normalization_layer, depth = 1, input_shape = x.output_shape)
    x = norm(x)
    print(f"norma: ub = {x.ub.flatten()} and lb = {x.lb.flatten()}, {x.ub.flatten().float().eq(torch.tensor([ 0.05, -0.9500, -0.9500, 0.05]))}")
    # assert torch.all(x.ub.flatten().eq(torch.tensor([ 0.0500, -0.9500, -0.9500, 0.0500]))), f"{x.ub.flatten()}"
    assert torch.all(x.ub >= x.lb)
    flatten_layer = nn.Flatten()
    flat = FlattenTransform(input_layer=flatten_layer, depth=3, input_shape=x.output_shape)
    x = flat(x)
    print(f"flatt: ub = {x.ub} and lb={x.lb}")
    # assert torch.all(x.ub.eq(torch.Tensor([0.05,-0.95,-0.95,0.05])))
    assert torch.all(x.ub >= x.lb)
    linear_layer = nn.Linear(4, 3)
    linear_layer.weight.data = torch.Tensor([[1.,0.,-1,1],[0.,1.,1.,0.],[-1,1,1,0]])
    linear_layer.bias.data = torch.Tensor([0.1,-0.1,0.2])
    lin = AffineTransform(input_layer = linear_layer, depth = 3, input_shape = x.output_shape)
    x = lin(x)
    print(f"linea: ub = {x.ub} and lb = {x.lb}")
    # assert torch.all(x.ub == torch.Tensor([1.25,-2.0,-1.65]))
    assert torch.all(x.ub >= x.lb)
    relu_layer = nn.ReLU()
    relu = ReLUTransform(input_layer = relu_layer, depth=4, input_shape = x.output_shape)
    x = relu(x)
    print(f"relu: ub = {x.ub} and lb = {x.lb}")
    assert torch.all(x.ub >= x.lb)
    linear_layer = nn.Linear(3,2)
    linear_layer.weight.data = torch.tensor([[-1.,0,1],[0,1,1]])
    linear_layer.bias.data = torch.Tensor([1,-0.2])
    lint = AffineTransform(linear_layer, 5, x.output_shape)
    x = lint(x)
    print(f"linea: ub = {x.ub} and lb ={x.lb}")
    assert torch.all(x.ub >= x.lb)
    relu_layer = nn.ReLU()
    relu = ReLUTransform(input_layer = relu_layer, depth=4, input_shape = x.output_shape)
    x = relu(x)
    print(f"relu: ub = {x.ub} and lb = {x.lb}")


if __name__ == "__main__":
    from networks import *
    test_deeppoly()