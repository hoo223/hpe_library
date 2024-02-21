#### MLP with skip connection
# https://github.com/motokimura/3d-pose-baseline-pytorch/blob/master/human_3d_pose_baseline/models/baseline_model.py

# references:
# https://github.com/weigq/3d_pose_baseline_pytorch/blob/master/src/model.py
# https://github.com/una-dinosauria/3d-pose-baseline/blob/master/src/linear_model.py

from lib_import import *

def init_weights(module):
    """Initialize weights of the baseline linear model.

    Our initialization scheme is different from the official implementation in TensorFlow.
    Official one inits bias of linear layer with kaiming normal but we init with 0.
    Also we init weights of batchnorm layer with 1 and bias with 0.
    We have not investigated if this affects the accuracy.

    Args:
        module (torch.nn.Module): torch.nn.Module composing the baseline linear model.
    """
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight.data, mode="fan_in", nonlinearity="relu")
        module.bias.data.zero_()
    if isinstance(module, nn.BatchNorm1d):
        module.weight.data.fill_(1)
        module.bias.data.zero_()

class Linear(nn.Module):
    def __init__(self, linear_size, p_dropout):
        """

        Args:
            linear_size (int): Number of nodes in the linear layers.
            p_dropout (float): Dropout probability.
        """
        super(Linear, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p_dropout)

        self.w1 = nn.Linear(linear_size, linear_size)
        self.bn1 = nn.BatchNorm1d(linear_size)

        self.w2 = nn.Linear(linear_size, linear_size)
        self.bn2 = nn.BatchNorm1d(linear_size)

    def forward(self, x):
        """Forward operations of the linear block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            y (torch.Tensor): Output tensor.
        """
        h = self.w1(x)
        h = self.bn1(h)
        h = self.relu(h)
        h = self.dropout(h)

        h = self.w2(h)
        h = self.bn2(h)
        h = self.relu(h)
        h = self.dropout(h)

        y = x + h
        return y


class BaselineModel(nn.Module):
    def __init__(self, linear_size=1024, num_stages=2, input_size=16*2, output_size=16*3, p_dropout=0.5, skip_connection=False):
        """

        Args:
            linear_size (int, optional): Number of nodes in the linear layers. Defaults to 1024.
            num_stages (int, optional): Number to repeat the linear block. Defaults to 2.
            p_dropout (float, optional): Dropout probability. Defaults to 0.5.
            predict_14 (bool, optional): Whether to predict 14 3d-joints. Defaults to False.
        """
        super(BaselineModel, self).__init__()

        input_size = input_size  # Input 2d-joints.
        output_size = output_size # Output 3d-joints.
        self.skip_connection = skip_connection

        self.w1 = nn.Linear(input_size, linear_size)
        self.bn1 = nn.BatchNorm1d(linear_size)

        self.linear_stages = [Linear(linear_size, p_dropout) for _ in range(num_stages)]
        self.linear_stages = nn.ModuleList(self.linear_stages)

        self.w2 = nn.Linear(linear_size, output_size)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p_dropout)

        # initialize model weights
        self.apply(init_weights)

    def forward(self, x):
        """Forward operations of the linear block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            y (torch.Tensor): Output tensor.
        """
        y = self.w1(x)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.dropout(y)

        # linear blocks
        for linear in self.linear_stages:
            y = linear(y)

        y = self.w2(y)
        
        
        if self.skip_connection:
            y += x[:, :15]
        
        return y

        #point = self.w2(y) + x[:, :3]
        #direction = self.w2(y) + x[:, 3:6]

        #return point, direction


class TorsoModel(nn.Module):
    def __init__(self, 
                 linear_size=1024, 
                 num_stages=2, 
                 input_size=20, 
                 output_size=7, 
                 input_list=[], 
                 output_list=[], 
                 input_idxs=[], 
                 output_idxs=[],
                 p_dropout=0.5, skip_connection=[None, None, None]):
        """

        Args:
            linear_size (int, optional): Number of nodes in the linear layers. Defaults to 1024.
            num_stages (int, optional): Number to repeat the linear block. Defaults to 2.
            p_dropout (float, optional): Dropout probability. Defaults to 0.5.
            predict_14 (bool, optional): Whether to predict 14 3d-joints. Defaults to False.
        """
        super(TorsoModel, self).__init__()

        self.input_size = input_size  # Input 2d-joints.
        self.output_size = output_size # Output 3d-joints.
        if skip_connection is None:
            self.skip_connection = [None for _ in range(len(output_list))]
        else:
            self.skip_connection = skip_connection
        self.input_idxs = input_idxs
        self.output_idxs = output_idxs

        # main layers
        #print(input_size, linear_size)
        self.w1 = nn.Linear(input_size, linear_size)
        self.bn1 = nn.BatchNorm1d(linear_size)
        self.linear_stages = [Linear(linear_size, p_dropout) for _ in range(num_stages)]
        self.linear_stages = nn.ModuleList(self.linear_stages)

        # output layers
        self.output_layers = nn.ModuleList()
        for i, _ in enumerate(output_list):
            temp_layer = nn.ModuleList()
            temp_layer.append(Linear(linear_size, p_dropout))
            temp_layer.append(nn.Linear(linear_size, self.output_idxs[i][1] - self.output_idxs[i][0]))
            self. output_layers.append(temp_layer)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p_dropout)

        # initialize model weights
        self.apply(init_weights)

    def forward(self, x):
        """Forward operations of the linear block.
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            y (torch.Tensor): Output tensor.
        """
        y = self.w1(x)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.dropout(y)

        # linear blocks
        for linear in self.linear_stages:
            y = linear(y)

        # output layers
        outputs = []
        for i, output_layer in enumerate(self.output_layers):
            o = output_layer[1](output_layer[0](y))
            if self.skip_connection[i] is not None:
                o += x[:, self.input_idxs[self.skip_connection[i]][0]:self.input_idxs[self.skip_connection[i]][1]]
            outputs.append(o)
        
        return outputs
    
    
def split_array_by_idxs(array, idxs):
    array_items = []
    for i, idx in enumerate(idxs):
        array_items.append(array[:, idx[0]:idx[1]])
    return array_items