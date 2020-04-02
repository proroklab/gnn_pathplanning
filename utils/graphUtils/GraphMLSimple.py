# 2018/11/01~2018/07/12
# Fernando Gama, fgama@seas.upenn.edu.
# GraphRNN editted by Qingbiao Li
"""
graphML.py Module for basic GSP and graph machine learning functions.

Functionals

LSIGF: Applies a linear shift-invariant graph filter
spectralGF: Applies a linear shift-invariant graph filter in spectral form
NVGF: Applies a node-variant graph filter
EVGF: Applies an edge-variant graph filter
learnAttentionGSO: Computes the GSO following the attention mechanism
graphAttention: Applies a graph attention layer

Filtering Layers (nn.Module)

GraphFilter: Creates a graph convolutional layer using LSI graph filters
SpectralGF: Creates a graph convolutional layer using LSI graph filters in
    spectral form
NodeVariantGF: Creates a graph filtering layer using node-variant graph filters
EdgeVariantGF: Creates a graph filtering layer using edge-variant graph filters
GraphAttentional: Creates a layer using graph attention mechanisms

Activation Functions - Nonlinearities (nn.Module)

MaxLocalActivation: Creates a localized max activation function layer
MedianLocalActivation: Creates a localized median activation function layer
NoActivation: Creates a layer for no activation function

Summarizing Functions - Pooling (nn.Module)

NoPool: No summarizing function.
MaxPoolLocal: Max-summarizing function
"""

import math
import numpy as np
import torch
import torch.nn as nn

import utils.graphUtils.graphTools as graphTools

zeroTolerance = 1e-9  # Values below this number are considered zero.
infiniteNumber = 1e12  # infinity equals this number


# WARNING: Only scalar bias.

def LSIGF(h, S, x, b=None):
    """
    LSIGF(filter_taps, GSO, input, bias=None) Computes the output of a linear
        shift-invariant graph filter on input and then adds bias.

    Denote as G the number of input features, F the number of output features,
    E the number of edge features, K the number of filter taps, N the number of
    nodes, S_{e} in R^{N x N} the GSO for edge feature e, x in R^{G x N} the
    input data where x_{g} in R^{N} is the graph signal representing feature
    g, and b in R^{F x N} the bias vector, with b_{f} in R^{N} representing the
    bias for feature f.

    Then, the LSI-GF is computed as
        y_{f} = \sum_{e=1}^{E}
                    \sum_{k=0}^{K-1}
                    \sum_{g=1}^{G}
                        [h_{f,g,e}]_{k} S_{e}^{k} x_{g}
                + b_{f}
    for f = 1, ..., F.

    Inputs:
        filter_taps (torch.tensor): array of filter taps; shape:
            output_features x edge_features x filter_taps x input_features
        GSO (torch.tensor): graph shift operator; shape:
            edge_features x number_nodes x number_nodes
        input (torch.tensor): input signal; shape:
            batch_size x input_features x number_nodes
        bias (torch.tensor): shape: output_features x number_nodes
            if the same bias is to be applied to all nodes, set number_nodes = 1
            so that b_{f} vector becomes b_{f} \mathbf{1}_{N}

    Outputs:
        output: filtered signals; shape:
            batch_size x output_features x number_nodes
    """
    # The basic idea of what follows is to start reshaping the input and the
    # GSO so the filter coefficients go just as a very plain and simple
    # linear operation, so that all the derivatives and stuff on them can be
    # easily computed.

    # h is output_features x edge_weights x filter_taps x input_features
    # S is edge_weighs x number_nodes x number_nodes
    # x is batch_size x input_features x number_nodes
    # b is output_features x number_nodes
    # Output:
    # y is batch_size x output_features x number_nodes

    # Get the parameter numbers:
    F = h.shape[0]
    E = h.shape[1]
    K = h.shape[2]
    G = h.shape[3]
    assert S.shape[0] == E
    N = S.shape[1]
    assert S.shape[2] == N
    B = x.shape[0]
    assert x.shape[1] == G
    assert x.shape[2] == N
    # Or, in the notation we've been using:
    # h in F x E x K x G
    # S in E x N x N
    # x in B x G x N
    # b in F x N
    # y in B x F x N

    # Now, we have x in B x G x N and S in E x N x N, and we want to come up
    # with matrix multiplication that yields z = x * S with shape
    # B x E x K x G x N.
    # For this, we first add the corresponding dimensions
    x = x.reshape([B, 1, G, N])
    S = S.reshape([1, E, N, N])
    z = x.reshape([B, 1, 1, G, N]).repeat(1, E, 1, 1, 1)  # This is for k = 0
    # We need to repeat along the E dimension, because for k=0, S_{e} = I for
    # all e, and therefore, the same signal values have to be used along all
    # edge feature dimensions.
    for k in range(1, K):
        x = torch.matmul(x, S)  # B x E x G x N
        xS = x.reshape([B, E, 1, G, N])  # B x E x 1 x G x N
        z = torch.cat((z, xS), dim=2)  # B x E x k x G x N
    # This output z is of size B x E x K x G x N
    # Now we have the x*S_{e}^{k} product, and we need to multiply with the
    # filter taps.
    # We multiply z on the left, and h on the right, the output is to be
    # B x N x F (the multiplication is not along the N dimension), so we reshape
    # z to be B x N x E x K x G and reshape it to B x N x EKG (remember we
    # always reshape the last dimensions), and then make h be E x K x G x F and
    # reshape it to EKG x F, and then multiply
    y = torch.matmul(z.permute(0, 4, 1, 2, 3).reshape([B, N, E * K * G]),
                     h.reshape([F, E * K * G]).permute(1, 0)).permute(0, 2, 1)
    # And permute againt to bring it from B x N x F to B x F x N.
    # Finally, add the bias
    if b is not None:
        y = y + b
    return y



class GraphFilter(nn.Module):
    """
    GraphFilter Creates a (linear) layer that applies a graph filter

    Initialization:

        GraphFilter(in_features, out_features, filter_taps,
                    edge_features=1, bias=True)

        Inputs:
            in_features (int): number of input features (each feature is a graph
                signal)
            out_features (int): number of output features (each feature is a
                graph signal)
            filter_taps (int): number of filter taps
            edge_features (int): number of features over each edge
            bias (bool): add bias vector (one bias per feature) after graph
                filtering

        Output:
            torch.nn.Module for a graph filtering layer (also known as graph
            convolutional layer).

        Observation: Filter taps have shape
            out_features x edge_features x filter_taps x in_features

    Add graph shift operator:

        GraphFilter.addGSO(GSO) Before applying the filter, we need to define
        the GSO that we are going to use. This allows to change the GSO while
        using the same filtering coefficients (as long as the number of edge
        features is the same; but the number of nodes can change).

        Inputs:
            GSO (torch.tensor): graph shift operator; shape:
                edge_features x number_nodes x number_nodes

    Forward call:

        y = GraphFilter(x)

        Inputs:
            x (torch.tensor): input data; shape:
                batch_size x in_features x number_nodes

        Outputs:
            y (torch.tensor): output; shape:
                batch_size x out_features x number_nodes
    """

    def __init__(self, G, F, K, E=1, bias=True):
        # K: Number of filter taps
        # GSOs will be added later.
        # This combines both weight scalars and weight vectors.
        # Bias will always be shared and scalar.

        # Initialize parent
        super().__init__()
        # Save parameters:
        self.G = G
        self.F = F
        self.K = K
        self.E = E
        self.S = None  # No GSO assigned yet
        # Create parameters:
        self.weight = nn.parameter.Parameter(torch.Tensor(F, E, K, G))
        if bias:
            self.bias = nn.parameter.Parameter(torch.Tensor(F, 1))
        else:
            self.register_parameter('bias', None)
        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        # Taken from _ConvNd initialization of parameters:
        stdv = 1. / math.sqrt(self.G * self.K)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def addGSO(self, S):
        # Every S has 3 dimensions.
        assert len(S.shape) == 3
        # S is of shape E x N x N
        assert S.shape[0] == self.E
        self.N = S.shape[1]
        assert S.shape[2] == self.N
        self.S = S

    def forward(self, x):
        # x is of shape: batchSize x dimInFeatures x numberNodesIn
        B = x.shape[0]
        F = x.shape[1]
        Nin = x.shape[2]
        # And now we add the zero padding
        if Nin < self.N:
            x = torch.cat((x,
                           torch.zeros(B, F, self.N - Nin) \
                           .type(x.dtype).to(x.device)
                           ), dim=2)
        # Compute the filter output
        u = LSIGF(self.weight, self.S, x, self.bias)
        # So far, u is of shape batchSize x dimOutFeatures x numberNodes
        # And we want to return a tensor of shape
        # batchSize x dimOutFeatures x numberNodesIn
        # since the nodes between numberNodesIn and numberNodes are not required
        if Nin < self.N:
            u = torch.index_select(u, 2, torch.arange(Nin).to(u.device))
        return u

    def extra_repr(self):
        reprString = "in_features=%d, out_features=%d, " % (
            self.G, self.F) + "filter_taps=%d, " % (
                         self.K) + "edge_features=%d, " % (self.E) + \
                     "bias=%s, " % (self.bias is not None)
        if self.S is not None:
            reprString += "GSO stored"
        else:
            reprString += "no GSO stored"
        return reprString


class GraphFilterRNN(nn.Module):
    """
    GraphFilterRNN Creates a (linear) layer that applies a graph filter
        with Hidden Markov Model

    Initialization:

        GraphFilterRNN(in_features, out_features, hidden_features, filter_taps,
                    edge_features=1, bias=True)

        Inputs:
            in_features (int): number of input features (each feature is a graph
                signal)
            out_features (int): number of output features (each feature is a
                graph signal)
            hidden_features (int): number of hidden features (each feature is a
                graph signal)
            filter_taps (int): number of filter taps
            edge_features (int): number of features over each edge
            bias (bool): add bias vector (one bias per feature) after graph
                filtering

        Output:
            torch.nn.Module for a graph filtering layer (also known as graph
            convolutional layer).

        Observation: Filter taps have shape
            out_features x edge_features x filter_taps x in_features

    Add graph shift operator:

        GraphFilter.addGSO(GSO) Before applying the filter, we need to define
        the GSO that we are going to use. This allows to change the GSO while
        using the same filtering coefficients (as long as the number of edge
        features is the same; but the number of nodes can change).

        Inputs:
            GSO (torch.tensor): graph shift operator; shape:
                edge_features x number_nodes x number_nodes

    Forward call:

        y = GraphFilter(x)

        Inputs:
            x (torch.tensor): input data; shape:
                batch_size x in_features x number_nodes

        Outputs:
            y (torch.tensor): output; shape:
                batch_size x out_features x number_nodes
    """

    def __init__(self, G, H, F, K, E=1, bias=True):
        # K: Number of filter taps
        # GSOs will be added later.
        # This combines both weight scalars and weight vectors.
        # Bias will always be shared and scalar.

        # Initialize parent
        super().__init__()
        # Save parameters:
        self.G = G  # in_features
        self.F = F  # out_features
        self.H = H  # hidden_features
        self.K = K  # filter_taps
        self.E = E  # edge_features
        self.S = None  # No GSO assigned yet
        # Create parameters:
        self.weight_A = nn.parameter.Parameter(torch.Tensor(H, E, K, G))
        self.weight_B = nn.parameter.Parameter(torch.Tensor(H, E, K, H))
        self.weight_U = nn.parameter.Parameter(torch.Tensor(F, E, K, H))
        if bias:
            self.bias_A = nn.parameter.Parameter(torch.Tensor(H, 1))
            self.bias_B = nn.parameter.Parameter(torch.Tensor(H, 1))
            self.bias_U = nn.parameter.Parameter(torch.Tensor(F, 1))
        else:
            self.register_parameter('bias', None)
        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        # Taken from _ConvNd initialization of parameters:
        stdv_a = 1. / math.sqrt(self.G * self.K)
        self.weight_A.data.uniform_(-stdv_a, stdv_a)
        if self.bias_A is not None:
            self.bias_A.data.uniform_(-stdv_a, stdv_a)

        stdv_b = 1. / math.sqrt(self.H * self.K)
        self.weight_B.data.uniform_(-stdv_b, stdv_b)
        if self.bias_B is not None:
            self.bias_B.data.uniform_(-stdv_b, stdv_b)

        stdv_u = 1. / math.sqrt(self.H * self.K)
        self.weight_U.data.uniform_(-stdv_u, stdv_u)
        if self.bias_U is not None:
            self.bias_U.data.uniform_(-stdv_u, stdv_u)

    def addGSO(self, S):
        # Every S has 3 dimensions.
        assert len(S.shape) == 3
        # S is of shape E x N x N
        assert S.shape[0] == self.E
        self.N = S.shape[1]
        assert S.shape[2] == self.N
        self.S = S

    def forward(self, x, h):
        # x is of shape: batchSize x dimInFeatures x numberNodesIn
        B = x.shape[0]
        F = x.shape[1]
        Nin = x.shape[2]
        # And now we add the zero padding
        if Nin < self.N:
            x = torch.cat((x,
                           torch.zeros(B, F, self.N - Nin) \
                           .type(x.dtype).to(x.device)
                           ), dim=2)
        # Compute the filter output
        u_a = LSIGF(self.weight_A, self.S, x, self.bias_A)

        u_b = LSIGF(self.weight_B, self.S, h, self.bias_B)

        h = u_a + u_b

        u = LSIGF(self.weight_U, self.S, h, self.bias_U)
        # So far, u is of shape batchSize x dimOutFeatures x numberNodes
        # And we want to return a tensor of shape
        # batchSize x dimOutFeatures x numberNodesIn
        # since the nodes between numberNodesIn and numberNodes are not required
        if Nin < self.N:
            u = torch.index_select(u, 2, torch.arange(Nin).to(u.device))
        return u

    def extra_repr(self):
        reprString = "in_features=%d, out_features=%d, hidden_features=%d" % (
            self.G, self.F, self.H) + "filter_taps=%d, " % (
                         self.K) + "edge_features=%d, " % (self.E) + \
                     "bias=%s, " % (self.bias is not None)
        if self.S is not None:
            reprString += "GSO stored"
        else:
            reprString += "no GSO stored"
        return reprString


def BatchLSIGF(h, S, x, b=None):
    """
    LSIGF(filter_taps, GSO, input, bias=None) Computes the output of a linear
        shift-invariant graph filter on input and then adds bias.

    Denote as F the number of input features, G the number of output features,
    E the number of edge features, K the number of filter taps, N the number of
    nodes, S_{e} in R^{N x N} the GSO for edge feature e, x in R^{f x N} the
    input data where x_{g} in R^{N} is the graph signal representing feature
    g, and b in R^{G x N} the bias vector, with b_{g} in R^{N} representing the
    bias for feature f.

    Then, the LSI-GF is computed as
        y_{g} = \sum_{e=1}^{E}
                    \sum_{k=0}^{K-1}
                    \sum_{g=1}^{F}
                        [h_{f,g,e}]_{k} S_{e}^{k} x_{f}
                + b_{f}
    for g = 1, ..., G.

    Inputs:
        filter_taps (torch.tensor): array of filter taps; shape:
            output_features x edge_features x filter_taps x input_features
        GSO (torch.tensor): graph shift operator; shape:
            edge_features x number_nodes x number_nodes
        input (torch.tensor): input signal; shape:
            batch_size x input_features x number_nodes
        bias (torch.tensor): shape: output_features x number_nodes
            if the same bias is to be applied to all nodes, set number_nodes = 1
            so that b_{f} vector becomes b_{f} \mathbf{1}_{N}

    Outputs:
        output: filtered signals; shape:
            batch_size x output_features x number_nodes
    """
    # The basic idea of what follows is to start reshaping the input and the
    # GSO so the filter coefficients go just as a very plain and simple
    # linear operation, so that all the derivatives and stuff on them can be
    # easily computed.

    # h is output_features x edge_weights x filter_taps x input_features
    # S is edge_weighs x number_nodes x number_nodes
    # x is batch_size x input_features x number_nodes
    # b is output_features x number_nodes
    # Output:
    # y is batch_size x output_features x number_nodes

    # Get the parameter numbers:
    G = h.shape[0]
    E = h.shape[1]
    K = h.shape[2]
    F = h.shape[3]
    assert S.shape[1] == E
    N = S.shape[2]
    assert S.shape[3] == N
    B = x.shape[0]
    assert x.shape[1] == F
    assert x.shape[2] == N
    # Or, in the notation we've been using:
    # h in G x E x K x F
    # S in B x E x N x N
    # x in B x F x N
    # b in G x N
    # y in B x G x N

    # Now, we have x in B x F x N and S in B x E x N x N, and we want to come up
    # with matrix multiplication that yields z = x * S with shape
    # B x E x K x F x N.
    # For this, we first add the corresponding dimensions
    x = x.reshape([B, 1, F, N])
    S = S.reshape([B, E, N, N])
    z = x.reshape([B, 1, 1, F, N]).repeat(1, E, 1, 1, 1)  # This is for k = 0
    # We need to repeat along the E dimension, because for k=0, S_{e} = I for
    # all e, and therefore, the same signal values have to be used along all
    # edge feature dimensions.
    for k in range(1, K):
        x = torch.matmul(x, S)  # B x E x F x N
        xS = x.reshape([B, E, 1, F, N])  # B x E x 1 x F x N
        z = torch.cat((z, xS), dim=2)  # B x E x k x F x N
    # This output z is of size B x E x K x F x N
    # Now we have the x*S_{e}^{k} product, and we need to multiply with the
    # filter taps.
    # We multiply z on the left, and h on the right, the output is to be
    # B x N x F (the multiplication is not along the N dimension), so we reshape
    # z to be B x N x E x K x F and reshape it to B x N x EKG (remember we
    # always reshape the last dimensions), and then make h be E x K x F x G and
    # reshape it to EKF x G, and then multiply
    y = torch.matmul(z.permute(0, 4, 1, 2, 3).reshape([B, N, E * K * F]),
                     h.reshape([F, E * K * G]).permute(1, 0)).permute(0, 2, 1)
    # And permute againt to bring it from B x N x G to B x G x N.
    # Finally, add the bias
    if b is not None:
        y = y + b
    return y


class GraphFilterBatch(nn.Module):
    """
    GraphFilter Creates a (linear) layer that applies a graph filter

    Initialization:

        GraphFilter(in_features, out_features, filter_taps,
                    edge_features=1, bias=True)

        Inputs:
            in_features (int): number of input features (each feature is a graph
                signal)
            out_features (int): number of output features (each feature is a
                graph signal)
            filter_taps (int): number of filter taps
            edge_features (int): number of features over each edge
            bias (bool): add bias vector (one bias per feature) after graph
                filtering

        Output:
            torch.nn.Module for a graph filtering layer (also known as graph
            convolutional layer).

        Observation: Filter taps have shape
            out_features x edge_features x filter_taps x in_features

    Add graph shift operator:

        GraphFilter.addGSO(GSO) Before applying the filter, we need to define
        the GSO that we are going to use. This allows to change the GSO while
        using the same filtering coefficients (as long as the number of edge
        features is the same; but the number of nodes can change).

        Inputs:
            GSO (torch.tensor): graph shift operator; shape:
                Batch edge_features x number_nodes x number_nodes

    Forward call:

        y = GraphFilter(x)

        Inputs:
            x (torch.tensor): input data; shape:
                batch_size x in_features x number_nodes

        Outputs:
            y (torch.tensor): output; shape:
                batch_size x out_features x number_nodes
    """

    def __init__(self, F, G, K, E=1, bias=True):
        # K: Number of filter taps
        # GSOs will be added later.
        # This combines both weight scalars and weight vectors.
        # Bias will always be shared and scalar.

        # Initialize parent
        super().__init__()
        # Save parameters:
        self.F = F
        self.G = G
        self.K = K
        self.E = E
        self.S = None  # No GSO assigned yet
        # Create parameters:
        self.weight = nn.parameter.Parameter(torch.Tensor(G, E, K, F))
        if bias:
            self.bias = nn.parameter.Parameter(torch.Tensor(G, 1))
        else:
            self.register_parameter('bias', None)
        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        # Taken from _ConvNd initialization of parameters:
        stdv = 1. / math.sqrt(self.F * self.K)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def addGSO(self, S):
        # Every S has 4 dimensions.
        assert len(S.shape) == 4
        # S is of shape B x E x N x N
        assert S.shape[1] == self.E
        self.N = S.shape[2]
        assert S.shape[3] == self.N
        self.S = S

    def forward(self, x):
        # x is of shape: batchSize x dimInFeatures x numberNodesIn
        B = x.shape[0]
        F = x.shape[1]
        Nin = x.shape[2]
        # And now we add the zero padding
        if Nin < self.N:
            x = torch.cat((x,
                           torch.zeros(B, F, self.N - Nin) \
                           .type(x.dtype).to(x.device)
                           ), dim=2)
        # Compute the filter output
        u = BatchLSIGF(self.weight, self.S, x, self.bias)
        # So far, u is of shape batchSize x dimOutFeatures x numberNodes
        # And we want to return a tensor of shape
        # batchSize x dimOutFeatures x numberNodesIn
        # since the nodes between numberNodesIn and numberNodes are not required
        if Nin < self.N:
            u = torch.index_select(u, 2, torch.arange(Nin).to(u.device))
        return u

    def extra_repr(self):
        reprString = "in_features=%d, out_features=%d, " % (
            self.F, self.G) + "filter_taps=%d, " % (
                         self.K) + "edge_features=%d, " % (self.E) + \
                     "bias=%s, " % (self.bias is not None)
        if self.S is not None:
            reprString += "GSO stored"
        else:
            reprString += "no GSO stored"
        return reprString


class GraphFilterRNNBatch(nn.Module):
    """
    GraphFilter Creates a (linear) layer that applies a graph filter

    Initialization:

        GraphFilter(in_features, out_features, filter_taps,
                    edge_features=1, bias=True)

        Inputs:
            in_features (int): number of input features (each feature is a graph
                signal)
            out_features (int): number of output features (each feature is a
                graph signal)
            filter_taps (int): number of filter taps
            edge_features (int): number of features over each edge
            bias (bool): add bias vector (one bias per feature) after graph
                filtering

        Output:
            torch.nn.Module for a graph filtering layer (also known as graph
            convolutional layer).

        Observation: Filter taps have shape
            out_features x edge_features x filter_taps x in_features

    Add graph shift operator:

        GraphFilter.addGSO(GSO) Before applying the filter, we need to define
        the GSO that we are going to use. This allows to change the GSO while
        using the same filtering coefficients (as long as the number of edge
        features is the same; but the number of nodes can change).

        Inputs:
            GSO (torch.tensor): graph shift operator; shape:
                Batch edge_features x number_nodes x number_nodes

    Forward call:

        y = GraphFilter(x)

        Inputs:
            x (torch.tensor): input data; shape:
                batch_size x in_features x number_nodes

        Outputs:
            y (torch.tensor): output; shape:
                batch_size x out_features x number_nodes
    """

    def __init__(self, G, F, H, K, E=1, bias=True):
        # K: Number of filter taps
        # GSOs will be added later.
        # This combines both weight scalars and weight vectors.
        # Bias will always be shared and scalar.

        # Initialize parent
        super().__init__()
        # Save parameters:
        self.F = F
        self.G = G
        self.H = H
        self.K = K
        self.E = E
        self.S = None  # No GSO assigned yet
        # Create parameters:
        self.weight_A = nn.parameter.Parameter(torch.Tensor(H, E, K, G))
        self.weight_B = nn.parameter.Parameter(torch.Tensor(H, E, K, H))
        self.weight_D = nn.parameter.Parameter(torch.Tensor(F, E, K, H))
        if bias:
            self.bias_A = nn.parameter.Parameter(torch.Tensor(H, 1))
            self.bias_B = nn.parameter.Parameter(torch.Tensor(H, 1))
            self.bias_D = nn.parameter.Parameter(torch.Tensor(G, 1))
        else:
            self.register_parameter('bias', None)
        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        # Taken from _ConvNd initialization of parameters:
        stdv_a = 1. / math.sqrt(self.F * self.K)
        self.weight_A.data.uniform_(-stdv_a, stdv_a)
        if self.bias_A is not None:
            self.bias_A.data.uniform_(-stdv_a, stdv_a)

        stdv_b = 1. / math.sqrt(self.H * self.K)
        self.weight_B.data.uniform_(-stdv_b, stdv_b)
        if self.bias_B is not None:
            self.bias_B.data.uniform_(-stdv_b, stdv_b)

        stdv_d = 1. / math.sqrt(self.H * self.K)
        self.weight_U.data.uniform_(-stdv_d, stdv_d)
        if self.bias_U is not None:
            self.bias_U.data.uniform_(-stdv_d, stdv_d)

    def addGSO(self, S):
        # Every S has 4 dimensions.
        assert len(S.shape) == 4
        # S is of shape B x E x N x N
        assert S.shape[1] == self.E
        self.N = S.shape[2]
        assert S.shape[3] == self.N
        self.S = S

    def updateHiddenState(self, hiddenState):

        self.hiddenState = hiddenState

    def forward(self, x, hidden_prev):
        # x is of shape: batchSize x dimInFeatures x numberNodesIn
        B = x.shape[0]
        F = x.shape[1]
        Nin = x.shape[2]
        # And now we add the zero padding
        if Nin < self.N:
            x = torch.cat((x,
                           torch.zeros(B, F, self.N - Nin) \
                           .type(x.dtype).to(x.device)
                           ), dim=2)
        # Compute the filter output
        u_a = BatchLSIGF(self.weight_A, self.S, x, self.bias_A)
        u_b = BatchLSIGF(self.weight_B, self.S, self.hiddenState, self.bias_B)

        sigma = nn.ReLU(inplace=True)
        self.hiddenStateNext = sigma(u_a + u_b)

        u = BatchLSIGF(self.weight_D, self.S, self.hiddenStateNext, self.bias_D)
        self.updateHiddenState(self.hiddenStateNext)

        # So far, u is of shape batchSize x dimOutFeatures x numberNodes
        # And we want to return a tensor of shape
        # batchSize x dimOutFeatures x numberNodesIn
        # since the nodes between numberNodesIn and numberNodes are not required
        if Nin < self.N:
            u = torch.index_select(u, 2, torch.arange(Nin).to(u.device))
        return u

    def extra_repr(self):
        reprString = "in_features=%d, out_features=%d, hidden_features=%d," % (
            self.G, self.F, self.H) + "filter_taps=%d, " % (
                         self.K) + "edge_features=%d, " % (self.E) + \
                     "bias=%s, " % (self.bias_D is not None)
        if self.S is not None:
            reprString += "GSO stored"
        else:
            reprString += "no GSO stored"
        return reprString




class NoActivation(nn.Module):
    """
    NoActivation creates an activation layer that does nothing
        It is for completeness, to be able to switch between linear models
        and nonlinear models, without altering the entire architecture model
    Initialization:
        NoActivation()
        Output:
            torch.nn.Module for an empty activation layer
    Forward call:
        y = NoActivation(x)
        Inputs:
            x (torch.tensor): input data; shape:
                batch_size x dim_features x number_nodes
        Outputs:
            y (torch.tensor): activated data; shape:
                batch_size x dim_features x number_nodes
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

    def extra_repr(self):
        reprString = "No Activation Function"
        return reprString


class NoPool(nn.Module):
    """
    This is a pooling layer that actually does no pooling. It has the same input
    structure and methods of MaxPoolLocal() for consistency. Basically, this
    allows us to change from pooling to no pooling without necessarily creating
    a new architecture.

    In any case, we're pretty sure this function should never ship, and pooling
    can be avoided directly when defining the architecture.
    """

    def __init__(self, nInputNodes, nOutputNodes, nHops):
        super().__init__()
        self.nInputNodes = nInputNodes
        self.nOutputNodes = nOutputNodes
        self.nHops = nHops
        self.neighborhood = None

    def addGSO(self, GSO):
        # This is necessary to keep the form of the other pooling strategies
        # within the SelectionGNN framework. But we do not care about any GSO.
        pass

    def forward(self, x):
        # x should be of shape batchSize x dimNodeSignals x nInputNodes
        assert x.shape[2] == self.nInputNodes
        # Check that there are at least the same number of nodes that
        # we will keep (otherwise, it would be unpooling, instead of
        # pooling)
        assert x.shape[2] >= self.nOutputNodes
        # And do not do anything
        return x

    def extra_repr(self):
        reprString = "in_dim=%d, out_dim=%d, number_hops = %d, " % (
            self.nInputNodes, self.nOutputNodes, self.nHops)
        reprString += "no neighborhood needed"
        return reprString


class MaxPoolLocal(nn.Module):
    """
    MaxPoolLocal Creates a pooling layer on graphs by selecting nodes

    Initialization:

        MaxPoolLocal(in_dim, out_dim, number_hops)

        Inputs:
            in_dim (int): number of nodes at the input
            out_dim (int): number of nodes at the output
            number_hops (int): number of hops to pool information

        Output:
            torch.nn.Module for a local max-pooling layer.

        Observation: The selected nodes for the output are always the top ones.

    Add a neighborhood set:

    Add graph shift operator:

        GraphFilter.addGSO(GSO) Before being used, we need to define the GSO
        that will determine the neighborhood that we are going to pool.

        Inputs:
            GSO (torch.tensor): graph shift operator; shape:
                edge_features x number_nodes x number_nodes

    Forward call:

        v = MaxPoolLocal(x)

        Inputs:
            x (torch.tensor): input data; shape:
                batch_size x dim_features x in_dim

        Outputs:
            y (torch.tensor): pooled data; shape:
                batch_size x dim_features x out_dim
    """

    def __init__(self, nInputNodes, nOutputNodes, nHops):

        super().__init__()
        self.nInputNodes = nInputNodes
        self.nOutputNodes = nOutputNodes
        self.nHops = nHops
        self.neighborhood = None

    def addGSO(self, S):
        # Every S has 3 dimensions.
        assert len(S.shape) == 3
        # S is of shape E x N x N (And I don't care about E, because the
        # computeNeighborhood function takes care of it)
        self.N = S.shape[1]
        assert S.shape[2] == self.N
        # Get the device (before operating with S and losing it, it's cheaper
        # to store the device now, than to duplicate S -i.e. keep a numpy and a
        # tensor copy of S)
        device = S.device
        # Move the GSO to cpu and to np.array so it can be handled by the
        # computeNeighborhood function
        S = np.array(S.cpu())
        # Compute neighborhood
        neighborhood = graphTools.computeNeighborhood(S, self.nHops,
                                                      self.nOutputNodes,
                                                      self.nInputNodes, 'matrix')
        # And move the neighborhood back to a tensor
        neighborhood = torch.tensor(neighborhood).to(device)
        # The neighborhood matrix has to be a tensor of shape
        #   nOutputNodes x maxNeighborhoodSize
        assert neighborhood.shape[0] == self.nOutputNodes
        assert neighborhood.max() <= self.nInputNodes
        # Store all the relevant information
        self.maxNeighborhoodSize = neighborhood.shape[1]
        self.neighborhood = neighborhood

    def forward(self, x):
        # x should be of shape batchSize x dimNodeSignals x nInputNodes
        batchSize = x.shape[0]
        dimNodeSignals = x.shape[1]
        assert x.shape[2] == self.nInputNodes
        # Check that there are at least the same number of nodes that
        # we will keep (otherwise, it would be unpooling, instead of
        # pooling)
        assert x.shape[2] >= self.nOutputNodes
        # And given that the self.neighborhood is already a torch.tensor matrix
        # we can just go ahead and get it.
        # So, x is of shape B x F x N. But we need it to be of shape
        # B x F x N x maxNeighbor. Why? Well, because we need to compute the
        # maximum between the value of each node and those of its neighbors.
        # And we do this by applying a torch.max across the rows (dim = 3) so
        # that we end up again with a B x F x N, but having computed the max.
        # How to fill those extra dimensions? Well, what we have is neighborhood
        # matrix, and we are going to use torch.gather to bring the right
        # values (torch.index_select, while more straightforward, only works
        # along a single dimension).
        # Each row of the matrix neighborhood determines all the neighbors of
        # each node: the first row contains all the neighbors of the first node,
        # etc.
        # The values of the signal at those nodes are contained in the dim = 2
        # of x. So, just for now, let's ignore the batch and feature dimensions
        # and imagine we have a column vector: N x 1. We have to pick some of
        # the elements of this vector and line them up alongside each row
        # so that then we can compute the maximum along these rows.
        # When we torch.gather along dimension 0, we are selecting which row to
        # pick according to each column. Thus, if we have that the first row
        # of the neighborhood matrix is [1, 2, 0] means that we want to pick
        # the value at row 1 of x, at row 2 of x in the next column, and at row
        # 0 of the last column. For these values to be the appropriate ones, we
        # have to repeat x as columns to build our b x F x N x maxNeighbor
        # matrix.
        x = x.unsqueeze(3)  # B x F x N x 1
        x = x.repeat([1, 1, 1, self.maxNeighborhoodSize])  # BxFxNxmaxNeighbor
        # And the neighbors that we need to gather are the same across the batch
        # and feature dimensions, so we need to repeat the matrix along those
        # dimensions
        gatherNeighbor = self.neighborhood.reshape([1, 1,
                                                    self.nOutputNodes,
                                                    self.maxNeighborhoodSize])
        gatherNeighbor = gatherNeighbor.repeat([batchSize, dimNodeSignals, 1, 1])
        # And finally we're in position of getting all the neighbors in line
        xNeighbors = torch.gather(x, 2, gatherNeighbor)
        #   B x F x nOutput x maxNeighbor
        # Note that this gather function already reduces the dimension to
        # nOutputNodes.
        # And proceed to compute the maximum along this dimension
        v, _ = torch.max(xNeighbors, dim=3)
        return v

    def extra_repr(self):
        reprString = "in_dim=%d, out_dim=%d, number_hops = %d, " % (
            self.nInputNodes, self.nOutputNodes, self.nHops)
        if self.neighborhood is not None:
            reprString += "neighborhood stored"
        else:
            reprString += "NO neighborhood stored"
        return reprString
