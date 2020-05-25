# -*- coding: utf-8 -*-
import torch
from torch.optim.optimizer import Optimizer, required


class SGD_Unitary(Optimizer):
    """Implements SGD gradient descent for unitary matrix.
        
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        
    .. note::
        This is the vanilla version of the gradient descent for unitary matrix, 
        i.e. formula (6) in H. D. Tagare. Notes on optimization on Stiefel manifolds. 
        Technical report, Yale University, 2011, and formula (6) in Scott Wisdom, 
        Thomas Powers, John Hershey, Jonathan Le Roux, and Les Atlas. Full-capacity 
        unitary recurrentneural networks. In NIPS 2016. 

        .. math::
                  A = G^H*W - W^H*G \\
                  W_new = (I+lr/2 * A)^(-1)*(I-lr/2 * A)*W

        where W, G and lr denote the parameters, gradient
        and learning rate respectively.
    """

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0,  eps=1e-10, nesterov=False, device = torch.device('cpu')):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening, eps=eps,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")

        self.device = device
        super(SGD_Unitary, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD_Unitary, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)


    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data).add_(group['eps'], d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf
                
                lr = group['lr']

                G_r = d_p[:,:,0]
                G_i = d_p[:,:,1]
                W_r = p.data[:,:,0]
                W_i = p.data[:,:,1]


                # A = G^H W - W^H G
                # A_r = (G_r^T W_r)+ (G_i^T W_i)- (W_r^T G_r) - (W_i^T G_i)
                # A_i = (G_r^T W_i)- (G_i^T W_r)- (W_r^T G_i)+ (W_i^T G_r)
                
                A_skew_r = torch.mm(G_r.t(),W_r) + torch.mm(G_i.t(),W_i) - torch.mm(W_r.t(),G_r) -  torch.mm(W_i.t(),G_i)
                A_skew_i = torch.mm(G_r.t(),W_i) - torch.mm(G_i.t(),W_r) - torch.mm(W_r.t(),G_i) +  torch.mm(W_i.t(),G_r)
                
  
                #W_new = (I+lr/2 * A)^(-1)*(I-lr/2 * A)*W
                idm = torch.eye(d_p.shape[0]).to(d_p.device)
    
                
                # cayley_numer = I-lr/2 * A
                cayley_numer_r = idm + (lr/2)* A_skew_r
                cayley_numer_i = + (lr/2)* A_skew_i
                
                # cayley_demon = (I + lr/2 * A)^(-1)
                X = idm - (lr/2)* A_skew_r
                Y = -(lr/2)* A_skew_i
                
                #(X + i*Y)^-1 = (X + Y*X^-1*Y)^-1 - i*(Y + X*Y^-1*X)^-1
                if X.det() == 0:
                    X.add_(idm,alpha=1e-8)
                
                if Y.det() == 0:
                    Y.add_(idm,alpha=1e-8)
                
                inv_cayley_denom_r = X + torch.mm(Y,torch.mm(X.inverse(),Y))
                if inv_cayley_denom_r.det() == 0:
                    inv_cayley_denom_r.add_(idm,alpha=1e-8)
                
                cayley_denom_r = inv_cayley_denom_r.inverse()
                
                #cayley_denom_i = - (Y + torch.mm(X,torch.mm(Y.inverse(),X))).inverse()
                inv_cayley_denom_i = Y + torch.mm(X,torch.mm(Y.inverse(),X))
                if inv_cayley_denom_i.det() == 0:
                    inv_cayley_denom_i.add_(idm,alpha=1e-8)
                
                cayley_denom_i = - inv_cayley_denom_i.inverse()

                #W_new = cayley_denom*cayley_numer*W
                W_new_r = torch.mm(cayley_denom_r, cayley_numer_r) - torch.mm(cayley_denom_i, cayley_numer_i)
                W_new_i = torch.mm(cayley_denom_r, cayley_numer_i) + torch.mm(cayley_denom_i, cayley_numer_r)

                W_new_r_2 = torch.mm(W_new_r, W_r) - torch.mm(W_new_i, W_i)
                W_new_i_2 = torch.mm(W_new_r, W_i) + torch.mm(W_new_i, W_r)
                
                p.data = torch.stack([W_new_r_2, W_new_i_2], dim= -1)

        return loss