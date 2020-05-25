# -*- coding: utf-8 -*-
import torch
from torch.optim.optimizer import Optimizer, required
        
class Vanilla_Unitary(Optimizer):
    """Implements gradient descent for unitary matrix.
        
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

    def __init__(self, params, lr=required, device = torch.device('cpu')):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        self.device = device
        defaults = dict(lr=lr)
        super(Vanilla_Unitary, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Vanilla_Unitary, self).__setstate__(state)
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
#            weight_decay = group['weight_decay']
#            momentum = group['momentum']
#            dampening = group['dampening']
#            nesterov = group['nesterov']
            lr = group['lr']
            
            for p in group['params']:
    
                if p.grad is None:
                    continue
                
                d_p = p.grad.data #G
                
                G_r = d_p[:,:,0]
                G_i = d_p[:,:,1]
                W_r = p.data[:,:,0]
                W_i = p.data[:,:,1]


                # A = G^H W - W^H G
                # A_r = (G_r^T W_r)+ (G_i^T W_i)- (W_r^T G_r) - (W_i^T G_i)
                # A_i = (G_r^T W_i)- (G_i^T W_r)- (W_r^T G_i)+ (W_i^T G_r)
                
                A_skew_r = torch.mm(G_r.t(),W_r) + torch.mm(G_i.t(),W_i) - torch.mm(W_r.t(),G_r) -  torch.mm(W_i.t(),G_i)
                A_skew_i = torch.mm(G_r.t(),W_i) - torch.mm(G_i.t(),W_r) - torch.mm(W_r.t(),G_i) +  torch.mm(W_i.t(),G_r)
                
                #print(torch.max(A_skew_r + A_skew_r.t()))
                #print(torch.max(A_skew_i - A_skew_i.t()))
  
                #W_new = (I+lr/2 * A)^(-1)*(I-lr/2 * A)*W
                idm = torch.eye(d_p.shape[0]).to(d_p.device)
    
                
                # cayley_numer = I-lr/2 * A
                cayley_numer_r = idm + (lr/2)* A_skew_r
                cayley_numer_i = + (lr/2)* A_skew_i
                
                # cayley_demon = (I + lr/2 * A)^(-1)
                X = idm - (lr/2)* A_skew_r
                Y = -(lr/2)* A_skew_i
                
                #(X + i*Y)^-1 = (X + Y*X^-1*Y)^-1 - i*(Y + X*Y^-1*X)^-1
                
                #cayley_denom_r = (X + torch.mm(Y,torch.mm(X.inverse(),Y))).inverse()
                
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