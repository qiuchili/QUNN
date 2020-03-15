# -*- coding: utf-8 -*-
import torch
from torch.optim.optimizer import Optimizer


class RMSprop_Unitary(Optimizer):
    """Implements RMSprop gradient descent for unitary matrix.
        
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        
    .. note::
        This is the vanilla version of the gradient descent for unitary matrix, 
        i.e. formula (6) in H. D. Tagare. Notes on optimization on Stiefel manifolds. 
        Technical report, Yale University, 2011, and formula (6) in Scott Wisdom, 
        Thomas Powers, John Hershey, Jonathan Le Roux, and Les Atlas. Full-capacity 
        unitary recurrent neural networks. In NIPS 2016. 

        .. math::
                  A = G^H*W - W^H*G \\
                  W_new = (I+lr/2 * A)^(-1)*(I-lr/2 * A)*W

        where W, G and lr denote the parameters, gradient
        and learning rate respectively.
    """

    def __init__(self, params, lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0, centered=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= momentum:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= alpha:
            raise ValueError("Invalid alpha value: {}".format(alpha))

        defaults = dict(lr=lr, momentum=momentum, alpha=alpha, eps=eps, centered=centered, weight_decay=weight_decay)
        super(RMSprop_Unitary, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RMSprop_Unitary, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('momentum', 0)
            group.setdefault('centered', False)

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
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('RMSprop does not support sparse gradients')
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['square_avg'] = torch.zeros_like(p.data)
                    if group['momentum'] > 0:
                        state['momentum_buffer'] = torch.zeros_like(p.data)
                    if group['centered']:
                        state['grad_avg'] = torch.zeros_like(p.data)
                        
                        
                square_avg = state['square_avg']
                alpha = group['alpha']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                square_avg.mul_(alpha).addcmul_(1 - alpha, grad, grad)
                
                if group['centered']:
                    grad_avg = state['grad_avg']
                    grad_avg.mul_(alpha).add_(1 - alpha, grad)
                    avg = square_avg.addcmul(-1, grad_avg, grad_avg).sqrt().add_(group['eps'])
                else:
                    avg = square_avg.sqrt().add_(group['eps'])

                
                if group['momentum'] > 0:
                    buf = state['momentum_buffer']
                    buf.mul_(group['momentum']).addcdiv_(grad, avg)
                    grad = buf
#                    p.data.add_(-group['lr'], buf)
                else:
                    grad = torch.zeros_like(p.data).addcdiv(grad, avg)
#                    p.data.addcdiv_(-group['lr'], grad, avg)
                
#                print(state['square_avg'] - square_avg) 
                #print(grad-p.grad.data)
                #print(grad)
                
                lr = group['lr']
                
                G_r = grad[:,:,0]
                G_i = grad[:,:,1]
                W_r = p.data[:,:,0]
                W_i = p.data[:,:,1]

                # A = G^H W - W^H G
                # A_r = (G_r^T W_r)+ (G_i^T W_i)- (W_r^T G_r) - (W_i^T G_i)
                # A_i = (G_r^T W_i)- (G_i^T W_r)- (W_r^T G_i)+ (W_i^T G_r)
                
                A_skew_r = torch.mm(G_r.t(),W_r) - torch.mm(W_r.t(),G_r) + torch.mm(G_i.t(),W_i) -  torch.mm(W_i.t(),G_i)
                A_skew_i = torch.mm(G_r.t(),W_i) + torch.mm(W_i.t(),G_r) - torch.mm(G_i.t(),W_r) - torch.mm(W_r.t(),G_i) 
                  
                #W_new = (I+lr/2 * A)^(-1)*(I-lr/2 * A)*W
                idm = torch.eye(grad.shape[0]).to(grad.device)
    
                
                # cayley_numer = I-lr/2 * A
                cayley_numer_r = idm - (lr/2)* A_skew_r
                cayley_numer_i = - (lr/2)* A_skew_i
                
                # cayley_demon = (I + lr/2 * A)^(-1)
                X = idm + (lr/2)* A_skew_r
                Y = (lr/2)* A_skew_i
                
                #(X + i*Y)^-1 = (X + Y*X^-1*Y)^-1 - i*(Y + X*Y^-1*X)^-1
                if X.det() == 0:
                    X.add_(idm,alpha=1e-5)
                
                if Y.det() == 0:
                    Y.add_(idm,alpha=1e-5)
                
                inv_cayley_denom_r = X + torch.mm(Y,torch.mm(X.inverse(),Y))
                if inv_cayley_denom_r.det() == 0:
                    inv_cayley_denom_r.add_(idm,alpha=1e-5)
                
                cayley_denom_r = inv_cayley_denom_r.inverse()
                
                #cayley_denom_i = - (Y + torch.mm(X,torch.mm(Y.inverse(),X))).inverse()
                inv_cayley_denom_i = Y + torch.mm(X,torch.mm(Y.inverse(),X))
                if inv_cayley_denom_i.det() == 0:
                    inv_cayley_denom_i.add_(idm,alpha=1e-5)
                
                cayley_denom_i = - inv_cayley_denom_i.inverse()
                
                #W_new = cayley_denom*cayley_numer*W
                W_new_r = torch.mm(cayley_denom_r, cayley_numer_r) - torch.mm(cayley_denom_i, cayley_numer_i)
                W_new_i = torch.mm(cayley_denom_r, cayley_numer_i) + torch.mm(cayley_denom_i, cayley_numer_r)            
                
                W_new_r_2 = torch.mm(W_new_r, W_r) - torch.mm(W_new_i, W_i)
                W_new_i_2 = torch.mm(W_new_r, W_i) + torch.mm(W_new_i, W_r)
                
                
                p.data = torch.stack([W_new_r_2, W_new_i_2], dim= -1)
               
        return loss


