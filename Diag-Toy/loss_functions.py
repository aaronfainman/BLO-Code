

import torch
import torch.nn as nn
import torch.nn.functional as F
from models import LearnableParameter



def lower_loss_cl(ll_model:nn.Module, 
                data:dict, 
                ul_model:nn.Module|None=None, 
                stoch:bool=False, 
                seed:int = None, 
                **kwargs):

        features_train, labels_train = data['features_train'], data['labels_train']
        if stoch:
                gen = torch.Generator()
                gen.manual_seed(seed)
                n = torch.randint(0, len(features_train), (1,), generator=gen)
                features_train, labels_train = ( features_train[[n],:], labels_train[[n]] )

        pred_logits, last_layer_features = ll_model(features_train, return_features = True)
        per_sample_logit_loss = F.cross_entropy(pred_logits,labels_train, reduction="none")

        if ul_model:
                # detach statement necessary for UL model weights not to
                # be updated to fit the loss function
                batch_weights = ul_model(last_layer_features.detach())
        else:
                batch_weights = 1

        loss = 1/len(features_train)*(batch_weights * per_sample_logit_loss).sum()

        return loss


def upper_loss_cl(ll_model:nn.Module, 
                data:dict, 
                ul_model:nn.Module | None, 
                stoch:bool=False, 
                seed:int = None, 
                **kwargs):

        features_val, labels_val = data['features_val'], data['labels_val']
        if stoch:
                gen = torch.Generator()
                gen.manual_seed(seed)
                n = torch.randint(0, len(features_val), (1,), generator=gen)
                features_val, labels_val = ( features_val[[n],:], labels_val[[n]] )

        pred_labels_val, _ = ll_model(features_val, return_features=True)

        loss =  F.cross_entropy(pred_labels_val,labels_val).mean()
        if ul_model:
            loss += 0.1*sum(p.pow(2).sum() for p in ul_model.parameters())

        return loss



def lower_loss_cl_window(ll_model:nn.Module, 
                data:dict, 
                ul_model:nn.Module|None=None, 
                stoch:bool=False, 
                seed:int = None, 
                **kwargs):

        features_train, labels_train = data['features_train'], data['labels_train']
        if stoch:
                gen = torch.Generator()
                gen.manual_seed(seed)
                n = torch.randint(0, len(features_train), (1,), generator=gen)
                features_train, labels_train = ( features_train[[n],:], labels_train[[n]] )

        pred_logits, _ = ll_model(features_train, return_features = True)
        per_sample_logit_loss = F.cross_entropy(pred_logits,labels_train, reduction="none")

        if ul_model:
                batch_weights = ul_model(per_sample_logit_loss)
        else:
                batch_weights = 1

        loss = 1/len(features_train)*(batch_weights * per_sample_logit_loss).sum()

        return loss


def upper_loss_cl_window(ll_model:nn.Module, 
                data:dict, 
                ul_model:nn.Module, 
                stoch:bool=False, 
                seed:int = None, 
                **kwargs):

        features_val, labels_val = data['features_val'], data['labels_val']
        # features_train, labels_train = data['features_train'], data['labels_train']
        if stoch:
                gen = torch.Generator()
                gen.manual_seed(seed)
                n = torch.randint(0, len(features_val), (1,), generator=gen)
                features_val, labels_val = ( features_val[[n],:], labels_val[[n]] )
                # n = torch.randint(0, len(features_train), (1,), generator=gen)
                # features_train, labels_train = ( features_train[[n],:], labels_train[[n]] )

        pred_labels_val = ll_model(features_val)
        # pred_labels_train = ll_model(features_val)

        loss =  F.cross_entropy(pred_labels_val,labels_val).mean()
        loss += 0.1*sum(p.pow(2).sum() for p in ul_model.parameters())
        # ul_weight_loss = sum(p.pow(2).sum() for p in ul_model.parameters())
        
        # loss = F.cross_entropy(pred_labels_val,labels_val).mean() #+ 0.1*ul_weight_loss

        return loss

def test_loss_cl(ll_model, ul_model, data, stoch:bool=False, seed:int = None, **kwargs):

        features_test, labels_test = data['features_test'], data['labels_test']

        pred_labels_val = ll_model(features_test)
        loss = F.cross_entropy(pred_labels_val,labels_test).mean()

        return loss



def lower_loss_lo(ll_model:nn.Module, 
                data:dict, 
                ul_model:nn.Module|None=None, 
                stoch:bool=False, 
                seed:int = None, 
                **kwargs):

        features_train, labels_train = data['features_train'], data['labels_train']
        if stoch:
                gen = torch.Generator()
                gen.manual_seed(seed)
                n = torch.randint(0, len(features_train), (1,), generator=gen)
                features_train, labels_train = ( features_train[[n],:], labels_train[[n]] )

        pred_logits = ll_model(features_train, return_features = False)

        loss = 1/len(features_train)*( ul_model(pred_logits,labels_train)).sum()

        return loss


def upper_loss_lo(ll_model:nn.Module, 
                data:dict, 
                ul_model:nn.Module, 
                stoch:bool=False, 
                seed:int = None, 
                **kwargs):

        features_val, labels_val = data['features_val'], data['labels_val']
        if stoch:
                gen = torch.Generator()
                gen.manual_seed(seed)
                n = torch.randint(0, len(features_val), (1,), generator=gen)
                features_val, labels_val = ( features_val[[n],:], labels_val[[n]] )


        pred_labels_val = ll_model(features_val, return_features = False)

        # ul_weight_loss = sum(p.pow(2).sum() for p in ul_model.parameters())
        
        loss = F.cross_entropy(pred_labels_val,labels_val).mean() #+ 0.1*ul_weight_loss

        return loss

def test_loss_lo(ll_model, ul_model, data, stoch:bool=False, seed:int = None, **kwargs):

        features_test, labels_test = data['features_test'], data['labels_test']

        pred_labels_val = ll_model(features_test)
        loss = F.cross_entropy(pred_labels_val,labels_test).mean()

        return loss



def lower_loss_toy(ll_model:LearnableParameter, 
                data:dict, 
                ul_model:LearnableParameter, 
                stoch:bool=False, 
                seed:int = None, 
                **kwargs):

    w = ll_model()
    theta = ul_model()

    return (w**4).sum() +(1/240*w**2*theta**2).sum()+(w**2).sum()+(theta**2).sum()
#     return torch.pow(w, 4).sum() + torch.pow(theta, 2).sum()

def upper_loss_toy(ll_model:LearnableParameter, 
                data:dict, 
                ul_model:LearnableParameter, 
                stoch:bool=False, 
                seed:int = None, 
                **kwargs):
    w = ll_model()
    theta = ul_model()

    return (2*w**4+w**2).sum()+(1/2000*(theta**4-theta**2)**2).sum()



# def lower_loss_lreg(model, params_dict, data, stoch:bool=False, seed:int = None):

#         features_train, labels_train = data['features_train'], data['labels_train']
#         if stoch:
#                 torch.manual_seed(seed)
#                 n = torch.randint(0, len(features_train), (1,))
#                 pred_labels_train = model(features_train[[n],:])
#                 mse_loss = 1/2*F.mse_loss(pred_labels_train,labels_train[[n],:])
#         else:
#                 pred_labels_train = model(features_train)
#                 mse_loss = 1/2*F.mse_loss(pred_labels_train,labels_train)

#         # 2. Quadratic Regularization: sum over layers (w_i^T * Theta_i * w_i)
#         reg_loss = 0
#         # Assuming for every weight w_i, there is a corresponding theta_i
#         # If theta is a vector (diagonal matrix), we use element-wise multiplication

#         w_keys = [k for k in params_dict.keys() if k.startswith('w_')]
    
#         for k in w_keys:
#                 # Extract the index (e.g., '0' from 'w_0')
#                 idx = k.split('_')[1]
                
#                 w = params_dict[f"w_{idx}"]
#                 theta = params_dict[f"t_{idx}"] # The corresponding theta
                
#                 # Flatten for vector-based math
#                 w_flat = w.reshape([-1,1])
#                 theta_w = w_flat.T@ theta
                
#                 # theta is likely a vector (diagonal preconditioner)
#                 # We use exp(theta) to ensure the penalty is always positive
#                 # This prevents the weights from exploding
#                 # theta_positive = torch.exp(theta.view(-1))
                
#                 reg_loss += 0.5 * torch.sum( theta_w@theta_w.T)

#         return mse_loss + reg_loss

# def upper_loss_lreg(model, params_dict, data, stoch:bool=False, seed:int = None):

#         features_val, labels_val = data['features_val'], data['labels_val']
#         if stoch:
#                 torch.manual_seed(seed)
#                 n = torch.randint(0, len(features_val), (1,))
#                 pred_labels_val = model(features_val[[n],:])
#                 mse_loss = 1/2*F.mse_loss(pred_labels_val,labels_val[[n],:])
#         else:
#                 pred_labels_val = model(features_val)
#                 mse_loss = 1/2*F.mse_loss(pred_labels_val,labels_val)

#         return mse_loss


# def lower_loss_cal(model, params_dict, data, stoch=False, seed=None):
    
#     features_train, labels_train = data['features_train'], data['labels_train']
    
# #     weights = torch.softmax(params_dict['t_0'].squeeze(), dim=0)
#     weights = torch.sigmoid(params_dict['t_0'].squeeze())

#     # Map w_ keys back to model parameter names
#     param_names = [name for name, _ in model.named_parameters()]
#     model_call_params = {param_names[i]: params_dict[f'w_{i}'] 
#                          for i in range(len(param_names))}

#     if stoch:
#         torch.manual_seed(seed)
#         n = torch.randint(0, len(features_train), (1,))
#         features_train = features_train[[n], :]
#         labels_train   = labels_train[[n], :]
#         weights        = weights[n]

#     pred           = functional_call(model, model_call_params, features_train)
#     per_sample_mse = F.mse_loss(pred, labels_train, reduction='none').mean(dim=-1)
    
#     return 0.5 * (weights * per_sample_mse).sum()


# def upper_loss_cal(model, params_dict, data, stoch:bool=False, seed:int = None):

#         features_val, labels_val = data['features_val'], data['labels_val']
#         if stoch:
#                 torch.manual_seed(seed)
#                 n = torch.randint(0, len(features_val), (1,))
#                 pred_labels_val = model(features_val[[n],:])
#                 mse_loss = 1/2*F.mse_loss(pred_labels_val,labels_val[[n],:])
#         else:
#                 pred_labels_val = model(features_val)
#                 mse_loss = 1/2*F.mse_loss(pred_labels_val,labels_val)

#         return mse_loss



# def lower_loss_toy(model, params_dict, data, stoch=False, seed=None):
#     # Use the ACTUAL names of the parameters in your model/dict
#     # If using our previous naming convention:
#     w = params_dict["w_0"]
#     theta = params_dict["t_0"]
    
#     # Ensure they are treated as scalars for the math
#     # loss = w^4 + theta^2

#     return (w**4).sum() +(1/240*w**2*theta**2).sum()+(w**2).sum()+(theta**2).sum()
# #     return torch.pow(w, 4).sum() + torch.pow(theta, 2).sum()

# def upper_loss_toy(model, params_dict, data, stoch=False, seed=None):
#     w = params_dict["w_0"]
#     theta = params_dict["t_0"]

#     return (2*w**4+w**2).sum()+(1/2000*(theta**4-theta**2)**2).sum()


# def lower_loss_toy2(model, params_dict, data, stoch=False, seed=None):
#     # Use the ACTUAL names of the parameters in your model/dict
#     # If using our previous naming convention:
#     w = params_dict["w_0"]
#     theta = params_dict["t_0"]

#     c=500
    
#     # Ensure they are treated as scalars for the math
#     # loss = w^4 + theta^2

#     return ((w-1)**4 - 2*(w-1)**2+1).sum()+0.5/c*(theta**2).sum()


# def upper_loss_toy2(model, params_dict, data, stoch=False, seed=None):
#     w = params_dict["w_0"]
#     theta = params_dict["t_0"]
#     c=500

#     return ((w+1)**4-2*(w+1)**2+1/c*(theta-theta**2)**2).sum()+1#+((w*theta)**2).sum()
    



# def lower_loss_multitask(model, params_dict, data, stoch=False, seed=None):
#     """
#     Decentralised lower loss: g(w, θ) = (1/K) Σ_k g_k(w_k, θ)
    
#     data: list of K dicts, each with 'features_train' and 'labels_train'
#     params_dict: expects w_0...w_{K-1} as per-agent weights, t_0 as theta
#     """
#     def get_agent_params(params_dict, k, n_model_params):
#         return [params_dict[f'w_{k * n_model_params + i}'] for i in range(n_model_params)]

#     K = len(data)
#     # theta = torch.sigmoid(params_dict['t_0'].squeeze())
#     theta = params_dict['t_0'].squeeze()
#     theta = 1/2*torch.abs(theta+theta.T)
#     param_names = [name for name, _ in model.named_parameters()]
#     n_model_params = len(param_names)

#     data_loss = torch.tensor(0.0)

#     for k, agent_data in enumerate(data):
#         features_train = agent_data[f'features_train']
#         labels_train   = agent_data[f'labels_train']

#         # Each agent has its own slice of w parameters: w_{k*n_model_params + i}
#         model_call_params = {
#             param_names[i]: params_dict[f'w_{k * n_model_params + i}']
#             for i in range(n_model_params)
#         }

#         if stoch:
#             torch.manual_seed(seed + k)  # different seed per agent
#             n = torch.randint(0, len(features_train), (1,))
#             features_train = features_train[[n], :]
#             labels_train   = labels_train[[n], :]

#         pred           = functional_call(model, model_call_params, features_train)
#         per_sample_mse = F.mse_loss(pred, labels_train, reduction='none').mean(dim=-1)
#         data_loss     = data_loss + 0.5 * (per_sample_mse).mean()

#     reg_loss = torch.tensor(0.0)
#     for k in range(K):
#         for l in range(K):
#             w_k = torch.cat([p.flatten() for p in get_agent_params(params_dict, k, n_model_params)])
#             w_l = torch.cat([p.flatten() for p in get_agent_params(params_dict, l, n_model_params)])
#             reg_loss += 0.5*theta[k, l] * (w_k - w_l).pow(2).sum()

#     total_loss = data_loss/K + reg_loss/K
    
#     return total_loss



# def upper_loss_multitask(model, params_dict, data, stoch=False, seed=None):
#     """
#     Decentralised lower loss: g(w, θ) = (1/K) Σ_k g_k(w_k, θ)
    
#     data: list of K dicts, each with 'features_train' and 'labels_train'
#     params_dict: expects w_0...w_{K-1} as per-agent weights, t_0 as theta
#     """
#     K = len(data)
# #     weights = torch.sigmoid(params_dict['t_0'].squeeze())
#     param_names = [name for name, _ in model.named_parameters()]
#     n_model_params = len(param_names)

#     total_loss = torch.tensor(0.0)

#     for k, agent_data in enumerate(data):
#         features_train = agent_data[f'features_val']
#         labels_train   = agent_data[f'labels_val']

#         # Each agent has its own slice of w parameters: w_{k*n_model_params + i}
#         model_call_params = {
#             param_names[i]: params_dict[f'w_{k * n_model_params + i}']
#             for i in range(n_model_params)
#         }

#         if stoch:
#             torch.manual_seed(seed + k)  # different seed per agent
#             n = torch.randint(0, len(features_train), (1,))
#             features_train = features_train[[n], :]
#             labels_train   = labels_train[[n], :]
#         #     agent_weights  = weights[n]
#         # else:
#         #     agent_weights = weights

#         pred = functional_call(model, model_call_params, features_train)
#         per_sample_mse = F.mse_loss(pred, labels_train, reduction='none').mean(dim=-1)
#         total_loss     = total_loss + 0.5 * (per_sample_mse).mean()

#     return total_loss / K
