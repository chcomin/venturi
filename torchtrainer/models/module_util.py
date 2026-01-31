"""Utility functions and classes for working with Pytorch modules"""

from collections import OrderedDict

import torch
from torch import nn

bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)

def split_modules(model, modules_to_split):
    """Split `model` layers into different groups. Useful for freezing part of the model
    or using different learning rates.
    """

    module_groups = [[]]
    for module in model.modules():
        if module in modules_to_split:
            module_groups.append([])
        module_groups[-1].append(module)
    return module_groups

def define_opt_params(module_groups, lr=None, wd=None, debug=False):
    """Define distinct learning rate and weight decay for parameters belonging
    to groupd modules in `module_groups`.
    """

    num_groups = len(module_groups)
    if isinstance(lr, int): 
        lr = [lr]*num_groups
    if isinstance(wd, int): 
        wd = [wd]*num_groups

    opt_params = []
    for idx, group in enumerate(module_groups):
        group_params = {"params":[]}
        if lr is not None: 
            group_params["lr"] = lr[idx]
        if wd is not None: 
            group_params["wd"] = wd[idx]
        for module in group:
            pars = module.parameters(recurse=False)
            if debug: 
                print(module.__class__)
            pars = list(filter(lambda p: p.requires_grad, pars))
            if len(pars)>0:
                group_params["params"] += pars
                if debug:
                    for p in pars:
                        print(p.shape)
        opt_params.append(group_params)
    return opt_params

def groups_requires_grad(module_groups, req_grad=True, keep_bn=False):
    """Set requires_grad to `req_grad` for all parameters in `module_groups`.
    If `keep_bn` is True, batchnorm layers are not changed.
    """

    for group in module_groups:
        for module in group:
            for p in module.parameters(recurse=False):
                if not keep_bn or not isinstance(module, bn_types): 
                    p.requires_grad=req_grad

def freeze_to(module_groups, group_idx=-1, keep_bn=False):
    """Freeze model groups up to the group with index `group_idx`. If `group_idx` is None,
    freezes the entire model. If `keep_bn` is True, batchnorm layers are not changed.
    """

    slice_freeze = slice(0, group_idx)
    if group_idx is not None:
        slice_unfreeze = slice(group_idx, None)

    groups_requires_grad(module_groups[slice_freeze], False, keep_bn)

    if group_idx is not None:
        groups_requires_grad(module_groups[slice_unfreeze], True)

def unfreeze(module_groups):
    """Unfreezes the entire model."""

    groups_requires_grad(module_groups, True)

def get_submodule(model, module):
    """Return a module inside `model`. Module should be a string of the form
    'layer_name.sublayer_name'
    """

    modules_names = module.split(".")
    curr_module = model
    for name in modules_names:
        curr_module = curr_module._modules[name]
    requested_module = curr_module

    return requested_module
    
def get_submodule_str(model, module):
    """Return a string representation of `module` in the form 'layer_name.sublayer_name...'"""

    for name, curr_module in model.named_modules():
        if curr_module is module:
            module_name = name
            break

    return module_name

def _iterate_modules(father_name, module, module_name, adj_list, modules_dict):
    
    modules_dict[module_name] = module
    for child_module_name, child_module in module.named_children():
        full_child_name = f"{module_name}.{child_module_name}"
        if module_name in adj_list:
            adj_list[module_name].append(full_child_name)
        else:
            adj_list[module_name] = [full_child_name]        
        _iterate_modules(module_name, child_module, full_child_name, adj_list, modules_dict)

def _modules_graph(model):
    """Get hiearchy of modules inside model as an adjacency list"""
    
    adj_list = {}
    modules_dict = {}
    _iterate_modules(None, model, model.__class__.__name__, adj_list, modules_dict)
    
    return adj_list, modules_dict

def model_up_to(model, module):
    """Return a new model with all layers in model up to layer `module`."""
    
    split_module_str = get_submodule_str(model, module)
    split_modules_names = split_module_str.split(".")
    module = model
    splitted_model = []
    name_prefix = ""
    for idx, split_module_name in enumerate(split_modules_names):
        for child_module_name, child_module in module.named_children():
            if child_module_name==split_module_name:
                if idx==len(split_modules_names)-1:
                    # If at last module
                    full_name = f"{name_prefix}{child_module_name}"
                    splitted_model.append((full_name, child_module))
                module = child_module
                name_prefix += split_module_name + "_"
                break
            else:
                full_name = f"{name_prefix}{child_module_name}"
                splitted_model.append((full_name, child_module))

    new_model = torch.nn.Sequential(OrderedDict(splitted_model))
    
    return new_model

