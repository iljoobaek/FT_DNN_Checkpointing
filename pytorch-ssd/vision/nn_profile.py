# Modified from source provided in issue on pytorch source
# https://github.com/pytorch/pytorch/issues/3749
from __future__ import division

from collections import defaultdict
from functools import lru_cache, reduce
import math
import operator

import torch


def profile(module, display_cpu=True, display_gpu=True):
    assert issubclass(module, torch.nn.Module)
    monkey_patch_init(module)
    return module

def monkey_patch_init(_class):
    old_init = _class.__init__
    def new_init(self, *args, **kwargs):
        old_init(self, *args, **kwargs)
        self.profiler = Profiler(self)
        _class.__str__ = self.profiler.__str__
    _class.__init__ = new_init

class Profiler(object):
    def __init__(self, module, enabled=True):
        """
        An operation is a graph node that performs computation on tensors.
        """
        self._module = module
        self._events = {
            'forward': defaultdict(Event),
            'backward': defaultdict(Event)}
        self._operations = {} # Will hold all modules
        self._enable = enabled 

        #consume generator
        list(map(self._hook_operation, operations(self._module)))

    def set_enable(self, enabled):
        self._enable = enabled
        return self._enable

    def _hook_operation(self, op):
        def wrapper_call(op, *input, **kwargs):
            # Wrapper function to "__call__", with time counter in it.
            if not self._enable:
                return self._operations[op.__class__](op, *input, **kwargs)

            with torch.autograd.profiler.profile(use_cuda=True) as prof:
                result = self._operations[op.__class__](op, *input, **kwargs)

            self._events['forward'][op] += Event(
                cpu_time=int(prof.total_average().cpu_time),
                gpu_time=int(prof.total_average().cuda_time),
                weights_gpu_mem=self.get_gpu_mem(op),
                parameters=self.get_nelements(op),
                input_gpu_mem=sum_tensor_memory(input),
                input_params=count_elements(input),
                hits=1)
            
            def backward_pre_hook(*args):
                if not self._enable:
                    return
                self._events['backward'][op].append(time.time())
            #result.grad_fn.register_pre_hook(backward_pre_hook);
            return result

        # monky patch "__call__" with "wrapper_call"  for this operation`
        if op.__class__ not in self._operations:
            self._operations[op.__class__] = op.__class__.__call__
            op.__class__.__call__ = wrapper_call

        #def backward_post_hook(*args):
        #    if not this_profiler.profiling_on:
        #        return
        #    # adds ending time
        #    backward = this_profiler.record['backward']
        #    backward[-1] = backward[-1] + (time.time(),) 
        #op.register_backward_hook(backward_post_hook)   

    @lru_cache(maxsize=None)
    def get_gpu_mem(self, module):
        print("Getting gpumem for module:", module)
        for t in module.parameters(recurse=False):
            print (t.element_size()*t.nelement())
        # Module should have constant GPU memory footprint 
        return sum(map(lambda t: t.element_size()*t.nelement(), module.parameters(recurse=False)))
 
    @lru_cache(maxsize=None)
    def get_nelements(self, module):
        # Module should have constant number of parameters 
        print("Getting nelements for module:", module)
        for t in module.parameters(recurse=False):
            print (t.nelement())
        return sum(map(lambda t: t.nelement(), module.parameters(recurse=False)))


    @lru_cache(maxsize=None)
    def get_metrics(self, module):
        if module in self._events['forward']:
            #it's an operation
            return self._events['forward'][module]
        
        return reduce(operator.add, map(self.get_metrics, module._modules.values()))
    
    def __str__(self, module=None, indentation=0, pre_msg=''):
        tmpstr = ''
        if module is None:
            module = self._module
            tmpstr += Event.header()

        # this is an operation
        metrics = self.get_metrics(module).tostring()
    
        if module.__class__ in self._operations:
            return  tmpstr + metrics + indent(pre_msg + module.__repr__(), indentation) + '\n'
        
        name = module.__class__.__name__
        tmpstr += metrics + indent(pre_msg + name  + '(', indentation) + '\n'
        for key, sub_module in module._modules.items():
            tmpstr +=  self.__str__(sub_module, indentation+2, pre_msg='(' + key + '): ')
        tmpstr +=  indent(')',indentation+len(metrics)) + '\n'
        return tmpstr        

class Event(object):
    def __init__(self, cpu_time=0, gpu_time=0, weights_gpu_mem=0, input_gpu_mem=0,
                    parameters=0, input_params=0, hits=0):
        self.cpu_time = cpu_time
        self.gpu_time = gpu_time
        self.weights_gpu_mem = weights_gpu_mem
        self.input_gpu_mem = input_gpu_mem
        self.parameters = parameters
        self.input_params = input_params
        self.hits = hits
    
    @classmethod
    def header(cls):
        header = format_columns(['CPU Time','GPU Time','Weights GPU Mem', 'Parameters', 'Input GPU Mem', 'Input','Architecture'])
        return '\n'.join([header,'='*len(header),''])

    def tostring(self):
        return format_columns([
                format_time(self.cpu_time),
                format_time(self.gpu_time),
                format_mem(self.weights_gpu_mem), 
                format_count(self.parameters),
                format_mem(self.input_gpu_mem), 
                format_count(self.input_params)])
    
    def __add__(self, other):
        return Event(
            cpu_time=self.cpu_time + other.cpu_time,
            gpu_time=self.gpu_time + other.gpu_time,
            weights_gpu_mem=max(self.weights_gpu_mem, other.weights_gpu_mem), # Use high-water mark for mem
            parameters=max(self.parameters, other.parameters), # High-water
            input_gpu_mem=max(self.input_gpu_mem, other.input_gpu_mem), # High-water mark for mem
            input_params=max(self.input_params, other.input_params), # High-water
            hits=self.hits + other.hits)

    def __radd__(self, other):
        return self.__add__(other)

def format_columns(cols, width=10):
    assert isinstance(cols, list)
    return  ' ' + ' '.join(col.center(width,' ') for col in cols) + '  '

def format_time(time_in_ns):
    if not time_in_ns:
        return '-'

    human_powers = ['n','u','m','']
    power = int(math.log(time_in_ns, 10) // 3)
    return '{:.2f}{}s '.format(
            time_in_ns/1000.**power,
            human_powers[power])

def format_mem(mem_in_bytes):
    if not mem_in_bytes:
        return '-'

    human_powers = ['k','M', 'G', '']
    power = int(math.log(mem_in_bytes, 2) // 10)
    return '{:.2f}{}B '.format(
            mem_in_bytes/1024.**power,
            human_powers[power])

def format_count(n):
    if not n:
        return '-'

    human_powers = ['','k','m','g']
    power = int(math.log(n, 10) // 3)
    return '{:.2f}{} '.format(
            n/1000.**power,
            human_powers[power])

def operations(module):
    """
    Given a module recursively transverse it
    to find all atomic operations.

    Atomic operations are the nodes in the graph which
    perform computations on the tensors.
    """
    if not len(list(module.children())):
        # nn.Module who doesn't have sub nn.Module, hook it.
        yield module

    for name, sub_module in module.named_children():
        if (isinstance(sub_module, torch.nn.Container)
            or isinstance(sub_module, torch.nn.Sequential)
            or isinstance(sub_module, torch.nn.ModuleList)
            or isinstance(sub_module, torch.nn.Module)):
            # Recursively visit their decendants.
            for op in operations(sub_module): #python2 compatibility
                yield op

def indent(s, indent):
    return '\n'.join((indent* ' ') + line for line in s.split('\n'))

@lru_cache(maxsize=None)
def sum_tensor_memory(tensors):
    return sum([reduce(operator.mul, (t.element_size(), t.nelement())) for t in tensors])

@lru_cache(maxsize=None)
def count_elements(tensors):
    return sum([t.nelement() for t in tensors])
