import torch



def get_model_params_groups(model):
    all_params = dict.fromkeys(model.parameters())
    
    wd_params = dict.fromkeys([])
    for module in model.modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
            wd_params[module.weight] = None

    no_wd_params = dict.fromkeys(x for x in all_params if x not in wd_params)

    return list(wd_params.keys()), list(no_wd_params.keys())



def gather_pairs(pairs, global_rank, world_size):
    batch_size = pairs.shape[0]//2

    pairs = pairs.view(2, batch_size, -1)

    with torch.no_grad():
        all_pairs = [torch.zeros_like(pairs) for _ in range(world_size)]
        torch.distributed.all_gather(all_pairs, pairs)
    
    all_pairs[global_rank] = pairs
    all_pairs = torch.cat(all_pairs, dim=1)
    all_pairs = all_pairs.view(2*batch_size*world_size, -1)

    return all_pairs



def cos(z1, z2):
    return (torch.nn.functional.normalize(z1)*torch.nn.functional.normalize(z2)).sum(1)




class TrackedValue():
    def __init__(self, device):
        self.device = device
        self.reset()

    def reset(self):
        self.value = None
        self.steps = None
        self.sum = torch.tensor(0.).to(self.device)
        self.count = torch.tensor(0.).to(self.device)
        self.average = torch.tensor(0.).to(self.device)
    
    def add(self, value, steps=1):
        self.value = value.to(self.device) if torch.is_tensor(value) else torch.tensor(value).to(self.device)
        self.steps = steps.to(self.device) if torch.is_tensor(steps) else torch.tensor(steps).to(self.device)

        self.sum += self.value
        self.count += self.steps

        self.average = self.sum/self.count

    def reduce(self):
        torch.distributed.all_reduce(self.sum, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(self.count, op=torch.distributed.ReduceOp.SUM)

        self.average = self.sum/self.count



class Tracker():
    def __init__(self, device):
        self.device = device
        self.tracked_values = {}
    
    def reset(self, name):
        self.tracked_values[name] = TrackedValue(self.device)

    def get(self, name):
        if name not in self.tracked_values.keys():
            self.reset(name)
        
        return self.tracked_values[name]

    def add(self, name, value, steps=1):
        if name not in self.tracked_values.keys():
            self.reset(name)
        
        self.tracked_values[name].add(value, steps)
