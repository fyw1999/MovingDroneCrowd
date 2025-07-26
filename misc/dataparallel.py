import torch
import torch.nn as nn

class MyDataParallel(nn.DataParallel):
    def __init__(self, module, device_ids=None, output_device=None):
        super(MyDataParallel, self).__init__(module, device_ids, output_device)
    
    def scatter(self, inputs, kwargs, device_ids):
        batch_size = inputs[0].size(0)
        assert batch_size % 2 == 0, "Batch size must be even."
        
        new_inputs = []
        
        for i in range(0, batch_size, 2):
            new_inputs.append(torch.stack([inputs[0][i], inputs[0][i+1]]))

        inputs_scattered = super().scatter(new_inputs, kwargs, device_ids)
        return inputs_scattered