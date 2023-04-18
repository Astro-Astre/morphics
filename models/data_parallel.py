from torch.nn.parallel import DataParallel  # Import DataParallel from PyTorch's nn.parallel module
import torch  # Import PyTorch library
from torch.nn.parallel._functions import Scatter  # Import Scatter function from PyTorch's nn.parallel._functions module
from torch.nn.parallel.parallel_apply import \
    parallel_apply  # Import parallel_apply function from PyTorch's nn.parallel.parallel_apply module


# Define scatter function that scatters the input tensor to multiple devices
def scatter(inputs, target_gpus, chunk_sizes, dim=0):
    # Define scatter_map function that maps the scatter operation over the input object
    def scatter_map(obj):
        if isinstance(obj, torch.Tensor):  # Check if the object is a tensor
            try:
                # Apply scatter operation on tensor with target GPUs, chunk sizes, and specified dimension
                return Scatter.apply(target_gpus, chunk_sizes, dim, obj)
            except:  # Catch any exception
                # Print debugging information
                print('obj', obj.size())
                print('dim', dim)
                print('chunk_sizes', chunk_sizes)
                quit()  # Exit the program
        # If the object is a tuple with at least one element, apply scatter_map recursively to each element
        if isinstance(obj, tuple) and len(obj) > 0:
            return list(zip(*map(scatter_map, obj)))
        # If the object is a list with at least one element, apply scatter_map recursively to each element
        if isinstance(obj, list) and len(obj) > 0:
            return list(map(list, zip(*map(scatter_map, obj))))
        # If the object is a dictionary with at least one key-value pair, apply scatter_map recursively to each item
        if isinstance(obj, dict) and len(obj) > 0:
            return list(map(type(obj), zip(*map(scatter_map, obj.items()))))
        return [obj for targets in target_gpus]  # Return a copy of the object for each target GPU

    try:
        return scatter_map(inputs)  # Apply scatter_map to the input object
    finally:
        scatter_map = None  # Clear scatter_map to free memory


# Define scatter_kwargs function to scatter both input tensors and keyword arguments
def scatter_kwargs(inputs, kwargs, target_gpus, chunk_sizes, dim=0):
    inputs = scatter(inputs, target_gpus, chunk_sizes, dim) if inputs else []  # Scatter input tensors if any
    kwargs = scatter(kwargs, target_gpus, chunk_sizes, dim) if kwargs else []  # Scatter keyword arguments if any
    if len(inputs) < len(kwargs):  # Check if the number of inputs is less than the number of keyword arguments
        inputs.extend([() for _ in range(len(kwargs) - len(inputs))])  # Extend inputs with empty tuples
    elif len(kwargs) < len(inputs):  # Check if the number of keyword arguments is less than the number of inputs
        kwargs.extend(
            [{} for _ in range(len(inputs) - len(kwargs))])  # Extend keyword arguments with empty dictionaries
    inputs = tuple(inputs)  # Convert inputs to a tuple
    kwargs = tuple(kwargs)  # Convert keyword arguments to a tuple
    return inputs, kwargs  # Return the scattered inputs and keyword arguments


# Define BalancedDataParallel class that extends DataParallel to support balanced data distribution among multiple GPUs
class BalancedDataParallel(DataParallel):
    # Initialize the class with GPU 0 batch size, and pass other arguments and keyword arguments to the parent class
    def __init__(self, gpu0_bsz, *args, **kwargs):
        self.gpu0_bsz = gpu0_bsz  # Store GPU 0 batch size as an instance variable
        super().__init__(*args,
                         **kwargs)  # Call the initialization method of the parent class with the remaining args and kwargs

    # Define forward method to handle the forward pass of the input through the model
    def forward(self, *inputs, **kwargs):
        if not self.device_ids:  # Check if there are no device IDs
            return self.module(*inputs, **kwargs)  # Perform the forward pass on the module without data parallelism
        if self.gpu0_bsz == 0:  # Check if GPU 0 batch size is set to 0
            device_ids = self.device_ids[1:]  # Ignore the first device ID (GPU 0) and use the remaining device IDs
        else:
            device_ids = self.device_ids  # Use all the device IDs
        inputs, kwargs = self.scatter(inputs, kwargs,
                                      device_ids)  # Scatter the inputs and kwargs across the target devices
        if len(self.device_ids) == 1:  # Check if there is only one device ID
            return self.module(*inputs[0], **kwargs[
                0])  # Perform the forward pass on the module with the first set of inputs and kwargs
        replicas = self.replicate(self.module, self.device_ids)  # Create replicas of the module on each target device
        if self.gpu0_bsz == 0:  # Check if GPU 0 batch size is set to 0
            replicas = replicas[1:]  # Ignore the first replica (GPU 0) and use the remaining replicas
        outputs = self.parallel_apply(replicas, device_ids, inputs,
                                      kwargs)  # Apply the forward pass in parallel using the replicas, device IDs, inputs, and kwargs
        return self.gather(outputs, self.output_device)  # Gather the outputs from all devices to the output device

    # Define parallel_apply method that applies the forward pass in parallel using the replicas, device IDs, inputs, and kwargs
    def parallel_apply(self, replicas, device_ids, inputs, kwargs):
        return parallel_apply(replicas, inputs, kwargs,
                              device_ids)  # Call the parallel_apply function with the given arguments

        # Define scatter method that scatters the inputs and kwargs across the target devices

    def scatter(self, inputs, kwargs, device_ids):
        bsz = inputs[0].size(self.dim)  # Get the batch size from the first input tensor
        num_dev = len(self.device_ids)  # Get the number of devices
        gpu0_bsz = self.gpu0_bsz  # Get GPU 0 batch size
        bsz_unit = (bsz - gpu0_bsz) // (num_dev - 1)  # Calculate the batch size unit for the remaining devices
        if gpu0_bsz < bsz_unit:  # Check if GPU 0 batch size is less than the batch size unit
            chunk_sizes = [gpu0_bsz] + [bsz_unit] * (num_dev - 1)  # Create a list of chunk sizes for each device
            delta = bsz - sum(
                chunk_sizes)  # Calculate the difference between the original batch size and the sum of chunk sizes
            for i in range(delta):  # Loop through the range of delta
                chunk_sizes[i + 1] += 1  # Increment the chunk size of the next device to balance the remaining delta
            if gpu0_bsz == 0:  # Check if GPU 0 batch size is set to 0
                chunk_sizes = chunk_sizes[1:]  # Ignore the first chunk size (GPU 0) and use the remaining chunk sizes
        else:  # If GPU 0 batch size is not less than the batch size unit
            return super().scatter(inputs, kwargs,
                                   device_ids)  # Call the scatter method of the parent class with the given inputs, kwargs, and device IDs
        # Call the scatter_kwargs function with the inputs, kwargs, device IDs, chunk sizes, and the scattering
        # dimension
        return scatter_kwargs(inputs, kwargs, device_ids, chunk_sizes, dim=self.dim)
