import torch
import time

# split value
tensor_size = 100

try:
    # Set up PyTorch to use the GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    if device.type == 'cuda':

        # Set up PyTorch to use the GPU
        torch.backends.cudnn.benchmark = True
        print(f'Successfully set up GPU {torch.cuda.current_device()} to use benchmark mode.')

        # Create a simple computation to continuously run
        while True:

            x = torch.randn(tensor_size, tensor_size, device=device)
            _ = torch.matmul(x, x)

    else:
        print('CUDA is not available. Exiting...')

except Exception as e:
    print(f'An error occurred: {e}')