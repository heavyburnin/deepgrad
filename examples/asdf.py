import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../python')))
from tensor import Tensor
from array import array

def to_tensor(data, shape):
    return Tensor(data, shape=shape)

def data_loader(path, batch_size):
    with open(path, "rb") as f:
        data = f.read()
    
    num_samples = len(data) // (28 * 28 + 1)
    for i in range(0, num_samples, batch_size):
        inputs = []
        labels = []
        for j in range(i, min(i + batch_size, num_samples)):
            offset = j * (28 * 28 + 1)
            label = int.from_bytes(data[offset:offset+1], 'little')
            pixels = [int.from_bytes(data[k:k+1], 'little') for k in range(offset + 1, offset + 1 + 28 * 28)]
            inputs.append([p / 255.0 for p in pixels])
            labels.append(label)
        
        actual_batch_size = len(inputs)
        x_tensor = to_tensor(sum(inputs, []), (actual_batch_size, 784))
        y_tensor = to_tensor(array('B', labels), (actual_batch_size,))
        yield x_tensor, y_tensor
        
def test_data_loader(path, batch_size):
    print(f"Testing data loader with file: {path} and batch size: {batch_size}")
    
    # Fetch a single batch of data
    data_gen = data_loader(path, batch_size)
    x_batch, y_batch = next(data_gen)
    
    # Verify input shape: it should be (batch_size, 784)
    assert x_batch.shape == (batch_size, 784), f"Expected x_batch shape: ({batch_size}, 784), got {x_batch.shape}"
    
    # Verify label shape: it should be (batch_size,)
    assert y_batch.shape == (batch_size,), f"Expected y_batch shape: ({batch_size},), got {y_batch.shape}"
    
    # Verify input data range: all pixel values should be between 0 and 1
    for pixel in x_batch.data:
        assert 0.0 <= pixel <= 1.0, f"Invalid pixel value: {pixel}, should be in the range [0, 1]"
    
    # Verify label values: labels should be integers between 0 and 9
    for label in y_batch.data:
        assert 0 <= label <= 9, f"Invalid label value: {label}, should be between 0 and 9"
    
    # Print out some sample data and labels for visual inspection
    print("Sample data (first batch):")
    print("Input data (first 10 pixels of first sample):", x_batch.data[:10])
    print("Label (first sample):", y_batch.data[0])

    # Test with more data by getting the next batch
    x_batch, y_batch = next(data_gen)
    print("Next batch loaded successfully:")
    print("Input data (first 10 pixels of first sample):", x_batch.data[:10])
    print("Label (first sample):", y_batch.data[0])

if __name__ == "__main__":
    # Replace this with the actual path to your MNIST bin file
    mnist_train_bin_path = "mnist_train.bin"  # Example path
    batch_size = 32
    
    if os.path.exists(mnist_train_bin_path):
        test_data_loader(mnist_train_bin_path, batch_size)
    else:
        print(f"Error: The file {mnist_train_bin_path} does not exist.")