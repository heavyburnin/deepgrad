# CSV to .bin conversion (run once)
def convert_csv_to_bin(csv_path, bin_path):
    with open(csv_path, 'r') as f_csv, open(bin_path, 'wb') as f_bin:
        next(f_csv)  # Skip header
        for line in f_csv:
            values = list(map(float, line.strip().split(',')))
            label_idx = int(values[0])
            one_hot = [0.0] * 10
            one_hot[label_idx] = 1.0
            image = values[1:]  # 784 values
            sample = image + one_hot  # total: 784 + 10
            f_bin.write(struct.pack(f'{len(sample)}f', *sample))

def load_bin_dataset(bin_path, num_samples, sample_size):
    data = []
    with open(bin_path, 'rb') as f:
        for _ in range(num_samples):
            raw = f.read(sample_size * 4)
            sample = list(struct.unpack(f'{sample_size}f', raw))
            data.append(sample)
    return data