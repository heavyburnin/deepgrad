def convert_mnist_csv_to_bin(csv_path, bin_path):
    with open(csv_path, 'r') as csv_file, open(bin_path, 'wb') as bin_file:
        next(csv_file)  # skip header
        for line in csv_file:
            parts = line.strip().split(',')
            label = int(parts[0])
            pixels = [int(p) for p in parts[1:]]
            bin_file.write(bytes([label]))
            bin_file.write(bytes(pixels))

def load_mnist_bin(path):
    # Not used here but available if you need full load
    data = []
    with open(path, "rb") as f:
        while True:
            try:
                label = int.from_bytes(f.read(1), "little")
                pixels = list(f.read(28 * 28))
                if len(pixels) < 784:
                    break
                data.append((pixels, label))
            except:
                break
    return data
