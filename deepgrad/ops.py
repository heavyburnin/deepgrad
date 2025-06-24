# ops.py

BINARY_OPS = {
    "add": ("tensor_add", "tensor_add_grad"),
    "sub": ("tensor_sub", "tensor_sub_grad"),
    "mul": ("tensor_mul", "tensor_mul_grad"),
    "div": ("tensor_div", "tensor_div_grad"),
    "pow": ("tensor_pow", "tensor_pow_grad"),
    # Add more ops here as needed
}

def get_op_names(op_name):
    """
    Returns (forward_fn, backward_fn) tuple for the given op.
    """
    if op_name not in BINARY_OPS:
        raise ValueError(f"Unsupported op: {op_name}")
    return BINARY_OPS[op_name]

def register_op(name, forward_fn, backward_fn):
    """
    Dynamically add a new op and its gradients.
    """
    BINARY_OPS[name] = (forward_fn, backward_fn)