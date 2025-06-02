import cffi
import os

ffibuilder = cffi.FFI()

ffibuilder.cdef("""
typedef struct {
    float* data;
    size_t* shape;
    size_t* strides;
    size_t ndim;
} Tensor;

void tensor_calc_strides(const size_t* shape, size_t ndim, size_t* strides_out);
Tensor* tensor_create(const size_t* shape, size_t ndim);
void tensor_free(Tensor* t);

size_t tensor_broadcast_offset(const Tensor* t, const size_t* idx);

int tensor_broadcast_shape(const size_t* shape_a, size_t ndim_a,
                           const size_t* shape_b, size_t ndim_b,
                           size_t* out_shape, size_t* out_ndim);

void tensor_add(const Tensor* a, const Tensor* b, Tensor* out);
void tensor_sub(const Tensor* a, const Tensor* b, Tensor* out);
void tensor_mul(const Tensor* a, const Tensor* b, Tensor* out);
void tensor_div(const Tensor* a, const Tensor* b, Tensor* out);

void tensor_relu(const Tensor* a, Tensor* out);
void tensor_exp(const Tensor* a, Tensor* out);
void tensor_neg(const Tensor* a, Tensor* out);
void tensor_sqrt(const Tensor* a, Tensor* out);
void tensor_log(const Tensor* a, Tensor* out);

void tensor_batched_gemm(const Tensor* A, const Tensor* B, Tensor* C);

float tensor_sum(const Tensor* a);

void tensor_softmax(const Tensor* input, Tensor* output);

float tensor_cross_entropy_loss(const Tensor* probs, const size_t* labels);

void tensor_copy(const Tensor* src, Tensor* dest);
void tensor_fill(Tensor* t, float value);
size_t tensor_argmax(const Tensor* t);
""")

lib_path = os.path.join(os.path.dirname(__file__), "../build/libsimd_tensor_backend.so")

C = ffibuilder.dlopen(lib_path)

class Tensor:
    def __init__(self, shape, data=None):
        self.ndim = len(shape)
        self.shape = tuple(shape)
        self._shape_c = ffi.new("size_t[]", self.shape)
        self._tensor = C.tensor_create(self._shape_c, self.ndim)
        if self._tensor == ffi.NULL:
            raise MemoryError("Could not allocate tensor")
        if data is not None:
            # Copy data
            for i in range(len(data)):
                self._tensor.data[i] = data[i]

    def __del__(self):
        if hasattr(self, "_tensor") and self._tensor != ffi.NULL:
            C.tensor_free(self._tensor)
            self._tensor = ffi.NULL

    @property
    def data(self):
        return ffi.buffer(self._tensor.data, self.size * 4)

    @property
    def size(self):
        s = 1
        for dim in self.shape:
            s *= dim
        return s

    def fill(self, value):
        C.tensor_fill(self._tensor, value)

    def relu(self, out):
        C.tensor_relu(self._tensor, out._tensor)

    def add(self, b, out):
        C.tensor_add(self._tensor, b._tensor, out._tensor)

    def softmax(self, out):
            C.tensor_softmax(self._tensor, out._tensor)

    def cross(self, b, out):
        C.cross_entropy(self._tensor, b._tensor, out._tensor)

ffi = cffi.FFI()
