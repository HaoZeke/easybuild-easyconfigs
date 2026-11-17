#include <Python.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/autograd/variable.h>

// Read the existing edge without taking another shared ownership reference.
extern "C" long long torch_grad_fn_owner_count(PyObject* object) {
    if (!THPVariable_Check(object)) {
        return -1;
    }
    const auto& tensor = THPVariable_Unpack(object);
    const auto* metadata = torch::autograd::impl::get_autograd_meta(tensor);
    if (metadata == nullptr) {
        return 0;
    }
    return static_cast<long long>(metadata->grad_fn_.use_count());
}
