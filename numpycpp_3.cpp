#include <iostream>

extern "C" {
    #define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
    #include <Python.h>
    #include <numpy/arrayobject.h>
}

bool init_numpy() {
    import_array1(false);
    return true;
}

PyArrayObject* create_array(double values[], npy_intp dims[2]) {
    PyArrayObject* arr = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    if (!arr) return nullptr;

    double* data = static_cast<double*>(PyArray_DATA(arr));
    for (int i = 0; i < dims[0] * dims[1]; ++i) {
        data[i] = values[i];
    }
    return arr;
}

int main() {
    Py_Initialize();
    if (!init_numpy()) {
        std::cerr << "NumPy init failed\n";
        return 1;
    }

    // Array A: shape (2, 3)
    npy_intp dimsA[2] = {2, 3};
    double valuesA[] = {
        1, 2, 3,
        4, 5, 6
    };
    PyArrayObject* arrayA = create_array(valuesA, dimsA);

    // Array B: shape (3, 2)
    npy_intp dimsB[2] = {3, 2};
    double valuesB[] = {
        10, 20,
        30, 40,
        50, 60
    };
    PyArrayObject* arrayB = create_array(valuesB, dimsB);

    if (!arrayA || !arrayB) {
        std::cerr << "Failed to create input arrays\n";
        Py_Finalize();
        return 1;
    }

    // Import numpy module
    PyObject* numpy_mod = PyImport_ImportModule("numpy");
    if (!numpy_mod) {
        PyErr_Print();
        std::cerr << "Failed to import numpy module\n";
        Py_Finalize();
        return 1;
    }

    // Call numpy.dot(A, B)
    PyObject* result = PyObject_CallMethod(numpy_mod, "dot", "OO", arrayA, arrayB);
    if (!result) {
        PyErr_Print();
        std::cerr << "Matrix multiplication failed\n";
        Py_DECREF(numpy_mod);
        Py_DECREF(arrayA);
        Py_DECREF(arrayB);
        Py_Finalize();
        return 1;
    }

    // Print results
    auto print_array = [](const char* label, PyObject* array) {
        PyObject* repr = PyObject_Repr(array);
        const char* str = PyUnicode_AsUTF8(repr);
        std::cout << label << " = " << str << "\n";
        Py_DECREF(repr);
    };

    print_array("A", (PyObject*)arrayA);
    print_array("B", (PyObject*)arrayB);
    print_array("np.dot(A, B)", result);

    // Cleanup
    Py_DECREF(result);
    Py_DECREF(numpy_mod);
    Py_DECREF(arrayA);
    Py_DECREF(arrayB);
    Py_Finalize();
    return 0;
}
