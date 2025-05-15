#include <iostream>

extern "C"
{
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
}

// Helper function to initialize NumPy
bool init_numpy()
{
    import_array1(false); // safer version for C++
    return true;
}

int main()
{
    Py_Initialize();
    if (!init_numpy())
    {
        std::cerr << "Failed to initialize NumPy!" << std::endl;
        return 1;
    }

    npy_intp dims[2] = {2, 3}; // 2x3 array
    PyObject *array = PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    if (!array)
    {
        std::cerr << "Failed to create NumPy array!" << std::endl;
        Py_Finalize();
        return 1;
    }

    // Fill the array
    double *data = static_cast<double *>(PyArray_DATA((PyArrayObject *)array));
    for (int i = 0; i < 6; ++i)
    {
        data[i] = i * 1.1;
    }

    // Print the array using Python's repr
    PyObject *repr = PyObject_Repr(array);
    const char *str = PyUnicode_AsUTF8(repr);
    std::cout << "NumPy Array:\n"
              << str << std::endl;

    // Clean up
    Py_DECREF(repr);
    Py_DECREF(array);
    Py_Finalize();
    return 0;
}
