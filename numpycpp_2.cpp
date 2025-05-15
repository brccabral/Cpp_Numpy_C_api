#include <iostream>

extern "C"
{
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
}

bool init_numpy()
{
    import_array1(false);
    return true;
}

PyArrayObject *create_array(double values[], npy_intp dims[2])
{
    PyArrayObject *arr = (PyArrayObject *)PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    if (!arr)
        return nullptr;

    double *data = static_cast<double *>(PyArray_DATA(arr));
    for (int i = 0; i < dims[0] * dims[1]; ++i)
    {
        data[i] = values[i];
    }
    return arr;
}

int main()
{
    Py_Initialize();
    if (!init_numpy())
    {
        std::cerr << "NumPy init failed\n";
        return 1;
    }

    // Create two arrays
    npy_intp dims[2] = {2, 3};
    double valuesA[] = {1, 2, 3, 4, 5, 6};
    double valuesB[] = {10, 20, 30, 40, 50, 60};

    PyArrayObject *arrayA = create_array(valuesA, dims);
    PyArrayObject *arrayB = create_array(valuesB, dims);

    if (!arrayA || !arrayB)
    {
        std::cerr << "Failed to create input arrays\n";
        Py_Finalize();
        return 1;
    }

    // Perform element-wise multiplication
    PyObject *result = PyNumber_Multiply((PyObject *)arrayA, (PyObject *)arrayB);
    if (!result)
    {
        PyErr_Print();
        std::cerr << "Element-wise multiplication failed\n";
        Py_DECREF(arrayA);
        Py_DECREF(arrayB);
        Py_Finalize();
        return 1;
    }

    // Print all arrays
    auto print_array = [](const char *label, PyObject *array)
    {
        PyObject *repr = PyObject_Repr(array);
        const char *str = PyUnicode_AsUTF8(repr);
        std::cout << label << " = " << str << "\n";
        Py_DECREF(repr);
    };

    print_array("A", (PyObject *)arrayA);
    print_array("B", (PyObject *)arrayB);
    print_array("A * B", result);

    // Cleanup
    Py_DECREF(arrayA);
    Py_DECREF(arrayB);
    Py_DECREF(result);
    Py_Finalize();
    return 0;
}
