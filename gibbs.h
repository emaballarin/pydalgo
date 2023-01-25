/* .... C vector utility functions ..................*/
PyArrayObject *pyvector(PyObject *objin);
double *pyvector_to_Carrayptrs(PyArrayObject *arrayin);
long int *pyvector_to_Carrayiptrs(PyArrayObject *arrayin);

// .... Python callable Vector functions ..................
static PyObject *GibbsSampling(PyObject *self, PyObject *args);
