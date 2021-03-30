#include "H5Cpp.h"
using namespace H5;

#define FILE "array1e5_16.h5"
#define NUM 100000*16

int main()
{
  hsize_t dims[2];
  herr_t status;
  H5File file(FILE, H5F_ACC_TRUNC);
  dims[0] = NUM;

  double *data = new double[dims[0]];

  for (size_t i = 0; i < dims[0]; ++i)
  {
    std::srand(i); // set the random seed for each index
    data[i] = std::rand() % 1000;
  }

  DataSpace dataspace = DataSpace(1, dims);
  DataSet dataset(file.createDataSet("test", PredType::IEEE_F64LE, dataspace));
  dataset.write(data, PredType::IEEE_F64LE);
  dataset.close();
  dataspace.close();
  file.close();

  return 0;
}