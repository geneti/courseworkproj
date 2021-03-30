#ifdef OLD_HEADER_FILENAME
#include <iostream.h>
#else
#include <iostream>
#endif
#include <string>
#include <new>
#include "hdf5.h"

#include "H5Cpp.h"

#ifndef H5_NO_NAMESPACE
using namespace H5;
#endif

const H5std_string FILE_NAME("array10_2.h5");
const H5std_string DATASET_NAME("test");
#define NUM 100
#define RANK 1

int main()
{
    H5File file(FILE_NAME, H5F_ACC_RDONLY);
    DataSet dataset = file.openDataSet(DATASET_NAME);
    DataSpace filespace = dataset.getSpace();

    int rank = filespace.getSimpleExtentNdims(); //Get number of dimensions in the file dataspace
    hsize_t dims[1];                             // dataset dimensions
    rank = filespace.getSimpleExtentDims(dims);
    DataSpace mspace1(RANK, dims); //Define the memory space to read dataset.

    int buffer[NUM];
    dataset.read(buffer, PredType::NATIVE_INT, mspace1, filespace );
    for (int i=0;i<NUM;i++){
        std::cout<<buffer[i]<<",";
    }
    std::cout<<std::endl;
    return 0;
}