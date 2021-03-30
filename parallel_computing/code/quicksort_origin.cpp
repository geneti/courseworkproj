#ifdef OLD_HEADER_FILENAME
#include <iostream.h>
#else
#include <iostream>
#endif
#include <string>
#include <stdio.h>
#include <algorithm>
#include <time.h>
#include <vector>
#include <mpi.h>
#include <math.h>
#include "verify.h"
#include <new>
#include "hdf5.h"
#include "get_walltime.c"

#include "H5Cpp.h"

#ifndef H5_NO_NAMESPACE
using namespace H5;
#endif

const H5std_string FILE_NAME("multi.h5");
const H5std_string DATASET_NAME("test");

// Important Notice: This variable must be the same with the dataset array length
#define NUM 100000

#define RANK 1

// swap 2 values in a list given the pointer
static inline
void swap(int *arr, int i, int j)
{
    int t = arr[i];
    arr[i] = arr[j];
    arr[j] = t;
}

/*
Function: separate the array into less than pivot, equal to pivot and larger than pivot
Return: correct index of the pivot
*/
int partition(int arr[], int left, int right)
{
    int i = (left - 1); //index
    int pivot = arr[right];
    for (int j = left; j <= right - 1; j++)
    {
        if (arr[j] <= pivot)
        {
            i++;
            swap(arr, i,j); // swap two values smaller and larger than pivot respectively
        }
    }
    swap(arr, i+1, right); //move the pivot with the correct position
    return (i + 1);
}

/*
Function: sequential quicksort function
Return: a sorted array
*/
void quickSort(int arr[], int low, int high)
{
    if (low < high)
    {
        int pos = partition(arr, low, high);
        quickSort(arr, low, pos - 1);  //recursive run the left part
        quickSort(arr, pos + 1, high); //recursive run the right part
    }
}

int main(int argc, char *argv[])
{
    FILE *dst = NULL;
    if (argc != 2)
    {
        fprintf(stderr, "Set the output file to execute\n", argv[0]);
        exit(1);
    }

    int *data;
    data = (int *)malloc(NUM * sizeof(int));

    //** IO part: read the test array from the h5 dataset
    H5File file(FILE_NAME, H5F_ACC_RDONLY);
    DataSet dataset = file.openDataSet(DATASET_NAME);
    DataSpace filespace = dataset.getSpace();

    int rank = filespace.getSimpleExtentNdims(); //Get number of dimensions in the file dataspace
    hsize_t dims[1];                             // dataset dimensions
    rank = filespace.getSimpleExtentDims(dims);
    printf("The dataspace length is:%d\n", dims[0]);
    DataSpace mspace1(RANK, dims); //Define the memory space to read dataset.

    int data_array[NUM]; // load data into buffer at the root processor
    dataset.read(data_array, PredType::NATIVE_INT, mspace1, filespace);
    printf("Read H5 successfully\n");
    for (int i = 0; i < NUM; i++)
    {
        data[i] = data_array[i];
    }

    // compute the time
    double *start_time = new double();
    double *end_time = new double();
    get_walltime(start_time);
    quickSort(data, 0, NUM - 1);
    get_walltime(end_time);
    dst = fopen(argv[1], "w");
    for (int i = 0; i < NUM; i++)
        fprintf(dst, "%d\n", data[i]);
    fclose(dst);

    check(data, NUM);

    printf("Run time of sequential algorithm is  %f\n", *end_time - *start_time);
    return 0;
}
