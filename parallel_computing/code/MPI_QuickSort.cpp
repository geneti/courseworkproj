#ifdef OLD_HEADER_FILENAME
#include <iostream.h>
#else
#include <iostream>
#endif
#include <string>
#include <new>
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
#include <math.h>
#include "hdf5.h"
#include "H5Cpp.h"
#include "verify.h"
#include "get_walltime.c"
#ifndef H5_NO_NAMESPACE
using namespace H5;
#endif

const H5std_string DATASET_NAME("test");

// Important Notice: This variable must be the same with the dataset array length
#define NUM 1000000

#define RANK 1

// swap 2 values in a list given the pointer
static inline void swap(int *arr, int i, int j)
{
    int t = arr[i];
    arr[i] = arr[j];
    arr[j] = t;
}

int partition(int arr[], int left, int right)
{
    int i = (left - 1); //index
    int pivot = arr[right];
    for (int j = left; j <= right - 1; j++)
    {
        if (arr[j] <= pivot)
        {
            i++;
            swap(arr, i, j); // swap two values smaller and larger than pivot respectively
        }
    }
    swap(arr, i + 1, right); //move the pivot with the correct position
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

// merge 2 sorted sublists given the 2 pointers
int *merge(int *arr1, int length1, int *arr2, int length2)
{
    int *result = (int *)malloc((length1 + length2) * sizeof(int));
    int i = 0;
    int j = 0;
    for (int k = 0; k < length1 + length2; k++)
    {
        if (i >= length1)
        {
            result[k] = arr2[j];
            j++;
        }
        else if (j >= length2)
        {
            result[k] = arr1[i];
            i++;
        }
        else if (arr1[i] < arr2[j])
        { // indices in bounds as i < length1 && j < length2
            result[k] = arr1[i];
            i++;
        }
        else
        { // arr2[j] <= arr1[i]
            result[k] = arr2[j];
            j++;
        }
    }
    return result;
}

int main(int argc, char **argv)
{
    FILE *file = NULL;
    int chucksize, subchucksize, recvchucksize, step, pSize, pRank;
    MPI_Status status;
    double elapsed_time;
    int *data = NULL, *chunk = NULL, *other = NULL;

    if (argc != 4)
    {
        fprintf(stderr, "Correct Input format: mpiexec -n num <./*.out> <nums of procs: n> <*.h5> <*.txt>\n", argv[0]);
        exit(1);
    }
    int NUMS = pow(10, 5) * std::stoi(argv[1]);
    // int NUMS = pow(10, 5) * 1;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &pSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &pRank);

    if (pRank == 0)
    {

        chucksize = (NUMS % pSize != 0) ? NUMS / pSize + 1 : NUMS / pSize;
        data = (int *)malloc(pSize * chucksize * sizeof(int));

        //** IO part: read the test array from the h5 dataset
        H5std_string FILE_NAME(argv[2]);

        H5File file(FILE_NAME, H5F_ACC_RDONLY);
        DataSet dataset = file.openDataSet(DATASET_NAME);
        DataSpace filespace = dataset.getSpace();

        int rank = filespace.getSimpleExtentNdims(); //Get number of dimensions in the file dataspace
        hsize_t dims[1];                             // dataset dimensions
        rank = filespace.getSimpleExtentDims(dims);
        DataSpace mspace1(RANK, dims); //Define the memory space to read dataset.

        int data_array[NUMS]; // load data into buffer at the root processor
        dataset.read(data_array, PredType::NATIVE_INT, mspace1, filespace);

        for (int i = 0; i < NUMS; i++)
        {
            data[i] = data_array[i];
        }
        for (int i = NUMS; i < pSize * chucksize; i++)
        {
            data[i] = 0;
        }
        printf("read h5 successfully\n");
    }

    // start the timer
    MPI_Barrier(MPI_COMM_WORLD);
    double *start_time = new double();
    double *end_time = new double();
    get_walltime(start_time);

    // broadcast size
    MPI_Bcast(&NUMS, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // compute chunk size
    chucksize = (NUMS % pSize != 0) ? NUMS / pSize + 1 : NUMS / pSize;

    // scatter the array to processors
    chunk = (int *)malloc(chucksize * sizeof(int));
    MPI_Scatter(data, chucksize, MPI_INT, chunk, chucksize, MPI_INT, 0, MPI_COMM_WORLD);
    free(data);
    data = NULL;

    // compute size of own chunk and sort it
    subchucksize = (NUMS >= chucksize * (pRank + 1)) ? chucksize : NUMS - chucksize * pRank;
    quickSort(chunk, 0, subchucksize - 1);

    // merge the binary tree nodes
    for (step = 1; step < pSize; step = 2 * step)
    {
        if (pRank % (2 * step) != 0)
        {
            MPI_Send(chunk, subchucksize, MPI_INT, pRank - step, 0, MPI_COMM_WORLD);
            break;
        }
        if (pRank + step < pSize)
        {
            recvchucksize = (NUMS >= chucksize * (pRank + 2 * step)) ? chucksize * step : NUMS - chucksize * (pRank + step);
            other = (int *)malloc(recvchucksize * sizeof(int));
            MPI_Recv(other, recvchucksize, MPI_INT, pRank + step, 0, MPI_COMM_WORLD, &status);
            data = merge(chunk, subchucksize, other, recvchucksize);
            free(chunk);
            free(other);
            chunk = data;
            subchucksize = subchucksize + recvchucksize;
        }
    }

    get_walltime(end_time);
    elapsed_time = *end_time - *start_time;

    if (pRank == 0)
    {
        file = fopen(argv[3], "w");
        for (int i = 0; i < subchucksize; i++)
            fprintf(file, "%d\n", chunk[i]);
        fclose(file);
        printf("Quicksort %d ints on %d procs: %f secs\n", NUMS, pSize, elapsed_time);
    }

    MPI_Finalize();
    return 0;
}