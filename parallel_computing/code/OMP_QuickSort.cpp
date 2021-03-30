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
#include <cstring>
#include <mpi.h>
#include <math.h>
#include <numeric>
#include <math.h>
#include <new>
#include <vector>
#include <queue>
#include "verify.h"
#include "hdf5.h"
#include "get_walltime.c"
#include <omp.h>

#include "H5Cpp.h"

#ifndef H5_NO_NAMESPACE
using namespace H5;
#endif

const H5std_string DATASET_NAME("test");

// Important Notice: This variable must be the same with the dataset array length

#define RANK 1
#define MIN_PRARLLEL_PREFIXSUM_COUNT 10000

// define a struct with left and right bound
struct Node
{
    int left, right;
};

// swap 2 values in a list given the pointer
static inline void swap(int *arr, int i, int j)
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
            swap(arr, i, j); // swap two values smaller and larger than pivot respectively
        }
    }
    swap(arr, i + 1, right); //move the pivot with the correct position
    return (i + 1);
}

/*
Function: Sequential prefix sum
*/
void Sequential_PrefixSum(int *pInput, int *pOutput, int nLen)
{
    int i;
    pOutput[0] = pInput[0];
    for (i = 1; i < nLen; i++)
    {
        pOutput[i] = pInput[i] + pOutput[i - 1];
    }
}

/*
Function: parallel prefix sum algorithm
*/
void Parallel_PrefixSum(int *pInput, int *pOutput, int nLen)

{
    int i;
    int nCore = omp_get_num_procs();
    if (nCore < 4 || nLen < MIN_PRARLLEL_PREFIXSUM_COUNT)
    {
        Sequential_PrefixSum(pInput, pOutput, nLen);
        return;
    }
    int nStep = nLen / nCore;
    if (nStep * nCore < nLen)
    {
        nStep += 1;
    }

#pragma omp parallel for num_threads(nCore)
    for (i = 0; i < nCore; i++)
    {
        int k;
        int nStart = i * nStep;
        int nEnd = (i + 1) * nStep;
        if (nEnd > nLen)
        {
            nEnd = nLen;
        }
        pOutput[nStart] = pInput[nStart];
        for (k = nStart + 1; k < nEnd; k++)
        {
            pOutput[k] = pInput[k] + pOutput[k - 1];
        }
    }
    for (i = 2; i < nCore; i++)
    {
        pOutput[i * nStep - 1] += pOutput[(i - 1) * nStep - 1];
    }

    pOutput[nLen - 1] += pOutput[(nCore - 1) * nStep - 1];

#pragma omp parallel for num_threads(nCore - 1)

    for (i = 1; i < nCore; i++)
    {
        int k;
        int nStart = i * nStep;
        int nEnd = (i + 1) * nStep - 1;
        if (nEnd >= nLen)
        {
            nEnd = nLen - 1;
        }

        for (k = nStart; k < nEnd; k++)
        {
            pOutput[k] += pOutput[nStart - 1];
        }
    }
    return;
}

void computeparallelprefix(int *iplist, int *_pprefixsum, unsigned long size)
{
    int nthr, *z, *x = _pprefixsum;
#pragma omp parallel
    {
        int i;
#pragma omp single
        {
            nthr = omp_get_num_threads();
            z = (int *)malloc(sizeof(int) * nthr + 1);
            z[0] = 0;
        }
        int tid = omp_get_thread_num();
        int sum = 0;
#pragma omp for schedule(static)
        for (i = 0; i < size; i++)
        {
            sum += iplist[i];
            x[i] = sum;
        }
        z[tid + 1] = sum;
#pragma omp barrier
        int offset = 0;
        for (i = 0; i < (tid + 1); i++)
        {
            offset += z[i];
        }
#pragma omp for schedule(static)
        for (i = 0; i < size; i++)
        {
            x[i] += offset;
        }
    }
    free(z);
}

/*
Function: parallel version of the partition
*/

int ppartition(int arr[], int left, int right)
{
    int array_length = right - left + 1;
    // if (array_length < 2)
    // {
    //     return left;
    // }

    int pivot = arr[right];

    int *lt = new int[array_length]();
    std::memset(lt, 0, sizeof(int) * array_length);
    int *gt = new int[array_length]();
    std::memset(gt, 0, sizeof(int) * array_length);
    int *lt2 = new int[array_length]();
    std::memset(lt2, 0, sizeof(int) * array_length);
    int *gt2 = new int[array_length]();
    std::memset(gt2, 0, sizeof(int) * array_length);
    int *B = new int[array_length]();
    std::memset(B, 0, sizeof(int) * array_length);

#pragma omp for
    for (int i = 0; i < array_length; i++)
    {
        B[i] = arr[left + i];
    }

// calculate the lt, gt prefix
#pragma omp for
    for (int i = 0; i < array_length; i++)
    {
        if (arr[left + i] < pivot)
        {
            lt[i] = 1;
            gt[i] = 0;
        }
        else if (arr[left + i] > pivot)
        {
            lt[i] = 0;
            gt[i] = 1;
        }
        else
        {
            lt[i] = 0;
            gt[i] = 0;
        }
    }

    int less, greater, pivot_num;

    less = std::accumulate(lt, lt + array_length, 0);
    greater = std::accumulate(gt, gt + array_length, 0);

    // calculate the prefix sum of the array
    // Parallel_PrefixSum(lt, lt2, array_length);
    // Parallel_PrefixSum(gt, gt2, array_length);

    computeparallelprefix(lt, lt2, array_length);
    computeparallelprefix(gt, gt2, array_length);

#pragma omp for
    for (int i = 0; i < array_length; i++)
    {
        if (lt[i] == 1)
        {
            arr[left + lt2[i] - 1] = B[i];
        }
    }

#pragma omp for
    for (int i = left + less; i < left + array_length - greater; i++)
    {
        arr[i] = pivot;
    }

#pragma omp for
    for (int i = 0; i < array_length; i++)
    {
        if (gt[i] == 1)
        {
            arr[left + array_length - greater - less - 1 + gt2[i] + less] = B[i];
        }
    }
    return left + less;
}

/*
Function: sequential quicksort function
Return: a sorted array
*/
void SquickSort(int arr[], int low, int high)
{
    if (low < high)
    {
        int pos = partition(arr, low, high);
        SquickSort(arr, low, pos - 1);  //recursive run the left part
        SquickSort(arr, pos + 1, high); //recursive run the right part
    }
}

/*
Function: sequential quicksort function
Return: a sorted array
*/
void PquickSort(int arr[], int low, int high)
{
    if (low < high)
    {
        int pos = ppartition(arr, low, high);
        // printf("low: %d and high: %d and pivot position:%d\n", low, high, pos);
        if (pos == low or pos == high)
        {
            SquickSort(arr, low, high);
            return;
        }

        PquickSort(arr, low, pos - 1);  //recursive run the left part
        PquickSort(arr, pos + 1, high); //recursive run the right part
    }
}

int main(int argc, char *argv[])
{
    FILE *dst = NULL;
    if (argc != 4)
    {
        fprintf(stderr, "Correct Input format: mpiexec -n num <./*.out> <nums of procs: n> <*.h5> <*.txt>\n", argv[0]);
        exit(1);
    }
    int NUM = pow(10, 5) * std::stoi(argv[1]);
    H5std_string FILE_NAME(argv[2]);
    int *data;
    data = (int *)malloc(NUM * sizeof(int));

    // IO part: read the test array from the h5 dataset
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

    // quicksort the array recursively
    PquickSort(data, 0, NUM - 1);

    get_walltime(end_time);
    dst = fopen(argv[3], "w");
    for (int i = 0; i < NUM; i++)
        fprintf(dst, "%d\n", data[i]);
    fclose(dst);
    check(data, NUM);
    int threads;
#pragma omp parallel
    {
        threads = omp_get_num_threads();
    }
    printf("Quicksort OMP %d parallel algorithm on %d processors takes %f\n", NUM, threads, *end_time - *start_time);
    return 0;
}
