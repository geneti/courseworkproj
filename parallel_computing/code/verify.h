#include <iostream>
#include <stdio.h>

void check(int arr[], int size){
    int err = 0;
    if (size < 2){
        printf("Correct order with %d errors\n", 0);
    }
    for (int i=1;i<size;i++){
        if (arr[i-1] > arr[i]){
            err++;
        }
    }
    if (err==0){
        printf("Correct order with %d errors\n", 0);
    }
    else{
        printf("Wrong order with %d errors\n", err);
    }
}