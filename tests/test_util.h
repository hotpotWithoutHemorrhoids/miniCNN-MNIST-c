#ifndef TESTS_UTIL_H
#define TESTS_UTIL_H
#include<math.h>

int compare_float_arr(const float* arr1, const float* arr2, int size, float epsilon){
    for (size_t i = 0; i < size; i++){
        if(fabs(arr1[i] - arr2[i])>epsilon)
            return 0;// not equals
    }
    return 1; // equal
}


#endif