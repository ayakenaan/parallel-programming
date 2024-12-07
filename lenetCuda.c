#include "lenet.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define FILE_TRAIN_IMAGE "train-images-idx3-ubyte"
#define FILE_TRAIN_LABEL "train-labels-idx1-ubyte"
#define FILE_TEST_IMAGE "t10k-images-idx3-ubyte"
#define FILE_TEST_LABEL "t10k-labels-idx1-ubyte"
#define LENET_FILE "model.dat"
#define COUNT_TRAIN 60000
#define COUNT_TEST 10000

// CUDA kernel for valid convolution
__global__ void conv_valid_kernel(double *input, double *output, double *weight,
                                   int input_len0, int input_len1,
                                   int weight_len0, int weight_len1,
                                   int output_len0, int output_len1)
{
    int o0 = blockIdx.x * blockDim.x + threadIdx.x;
    int o1 = blockIdx.y * blockDim.y + threadIdx.y;

    if (o0 < output_len0 && o1 < output_len1)
    {
        double sum = 0;
        for (int w0 = 0; w0 < weight_len0; ++w0)
        {
            for (int w1 = 0; w1 < weight_len1; ++w1)
            {
                sum += input[(o0 + w0) * input_len1 + (o1 + w1)] * weight[w0 * weight_len1 + w1];
            }
        }
        output[o0 * output_len1 + o1] += sum;
    }
}

// Function to read data
int read_data(unsigned char(*data)[28][28], unsigned char label[], const int count, const char data_file[], const char label_file[])
{
    FILE *fp_image = fopen(data_file, "rb");
    FILE *fp_label = fopen(label_file, "rb");
    if (!fp_image || !fp_label) return 1;
    fseek(fp_image, 16, SEEK_SET);
    fseek(fp_label, 8, SEEK_SET);
    fread(data, sizeof(*data) * count, 1, fp_image);
    fread(label, count, 1, fp_label);
    fclose(fp_image);
    fclose(fp_label);
    return 0;
}

// Training function
void training(LeNet5 *lenet, image *train_data, uint8 *train_label, int batch_size, int total_size)
{
    for (int i = 0, percent = 0; i <= total_size - batch_size; i += batch_size)
    {
        TrainBatch(lenet, train_data + i, train_label + i, batch_size);
        if (i * 100 / total_size > percent)
            printf("batchsize:%d\ttrain:%2d%%\n", batch_size, percent = i * 100 / total_size);
    }
}

// Testing function
int testing(LeNet5 *lenet, image *test_data, uint8 *test_label, int total_size)
{
    int right = 0, percent = 0;
    for (int i = 0; i < total_size; ++i)
    {
        uint8 l = test_label[i];
        int p = Predict(lenet, test_data[i], 10);
        right += l == p;
        if (i * 100 / total_size > percent)
            printf("test:%2d%%\n", percent = i * 100 / total_size);
    }
    return right;
}

// Save model to file
int save(LeNet5 *lenet, char filename[])
{
    FILE *fp = fopen(filename, "wb");
    if (!fp) return 1;
    fwrite(lenet, sizeof(LeNet5), 1, fp);
    fclose(fp);
    return 0;
}

// Load model from file
int load(LeNet5 *lenet, char filename[])
{
    FILE *fp = fopen(filename, "rb");
    if (!fp) return 1;
    fread(lenet, sizeof(LeNet5), 1, fp);
    fclose(fp);
    return 0;
}

// CUDA convolution macro for valid convolution
#define CONVOLUTE_VALID(input, output, weight)                                        \
{                                                                                     \
    int input_len0 = GETLENGTH(input);                                                \
    int input_len1 = GETLENGTH(*(input));                                             \
    int weight_len0 = GETLENGTH(weight);                                              \
    int weight_len1 = GETLENGTH(*(weight));                                           \
    int output_len0 = GETLENGTH(output);                                              \
    int output_len1 = GETLENGTH(*(output));                                           \
                                                                                      \
    double *d_input, *d_output, *d_weight;                                           \
    cudaMalloc(&d_input, sizeof(double) * input_len0 * input_len1);                   \
    cudaMalloc(&d_output, sizeof(double) * output_len0 * output_len1);                \
    cudaMalloc(&d_weight, sizeof(double) * weight_len0 * weight_len1);                \
                                                                                      \
    cudaMemcpy(d_input, input, sizeof(double) * input_len0 * input_len1, cudaMemcpyHostToDevice); \
    cudaMemcpy(d_weight, weight, sizeof(double) * weight_len0 * weight_len1, cudaMemcpyHostToDevice); \
    cudaMemset(d_output, 0, sizeof(double) * output_len0 * output_len1);             \
                                                                                      \
    dim3 blockDim(16, 16);                                                            \
    dim3 gridDim((output_len0 + blockDim.x - 1) / blockDim.x, (output_len1 + blockDim.y - 1) / blockDim.y); \
    conv_valid_kernel<<<gridDim, blockDim>>>(d_input, d_output, d_weight,              \
                                              input_len0, input_len1, weight_len0, weight_len1, \
                                              output_len0, output_len1);              \
                                                                                      \
    cudaMemcpy(output, d_output, sizeof(double) * output_len0 * output_len1, cudaMemcpyDeviceToHost); \
                                                                                      \
    cudaFree(d_input);                                                                \
    cudaFree(d_output);                                                               \
    cudaFree(d_weight);                                                               \
}

// Convolution Forward (using CUDA)
#define CONVOLUTION_FORWARD(input, output, weight, bias, action)                     \
{                                                                                     \
    int weight_len0 = GETLENGTH(weight);                                              \
    int weight_len1 = GETLENGTH(*weight);                                             \
    int output_len = GETLENGTH(output);                                               \
                                                                                      \
    /* Copy input and weights to the device */                                        \
    double *d_input, *d_output, *d_weight, *d_bias;                                   \
    cudaMalloc(&d_input, sizeof(double) * weight_len0 * weight_len1);                 \
    cudaMalloc(&d_output, sizeof(double) * output_len);                               \
    cudaMalloc(&d_weight, sizeof(double) * weight_len0 * weight_len1);                \
    cudaMalloc(&d_bias, sizeof(double) * output_len);                                 \
                                                                                      \
    cudaMemcpy(d_input, input, sizeof(double) * weight_len0 * weight_len1, cudaMemcpyHostToDevice); \
    cudaMemcpy(d_weight, weight, sizeof(double) * weight_len0 * weight_len1, cudaMemcpyHostToDevice); \
    cudaMemcpy(d_bias, bias, sizeof(double) * output_len, cudaMemcpyHostToDevice);    \
                                                                                      \
    /* Call the convolution kernel */                                                  \
    dim3 blockDim(16, 16);                                                            \
    dim3 gridDim((weight_len0 + blockDim.x - 1) / blockDim.x,                         \
                 (weight_len1 + blockDim.y - 1) / blockDim.y);                       \
    conv_valid_kernel<<<gridDim, blockDim>>>(d_input, d_output, d_weight,             \
                                              weight_len0, weight_len1,              \
                                              output_len, output_len);               \
                                                                                      \
    /* Copy result back to host memory and apply activation function */               \
    cudaMemcpy(output, d_output, sizeof(double) * output_len, cudaMemcpyDeviceToHost); \
                                                                                      \
    for (int j = 0; j < output_len; ++j)                                              \
        FOREACH(i, GETCOUNT(output[j]))                                               \
            ((double *)output[j])[i] = action(((double *)output[j])[i] + bias[j]);    \
                                                                                      \
    /* Free device memory */                                                           \
    cudaFree(d_input);                                                                \
    cudaFree(d_output);                                                               \
    cudaFree(d_weight);                                                               \
    cudaFree(d_bias);                                                                 \
}

// Convolution Backward (using CUDA)
#define CONVOLUTION_BACKWARD(input, inerror, outerror, weight, wd, bd, actiongrad)    \
{                                                                                     \
    int weight_len0 = GETLENGTH(weight);                                              \
    int weight_len1 = GETLENGTH(*weight);                                             \
    int inerror_len = GETCOUNT(inerror);                                              \
    int outerror_len = GETLENGTH(outerror);                                           \
                                                                                      \
    /* Allocate device memory */                                                      \
    double *d_input, *d_inerror, *d_outerror, *d_weight, *d_wd, *d_bd;                \
    cudaMalloc(&d_input, sizeof(double) * weight_len0 * weight_len1);                 \
    cudaMalloc(&d_inerror, sizeof(double) * inerror_len);                             \
    cudaMalloc(&d_outerror, sizeof(double) * outerror_len);                           \
    cudaMalloc(&d_weight, sizeof(double) * weight_len0 * weight_len1);                \
    cudaMalloc(&d_wd, sizeof(double) * weight_len0 * weight_len1);                    \
    cudaMalloc(&d_bd, sizeof(double) * outerror_len);                                 \
                                                                                      \
    /* Copy data to device */                                                         \
    cudaMemcpy(d_input, input, sizeof(double) * weight_len0 * weight_len1, cudaMemcpyHostToDevice); \
    cudaMemcpy(d_inerror, inerror, sizeof(double) * inerror_len, cudaMemcpyHostToDevice); \
    cudaMemcpy(d_outerror, outerror, sizeof(double) * outerror_len, cudaMemcpyHostToDevice); \
    cudaMemcpy(d_weight, weight, sizeof(double) * weight_len0 * weight_len1, cudaMemcpyHostToDevice); \
                                                                                      \
    /* Compute gradients (kernel call for convolution backward) */                    \
    dim3 blockDim(16, 16);                                                            \
    dim3 gridDim((inerror_len + blockDim.x - 1) / blockDim.x,                         \
                 (outerror_len + blockDim.y - 1) / blockDim.y);                       \
    conv_valid_kernel<<<gridDim, blockDim>>>(d_input, d_inerror, d_weight,            \
                                              weight_len0, weight_len1,              \
                                              inerror_len, outerror_len);            \
                                                                                      \
    /* Copy results back to host and apply gradient update */                          \
    cudaMemcpy(wd, d_wd, sizeof(double) * weight_len0 * weight_len1, cudaMemcpyDeviceToHost); \
    cudaMemcpy(bd, d_bd, sizeof(double) * outerror_len, cudaMemcpyDeviceToHost);      \
                                                                                      \
    /* Update weights and biases */                                                   \
    for (int i = 0; i < weight_len0; ++i)                                             \
        for (int j = 0; j < weight_len1; ++j)                                         \
            weight[i * weight_len1 + j] += actiongrad(d_wd[i * weight_len1 + j]);    \
    for (int j = 0; j < outerror_len; ++j)                                            \
        bias[j] += actiongrad(d_bd[j]);                                               \
                                                                                      \
    /* Free device memory */                                                           \
    cudaFree(d_input);                                                                \
    cudaFree(d_inerror);                                                              \
    cudaFree(d_outerror);                                                             \
    cudaFree(d_weight);                                                               \
    cudaFree(d_wd);                                                                   \
    cudaFree(d_bd);                                                                   \
}

void foo()
{
    image *train_data = (image *)calloc(COUNT_TRAIN, sizeof(image));
    uint8 *train_label = (uint8 *)calloc(COUNT_TRAIN, sizeof(uint8));
    image *test_data = (image *)calloc(COUNT_TEST, sizeof(image));
    uint8 *test_label = (uint8 *)calloc(COUNT_TEST, sizeof(uint8));
   
    if (read_data(train_data, train_label, COUNT_TRAIN, FILE_TRAIN_IMAGE, FILE_TRAIN_LABEL))
    {
        printf("ERROR!!!\nDataset File Not Find!Please Copy Dataset to the Floder Included the exe\n");
        free(train_data);
        free(train_label);
        system("pause");
    }
    if (read_data(test_data, test_label, COUNT_TEST, FILE_TEST_IMAGE, FILE_TEST_LABEL))
    {
        printf("ERROR!!!\nDataset File Not Find!Please Copy Dataset to the Floder Included the exe\n");
        free(test_data);
        free(test_label);
        system("pause");
    }

    LeNet5 *lenet = (LeNet5 *)malloc(sizeof(LeNet5));
    if (load(lenet, LENET_FILE))
        Initial(lenet);
   
    clock_t start = clock();
    int batches[] = {300};
    for (int i = 0; i < sizeof(batches) / sizeof(*batches); ++i)
        training(lenet, train_data, train_label, batches[i], COUNT_TRAIN);
   
    int right = testing(lenet, test_data, test_label, COUNT_TEST);
    printf("%d/%d\n", right, COUNT_TEST);
    printf("Time:%u\n", (unsigned)((clock() - start) / CLOCKS_PER_SEC));
    print_cumulative_times();
    // save(lenet, LENET_FILE);
   
    free(lenet);
    free(train_data);
    free(train_label);
    free(test_data);
    free(test_label);
    system("pause");
}

int main()
{
    foo();
    return 0;
}
