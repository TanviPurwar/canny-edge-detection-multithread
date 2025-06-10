#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "bmp.c"

#define HEIGHT 200
#define WIDTH 200
#define DIM 3

#define BATCH_SIZE 32
#define NUM_THREADS 32

//----------defining various filters needed----------

__constant__ float gaussianBlur_filter[] = {1/16.0, 2/16.0, 1/16.0, 2/16.0, 4/16.0, 2/16.0, 1/16.0, 2/16.0, 4/16.0}; //approximate kernel with sigma=1
__constant__ int sobelx_filter[] = {1, 0, -1, 2, 0, -2, 1, 0, -1};
__constant__ float sobely_filter[] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};

cudaEvent_t start[7];
cudaEvent_t end[7];
float elapsed_time[7];

void recordStartEvent(int event) {
	cudaEventCreate(&start[event]);
	cudaEventRecord(start[event], 0);
}

void recordEndEvent(int event){
	cudaEventCreate(&end[event]);
	cudaEventRecord(end[event], 0);
	cudaEventSynchronize(end[event]);
}

float calculateElapsedTime(int event) {
	float elapsed_time;
	cudaEventElapsedTime(&elapsed_time, start[event], end[event]);
	cudaEventDestroy(start[event]);
	cudaEventDestroy(end[event]);
	return elapsed_time; //in ms
}

void convert3Dto1D(unsigned char images[BATCH_SIZE][HEIGHT][WIDTH][DIM], int *img_batch){

	for(int n = 0; n < BATCH_SIZE; n++){
		for(int k = 0; k < DIM; k++){
			for (int i=0; i<HEIGHT; i++){
        			for (int j=0; j<WIDTH; j++) {
    					img_batch[(i*WIDTH + j) + (WIDTH*HEIGHT*k) + (n*WIDTH*HEIGHT*DIM)] = images[n][i][j][k];
    				}
    			}
		}
	}
}

void convert1Dto3D(int *img_batch, unsigned char images[BATCH_SIZE][HEIGHT][WIDTH][DIM]){

	for(int n = 0; n < BATCH_SIZE; n++){
               	for (int i=0; i<HEIGHT; i++){
                       	for (int j=0; j<WIDTH; j++) {

                               	images[n][i][j][0] = int(img_batch[(i*WIDTH + j) + (n*WIDTH*HEIGHT)]);
				images[n][i][j][1] = int(img_batch[(i*WIDTH + j) + (n*WIDTH*HEIGHT)]);
				images[n][i][j][2] = int(img_batch[(i*WIDTH + j) + (n*WIDTH*HEIGHT)]);
                       	}
                }
        }
}

__global__ void convertToGrayscale(int *img, int *gray_img){
	
	int x_idx = blockIdx.x * blockDim.x + threadIdx.x;
	int y_idx = blockIdx.y * blockDim.y + threadIdx.y;
	int z_idx = blockIdx.z;

	int gray_idx = ((y_idx * WIDTH) + x_idx) + (WIDTH*HEIGHT*z_idx);
	int idx = ((y_idx * WIDTH) + x_idx) + (WIDTH*HEIGHT*DIM*z_idx);	

	if(x_idx < WIDTH && y_idx < HEIGHT){
		gray_img[gray_idx] = (img[idx + (0 * WIDTH * HEIGHT)] + img[idx + (1 * WIDTH * HEIGHT)] + 
				img[idx + (2 * WIDTH * HEIGHT)])/3;
	}
}

__global__ void gaussianBlur(int *img, int *blur_img){

	int x_idx = blockIdx.x * blockDim.x + threadIdx.x;
        int y_idx = blockIdx.y * blockDim.y + threadIdx.y;
        int z_idx = blockIdx.z;

        int idx = ((y_idx * WIDTH) + x_idx) + (WIDTH*HEIGHT*z_idx);

	if(x_idx < WIDTH && y_idx < HEIGHT){

		float sum = 0.0;

		for(int row = y_idx-1; row <= y_idx+1; row++){
			for(int col = x_idx-1; col < x_idx+1; col++){		
				if(row >= 0 && row < HEIGHT && col >= 0 && col < WIDTH){
					
					int ky = row - y_idx + 1;
					int kx = col - x_idx + 1;
					int temp_idx = (row*WIDTH)+col + (WIDTH*HEIGHT*z_idx);

					sum = sum + (float(img[temp_idx]) * gaussianBlur_filter[(ky*3)+kx]);
				}
			}
		}

		blur_img[idx] = int(sum);
	}
}

__global__ void sobel(int *img, int *sobel_img, float *atan){
        int x_idx = blockIdx.x * blockDim.x + threadIdx.x;
        int y_idx = blockIdx.y * blockDim.y + threadIdx.y;
        int z_idx = blockIdx.z;

        int idx = ((y_idx * WIDTH) + x_idx) + (WIDTH*HEIGHT*z_idx);

        if(x_idx < WIDTH && y_idx < HEIGHT){

                int sumx = 0;
		int sumy = 0;

                for(int row = y_idx-1; row <= y_idx+1; row++){
                        for(int col = x_idx-1; col <= x_idx+1; col++){
                                if(row >= 0 && row < HEIGHT && col >= 0 && col < WIDTH){

                                        int ky = row - y_idx + 1;
                                        int kx = col - x_idx + 1;
                                        int temp_idx = (row*WIDTH)+col + (WIDTH*HEIGHT*z_idx);

                                        sumx = sumx + (img[temp_idx] * sobelx_filter[(ky*3)+kx]);
					sumy = sumy + (img[temp_idx] * sobely_filter[(ky*3)+kx]);
                                }
                        }
                }
		
		float sum = sqrt(float(sumx*sumx) + float(sumy*sumy));
                sobel_img[idx] = int(sum);
		atan[idx] = atan2(float(sumy), float(sumx));
        }
}

__global__ void nonmaxSupression(int *img, int *nonmax_img, float *atan){
	
	int x_idx = blockIdx.x * blockDim.x + threadIdx.x;
        int y_idx = blockIdx.y * blockDim.y + threadIdx.y;
        int z_idx = blockIdx.z;

        int idx = ((y_idx * WIDTH) + x_idx) + (WIDTH*HEIGHT*z_idx);

	int q = 255;
	int r = 255;
	
	if(x_idx < WIDTH && y_idx < HEIGHT){
		if((atan[idx] >= 0.0 && atan[idx] < 22.5) || (atan[idx] >= 157.5 && atan[idx] <= 180)){
			q = img[((y_idx*WIDTH)+(x_idx+1)) + (WIDTH*HEIGHT*z_idx)];
			r = img[((y_idx*WIDTH)+(x_idx-1)) + (WIDTH*HEIGHT*z_idx)];
		}
		else if(atan[idx] >= 22.5  && atan[idx] < 67.5){
			q = img[(((y_idx+1)*WIDTH)+(x_idx-1)) + (WIDTH*HEIGHT*z_idx)];
                	r = img[(((y_idx-1)*WIDTH)+(x_idx+1)) + (WIDTH*HEIGHT*z_idx)];
		}
		else if(atan[idx] >= 67.5  && atan[idx] < 112.5){
                        q = img[(((y_idx+1)*WIDTH)+x_idx) + (WIDTH*HEIGHT*z_idx)];
                        r = img[(((y_idx-1)*WIDTH)+x_idx) + (WIDTH*HEIGHT*z_idx)];
                }
		else if(atan[idx] >= 112.5  && atan[idx] < 157.5){
                        q = img[(((y_idx-1)*WIDTH)+(x_idx-1)) + (WIDTH*HEIGHT*z_idx)];
                        r = img[(((y_idx+1)*WIDTH)+(x_idx+1)) + (WIDTH*HEIGHT*z_idx)];
                }

		if(img[idx] > q && img[idx] > r){
			nonmax_img[idx] = img[idx];
		}
		else{
			nonmax_img[idx] = 0;
		}
	}
}

__global__ void doubleThreshold(int *img, int *dt_img){

	int x_idx = blockIdx.x * blockDim.x + threadIdx.x;
        int y_idx = blockIdx.y * blockDim.y + threadIdx.y;
        int z_idx = blockIdx.z;

        int idx = ((y_idx * WIDTH) + x_idx) + (WIDTH*HEIGHT*z_idx);

	if(x_idx < WIDTH && y_idx < HEIGHT){
		if(img[idx] <= 20){
			dt_img[idx] = 0;
		}
		else if(img[idx] >= 40){
                	dt_img[idx] = 225; //strong pixel
        	}
		else{
                	dt_img[idx] = 30; //weak pixel
        	}
	}
}

__global__ void hysterisis(int *img, int *hyst_img){

	int x_idx = blockIdx.x * blockDim.x + threadIdx.x;
        int y_idx = blockIdx.y * blockDim.y + threadIdx.y;
        int z_idx = blockIdx.z;

	int idx = ((y_idx * WIDTH) + x_idx) + (WIDTH*HEIGHT*z_idx);

        if(x_idx < WIDTH && y_idx < HEIGHT){

                for(int row = y_idx-1; row <= y_idx+1; row++){
                        for(int col = x_idx-1; col < x_idx+1; col++){
                                if(row >= 0 && row < HEIGHT && col >= 0 && col < WIDTH){

                                        int temp_idx = (row*WIDTH)+col + (WIDTH*HEIGHT*z_idx);

                                        if(img[temp_idx] == 255){
						img[idx] = 255;
					}
                                }
                        }
                }

                hyst_img[idx] = img[idx];
        }
}

int main(){

	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);

	printf("  Device Name: %s\n", deviceProp.name);
	printf("  Compute Capability: %d.%d\n", deviceProp.major, deviceProp.minor);
	printf("  Total Global Memory: %zu bytes\n", deviceProp.totalGlobalMem);
	printf("  Multiprocessor Count: %d\n", deviceProp.multiProcessorCount);
	printf("  Max Threads Per Block: %d\n", deviceProp.maxThreadsPerBlock);
	printf("  Max Threads Dimension: (%d, %d, %d)\n", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1],
			deviceProp.maxThreadsDim[2]);
        printf("  Max Grid Size: (%d, %d, %d)\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
	printf("  Max Threads Per Multiprocessor: %d\n", deviceProp.maxThreadsPerMultiProcessor);


	// ----------loading the images----------
	unsigned char images[BATCH_SIZE][HEIGHT][WIDTH][DIM]; // to store the input image batch

	for(int i = 0; i < BATCH_SIZE; i++){ //loops over the batch of images and stores them

		char inputImageFileName[20];
		sprintf(inputImageFileName, "image%d.bmp", i+1);
		
		char* inputImageFileNamePtr = strdup(inputImageFileName);

		// function from bmp.c to read the input image
		readBitmapImage((unsigned char*) images[i], HEIGHT, WIDTH, inputImageFileNamePtr);
	}

	// ----------converting images from 3d to 1d and joining all of them in a single array----------
	
	// declare and allocate memory for image batch (3D to 1D) in host
	int *img_batch = (int*)malloc((BATCH_SIZE*HEIGHT*WIDTH*DIM) * sizeof(int));

	// flatten the image
	convert3Dto1D(images, img_batch);
	
	//----------define blocksize and gridsize for various kernels----------

	dim3 grid((WIDTH + NUM_THREADS - 1)/NUM_THREADS, (HEIGHT + NUM_THREADS - 1)/NUM_THREADS, BATCH_SIZE);
        dim3 block(NUM_THREADS, NUM_THREADS, 1);


	//----------allocate memory for various arrays in host and device----------

	int *d_img;
        cudaMalloc((void**)&d_img, (BATCH_SIZE*HEIGHT*WIDTH*DIM) * sizeof(int));

	// grayscale
	int *gray_img = (int*)malloc((BATCH_SIZE*HEIGHT*WIDTH) * sizeof(int));

	int *d_gray_img;
    	cudaMalloc((void**)&d_gray_img, (BATCH_SIZE*HEIGHT*WIDTH) * sizeof(int));
	
	// gaussian blur
	int *blur_img = (int*)malloc((BATCH_SIZE*HEIGHT*WIDTH) * sizeof(int));

	int *d_blur_img;
	cudaMalloc((void**)&d_blur_img, (BATCH_SIZE*HEIGHT*WIDTH) * sizeof(int));

	int *d_blur_input;
        cudaMalloc((void**)&d_blur_input, (BATCH_SIZE*HEIGHT*WIDTH) * sizeof(int));

	// sobel
	int *sobel_img = (int*)malloc((BATCH_SIZE*HEIGHT*WIDTH) * sizeof(int));

	int *d_sobel_img;
	cudaMalloc((void**)&d_sobel_img, (BATCH_SIZE*HEIGHT*WIDTH) * sizeof(int));

	float *d_atan2;
	cudaMalloc((void**)&d_atan2, (BATCH_SIZE*HEIGHT*WIDTH) * sizeof(float));

	int *d_sobel_input;
	cudaMalloc((void**)&d_sobel_input, (BATCH_SIZE*HEIGHT*WIDTH) * sizeof(int));
	
	float *atan = (float*)malloc((BATCH_SIZE*HEIGHT*WIDTH) * sizeof(float));

	// non maxima supression
	int *nonmax_img = (int*)malloc((BATCH_SIZE*HEIGHT*WIDTH) * sizeof(int));

	int *d_nonmax_img;
	cudaMalloc((void**)&d_nonmax_img, (BATCH_SIZE*HEIGHT*WIDTH) * sizeof(int));

	int *d_nonmax_input;
	cudaMalloc((void**)&d_nonmax_input, (BATCH_SIZE*HEIGHT*WIDTH) * sizeof(int));

	float *d_atan2_input;
	cudaMalloc((void**)&d_atan2_input, (BATCH_SIZE*HEIGHT*WIDTH) * sizeof(float));

	// double threshold
	int *doubThresh_img = (int*)malloc((BATCH_SIZE*HEIGHT*WIDTH) * sizeof(int));

        int *d_doubThresh_img;
        cudaMalloc((void**)&d_doubThresh_img, (BATCH_SIZE*HEIGHT*WIDTH) * sizeof(int));

        int *d_doubThresh_input;
        cudaMalloc((void**)&d_doubThresh_input, (BATCH_SIZE*HEIGHT*WIDTH) * sizeof(int));

	// hysterisis
	int *hyst_img = (int*)malloc((BATCH_SIZE*HEIGHT*WIDTH) * sizeof(int));

        int *d_hyst_img;
        cudaMalloc((void**)&d_hyst_img, (BATCH_SIZE*HEIGHT*WIDTH) * sizeof(int));

        int *d_hyst_input;
        cudaMalloc((void**)&d_hyst_input, (BATCH_SIZE*HEIGHT*WIDTH) * sizeof(int));

	
	//----------run kernels---------- 
	recordStartEvent(0);

	//grayscale
        cudaMemcpy(d_img, img_batch, (BATCH_SIZE*HEIGHT*WIDTH*DIM) * sizeof(int), cudaMemcpyHostToDevice);
	
	recordStartEvent(1);	
	convertToGrayscale<<<grid, block>>>(d_img, d_gray_img);
	recordEndEvent(1);
	elapsed_time[1] = calculateElapsedTime(1);
		
	cudaMemcpy(gray_img, d_gray_img, (BATCH_SIZE*HEIGHT*WIDTH) * sizeof(int), cudaMemcpyDeviceToHost);

	//gaussian blur
	cudaMemcpy(d_blur_input, gray_img, (BATCH_SIZE*HEIGHT*WIDTH) * sizeof(int), cudaMemcpyHostToDevice);
	
	recordStartEvent(2);
	gaussianBlur<<<grid, block>>>(d_blur_input, d_blur_img);
	recordEndEvent(2);
        elapsed_time[2] = calculateElapsedTime(2);

	cudaMemcpy(blur_img, d_blur_img, (BATCH_SIZE*HEIGHT*WIDTH) * sizeof(int), cudaMemcpyDeviceToHost);
	

	//sobel
	cudaMemcpy(d_sobel_input, blur_img, (BATCH_SIZE*HEIGHT*WIDTH) * sizeof(int), cudaMemcpyHostToDevice);
	
	recordStartEvent(3); 
	sobel<<<grid, block>>>(d_sobel_input, d_sobel_img, d_atan2);
	recordEndEvent(3);
        elapsed_time[3] = calculateElapsedTime(3);

	cudaMemcpy(sobel_img, d_sobel_img, (BATCH_SIZE*HEIGHT*WIDTH) * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(atan, d_atan2, (BATCH_SIZE*HEIGHT*WIDTH) * sizeof(float), cudaMemcpyDeviceToHost);

	
	//non-maxima supression
	cudaMemcpy(d_nonmax_input, sobel_img, (BATCH_SIZE*HEIGHT*WIDTH) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_atan2_input, atan, (BATCH_SIZE*HEIGHT*WIDTH) * sizeof(float), cudaMemcpyHostToDevice);
	
	recordStartEvent(4);
	nonmaxSupression<<<grid, block>>>(d_nonmax_input, d_nonmax_img, d_atan2_input);
	recordEndEvent(4);
        elapsed_time[4] = calculateElapsedTime(4);

	cudaMemcpy(nonmax_img, d_nonmax_img, (BATCH_SIZE*HEIGHT*WIDTH) * sizeof(int), cudaMemcpyDeviceToHost);
	

	//double thresholding
	cudaMemcpy(d_doubThresh_input, nonmax_img, (BATCH_SIZE*HEIGHT*WIDTH) * sizeof(int), cudaMemcpyHostToDevice);
	
	recordStartEvent(5); 
        doubleThreshold<<<grid, block>>>(d_doubThresh_input, d_doubThresh_img);
	recordEndEvent(5);
        elapsed_time[5] = calculateElapsedTime(5);

        cudaMemcpy(doubThresh_img, d_doubThresh_img, (BATCH_SIZE*HEIGHT*WIDTH) * sizeof(int), cudaMemcpyDeviceToHost);
	
	
	//hysterisis
	cudaMemcpy(d_hyst_input, doubThresh_img, (BATCH_SIZE*HEIGHT*WIDTH) * sizeof(int), cudaMemcpyHostToDevice);
	
	recordStartEvent(6); 
        hysterisis<<<grid, block>>>(d_hyst_input, d_hyst_img);
	recordEndEvent(6);
        elapsed_time[6] = calculateElapsedTime(6);

        cudaMemcpy(hyst_img, d_hyst_img, (BATCH_SIZE*HEIGHT*WIDTH) * sizeof(int), cudaMemcpyDeviceToHost);

	recordEndEvent(0);
        elapsed_time[0] = calculateElapsedTime(0); 


	//----------extracting results----------
	
	unsigned char final_image[BATCH_SIZE][HEIGHT][WIDTH][DIM]; // to store the grayscaled image
	
	convert1Dto3D(hyst_img, final_image);

	for(int i = 0; i < BATCH_SIZE; i++){

		char outputImageFileName[15];
                sprintf(outputImageFileName, "canny%d.bmp", i+1);
		char* outputImageFileNamePtr = strdup(outputImageFileName);
		
		// function from bmp.c to write the output image
		writeBitmapImage((unsigned char*) final_image[i], HEIGHT, WIDTH, outputImageFileName);
	}
	
	//----------freeing allocated memory----------
	
	free(img_batch);
	free(gray_img);
	free(blur_img);
	cudaFree(d_img);
	cudaFree(d_gray_img);
	cudaFree(d_blur_input);
	cudaFree(d_blur_img);
	cudaFree(d_sobel_img);
	cudaFree(d_sobel_input);
	cudaFree(d_atan2);
	cudaFreeHost(sobel_img);
	free(atan);
	free(nonmax_img);
	cudaFree(d_nonmax_img);
	cudaFree(d_nonmax_input);
	cudaFree(d_atan2_input);
	free(doubThresh_img);
	cudaFree(d_doubThresh_img);
        cudaFree(d_doubThresh_input);
	free(hyst_img);
        cudaFree(d_hyst_img);
        cudaFree(d_hyst_input);
	

	//----------print elapsed time for various kernels----------
	for(int i = 0; i < 7; i++){
		printf("%f\n", elapsed_time[i]);
	}

	return 0;
}

