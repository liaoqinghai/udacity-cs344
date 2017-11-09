/**
 ******************************************************************************
 * @file	gaussian_blur.cpp
 * @author	Nick.Liao
 * @version V1.0.0
 * @date	2017-11-08
 * @brief	gaussian_blur.cpp
 ******************************************************************************
 * @attention
 *
 * Copyright (C) 2017 Nick.Liao <simplelife_nick@hotmail.com>
 * Distributed under terms of the MIT license.
 ******************************************************************************
 */

/* Includes ------------------------------------------------------------------*/

#include <string>
#include <stdint.h>
#include <iostream>
#include <cmath>

#include <opencv2/opencv.hpp>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "utils.h"
#include "timer.h"

/* Private typedef -----------------------------------------------------------*/
/* Private define ------------------------------------------------------------*/
/* Private macro -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
const double PI =  3.1415926;

/* Private function prototypes -----------------------------------------------*/
/* Private functions ---------------------------------------------------------*/

__global__ void  guassian_kernel(uint8_t* d_in, uint8_t* d_out,
								 float* d_weights, uint32_t img_h,
								 uint32_t img_w, uint32_t filter_size )
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	int z = threadIdx.z;

	int offset = (filter_size-1)/2;

	// don't process edge area
	if( x<offset || x >= (img_w-offset) ||
			y<offset ||	y >= (img_h-offset) )
	{
		return;
	}

	float pixel = 0;
	for(int i=-offset; i<= offset; i++)
	{
		for(int j=-offset; j<= offset; j++)
		{
			pixel += d_in[((y+i)*img_w+(x+j))*3+z]
					 *d_weights[(i+offset)*filter_size + (j+offset)];
		}
	}

	d_out[(y*img_w+x)*3+z] = pixel;
}

/* Exported functions --------------------------------------------------------*/

/**
 * @brief
 * @param
 * @note
 * @return None
 */

int main(int argc, char** argv)
{
	uint32_t filter_size = 5;
	float filter_sigma = 1.0;
	std::string fn_in, fn_out;
	cv::Mat img_in, img_out;

	uint32_t img_w, img_h;
	float* filter_weights;


	// phrase arguments
	if(argc <3)
	{
		std::cerr << "ERROR: no enough arguments" << std::endl
				  << "usage: " << argv[0]
				  << " file_input file_output [filter_size=5, filter_sigma=1.0]"
				  << std::endl;
		return 1;
	}
	else
	{
		fn_in = std::string(argv[1]);
		fn_out = std::string(argv[2]);

		if(argc >=4)
		{
			int fs = std::stoi(std::string(argv[3]));
			if(fs <= 1 || fs%2 == 0  )
			{
				std::cerr << "Filter size is illegal, use default 5" << std::endl;
			}
			else
			{
				filter_size = fs;
			}
		}

		if(argc >=5)
		{
			int fs = std::stof(std::string(argv[4]));
			if(fs <= 0  )
			{
				std::cerr << "Filter sigma is illegal, use default 1.0" << std::endl;
			}
			else
			{
				filter_sigma = fs;
			}
		}
	}




	// load image
	img_in = cv::imread(fn_in, CV_LOAD_IMAGE_COLOR);
	if(img_in.data == NULL)
	{
		std::cerr << "Fail to load image, " << fn_in << std::endl;
		return 2;
	}
	img_w = img_in.cols;
	img_h = img_in.rows;
	img_out.create(img_h, img_w, CV_8UC3);
	if(!img_out.isContinuous())
	{
		std::cerr << "img out not continuour" << std::endl;
		return 3;
	}


	// get gaussian weight
	filter_weights = new float[filter_size*filter_size];
	double normal_sum = 0;
	for(uint32_t i=0; i<filter_size; i++)
	{
		for(uint32_t j=0; j<filter_size; j++)
		{
			int y = i-(filter_size-1)/2;
			int x = j-(filter_size-1)/2;
			filter_weights[i*filter_size+j] = std::exp((y*y+x*x)/(-2*filter_sigma*filter_sigma))/(2*PI*filter_sigma*filter_sigma);
			normal_sum += filter_weights[i*filter_size+j];
		}
	}
	for(uint32_t i=0; i<filter_size*filter_size; i++)
	{
		filter_weights[i] /= normal_sum;
	}

	// prepare GPU
	uint8_t* d_input;
	uint8_t* d_output;
	float*  d_weights;

	checkCudaErrors( cudaMalloc(&d_input, img_w*img_h*3) );
	checkCudaErrors( cudaMalloc(&d_output, img_w*img_h*3) );
	checkCudaErrors( cudaMalloc(&d_weights, filter_size*filter_size*sizeof(float)) );

	checkCudaErrors( cudaMemcpy(d_input, img_in.ptr(),
								img_w*img_h*3, cudaMemcpyHostToDevice) );
	checkCudaErrors( cudaMemset(d_output, 0, img_w*img_h*3) );
	checkCudaErrors( cudaMemcpy(d_weights, filter_weights,
								filter_size*filter_size*sizeof(float),
								cudaMemcpyHostToDevice));


	// map
	dim3 blockSize (18,18,3);
	dim3 gridSize (1,1,1);

	gridSize.x = img_w%18 == 0? img_w/18:img_w/18+1;
	gridSize.y = img_h%18 == 0? img_h/18:img_h/18+1;

	GpuTimer timer;
	
	timer.Start();
	guassian_kernel<<<gridSize, blockSize>>> (d_input, d_output, d_weights,
			img_h, img_w, filter_size);
	timer.Stop();

	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());
	
	std::cout << "take time(ms): " << timer.Elapsed() << std::endl;

	// copy data back and clean
	checkCudaErrors( cudaMemcpy( img_out.ptr(), d_output,
								 img_h*img_w*3, cudaMemcpyDeviceToHost) );
	checkCudaErrors( cudaFree(d_input) );
	checkCudaErrors( cudaFree(d_output) );
	checkCudaErrors( cudaFree(d_weights) );
	delete[] filter_weights;

	// crop and save image
	uint32_t offset = (filter_size-1)/2;
	cv::Rect roi(offset, offset, img_w-offset*2, img_h-offset*2 );
	cv::Mat crop = img_out(roi);
	if(!cv::imwrite(fn_out, crop))
	{
		std::cerr << "ERROR: Fail to save image!" << std::endl;
		return 4;
	}

	return 0;
}


/*****************************END OF FILE**************************************/
