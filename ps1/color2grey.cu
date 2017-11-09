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

/* Private function prototypes -----------------------------------------------*/
/* Private functions ---------------------------------------------------------*/

__global__ void  color2grey_kernel(uint8_t* d_in, uint8_t* d_out,
								   uint32_t img_h, uint32_t img_w )
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	
	// note: bgr
	int id = (y*img_w + x)*3;

	d_out[y*img_w+x] = 0.0722*d_in[id] + 0.7152*d_in[id+1] + 0.2126*d_in[id+2];
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
	std::string fn_in, fn_out;
	cv::Mat img_in, img_out;
	uint32_t img_w, img_h;


	// phrase arguments
	if(argc <3)
	{
		std::cerr << "ERROR: no enough arguments" << std::endl
				  << "usage: " << argv[0]
				  << " file_input file_output "
				  << std::endl;
		return 1;
	}
	else
	{
		fn_in = std::string(argv[1]);
		fn_out = std::string(argv[2]);
	}

	std::cout << "input file: " << fn_in << std::endl
			  << "output file: " << fn_out << std::endl;



	// load image
	img_in = cv::imread(fn_in, CV_LOAD_IMAGE_COLOR);
	if(img_in.data == NULL)
	{
		std::cerr << "Fail to load image, " << fn_in << std::endl;
		return 2;
	}
	img_w = img_in.cols;
	img_h = img_in.rows;
	img_out.create(img_h, img_w, CV_8UC1);
	if(!img_out.isContinuous())
	{
		std::cerr << "img out not continuour" << std::endl;
		return 3;
	}


	// prepare GPU
	uint8_t* d_input;
	uint8_t* d_output;

	checkCudaErrors( cudaMalloc(&d_input, img_w*img_h*3) );
	checkCudaErrors( cudaMalloc(&d_output, img_w*img_h) );

	checkCudaErrors( cudaMemcpy(d_input, img_in.ptr(),
								img_w*img_h*3, cudaMemcpyHostToDevice) );
	checkCudaErrors( cudaMemset(d_output, 0, img_w*img_h) );
	
	// map
	dim3 blockSize (32,32,1);
	dim3 gridSize (1,1,1);

	gridSize.x = img_w%32 == 0? img_w/32:img_w/32+1;
	gridSize.y = img_h%32 == 0? img_h/32:img_h/32+1;

	GpuTimer timer;
	
	timer.Start();
	color2grey_kernel<<<gridSize, blockSize>>> (d_input, d_output, img_h, img_w);
	timer.Stop();

	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());

	std::cout << "Used time(ms): " << timer.Elapsed() << std::endl;

	// copy data back and clean
	checkCudaErrors( cudaMemcpy( img_out.ptr(), d_output,
								 img_h*img_w, cudaMemcpyDeviceToHost) );
	checkCudaErrors( cudaFree(d_input) );
	checkCudaErrors( cudaFree(d_output) );

	// crop and save image
	if(!cv::imwrite(fn_out, img_out))
	{
		std::cerr << "ERROR: Fail to save image!" << std::endl;
		return 4;
	}
	
	std::cout << "-----------END---------------" << std::endl;
	return 0;
}


/*****************************END OF FILE**************************************/
