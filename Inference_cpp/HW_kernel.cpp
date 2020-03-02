#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// macro for HLS synthesis
//#define HLS

#ifdef HLS
#include "ap_int.h"
#include "hls_half.h"
typedef float DTYPE;
#else
typedef float DTYPE;
#endif

void kernel(
#ifdef HLS
	DTYPE src_img[3*416*416],
	DTYPE dst_img[64*102*102],
	DTYPE weight[64*3*11*11],
	DTYPE bias[64]
#else
	DTYPE *src_img,
	DTYPE *dst_img,
	DTYPE *weight,
	DTYPE *bias
#endif
);

// ---------------------------------------------------
// Convolution(3chx416x416) in the 1st layer
// ---------------------------------------------------
void kernel(
#ifdef HLS
	DTYPE src_img[3*416*416],
	DTYPE dst_img[64*102*102],
	DTYPE weight[64*3*11*11],
	DTYPE bias[64]
#else
	DTYPE *src_img,
	DTYPE *dst_img,
	DTYPE *weight,
	DTYPE *bias
#endif
)
{
//(0): Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4))

	for( int och = 0; och < 64; och++){
		int dst_y;
		int dst_x;
		dst_y = 0;

		for( int y = 0; y < 416 - 11; y += 4){
			dst_x = 0;

			for( int x = 0; x < 416 - 11; x+= 4){
				float tmp = 0;

				for( int ich = 0; ich < 3; ich++){
					for( int ky = 0; ky < 11; ky++){
						for( int kx = 0; kx < 11; kx++){
							tmp += src_img[ich*416*416+(y+ky)*416+(x+kx)] 
								* weight[och*3*11*11+ich*11*11+ky*11+kx];
						}
					}
				}

				dst_img[och*102*102 + dst_y*102 + dst_x] = tmp + bias[och];

				//printf("src(%d,%d) -> dst(%d,%d,%d)=%f+%f=%f\n", y,x,och,dst_y,dst_x,tmp,bias[och],tmp+bias[och]);
				//exit(0);

				dst_x++;
			}

			dst_y++;
		}

		//exit(-1);
	}
}
