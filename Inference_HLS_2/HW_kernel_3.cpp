#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// uncomment ... C-simulation
// comment out ... HLS
#define HLS

#include "weight_l0.h"
#include "bias_l0.h"

#ifdef HLS
#include "ap_int.h"
#include "hls_half.h"
#include "hls_stream.h"
#include "ap_axi_sdata.h"

typedef float DTYPE;

struct int_s{
	int data;
	bool last;
};

#else
typedef float DTYPE;
#endif

void kernel(
#ifdef HLS
	hls::stream<int_s>& stream_in,
	hls::stream<int_s>& stream_out
#else
	DTYPE *stream_in,
	DTYPE *stream_out
#endif
);

// ---------------------------------------------------
// Convolution(3chx416x416) in the 1st layer
// ---------------------------------------------------
void kernel(
#ifdef HLS
	hls::stream<int_s>& stream_in,
	hls::stream<int_s>& stream_out
#else
	DTYPE *stream_in,
	DTYPE *stream_out
#endif
)
{
//(0): Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4)) // padding zero????
#pragma HLS INTERFACE axis port=stream_in
#pragma HLS INTERFACE axis port=stream_out
#pragma HLS INTERFACE s_axilite port=return

	// buffer
#ifdef HLS
	DTYPE in_img_buf[3*416*11];
	DTYPE out_fmap_buf[64*102*1];
#else
	DTYPE *in_img_buf   = (float *)malloc(3*416*11*sizeof(float));
	DTYPE *out_fmap_buf = (float *)malloc(64*102*1*sizeof(float));
#endif

	// -----------------------------------------
	// Convolutional operation
	// -----------------------------------------
	int dst_y;
	int dst_x;
	dst_y = 0;

	CONV_Y: for( int y = 0; y < 416 - 11; y += 4){
		// load the next data to a line buffer
#ifdef HLS
		int_s tmp_din, tmp_dout;
		int i;

		i = 0;
		LOAD_IMG: do{
#pragma HLS LOOP_TRIPCOUNT min=13728 max=13728 // 13728=416x3x11
			tmp_din = stream_in.read();
			in_img_buf[i] = (DTYPE)(tmp_din.data) / 1024.0; // (Y,X,CH)
			i++;
		}while( tmp_din.last == 0);
#else
		for( int buf_ch = 0; buf_ch < 3; buf_ch++){
			for( int tmp_y = 0; tmp_y < 11; tmp_y++){
				for( int buf_x = 0; buf_x < 416; buf_x++){
					in_img_buf[buf_ch*11*416 + tmp_y*416 + buf_x] 
						= stream_in[buf_ch*416*416 + (y+tmp_y)*416 + buf_x];
				}
			}
		}
#endif

		// perform convolution
		CONV_OCH: for( int och = 0; och < 64; och++){
			dst_x = 0;

			CONV_X: for( int x = 0; x < 416 - 11; x+= 4){
				float tmp = 0;

				CONV_ICH: for( int ich = 0; ich < 3; ich++){
					CONV_KY: for( int ky = 0; ky < 11; ky++){
						CONV_KX: for( int kx = 0; kx < 11; kx++){
							tmp += in_img_buf[ich*11*416+(ky)*416+(x+kx)]
								* weight_l0[och*3*11*11+ich*11*11+ky*11+kx];
						}
					}
				}
				out_fmap_buf[och*102 + dst_x] = tmp + bias_l0[och];

				dst_x++;
			}
		}

		// data write to the host processor
#ifdef HLS
		WB: for(int j = 0; j < 64*102; j++){
#pragma HLS pipeline
			tmp_dout.data = (int)(out_fmap_buf[j] * 1024.0);
			tmp_dout.last = 0;
			stream_out.write( tmp_dout);
		}
		tmp_dout.data = 0;
		tmp_dout.last = 1;
		stream_out.write( tmp_dout);
#else
		for( int och = 0; och < 64; och++){
			for( int x = 0; x < 102; x++){
				stream_out[och*102*102 + dst_y*102 + x] 
					= out_fmap_buf[och*102 + x];
			}
		}
#endif

		// update destination y address
		dst_y++;
	}
}
