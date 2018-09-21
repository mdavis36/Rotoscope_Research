/*
 3D PRW_bal CUDA
 Copyright Spatial Reasoning Group,
 Department of Computer Science, University of Oxford, 2017.
 All rights reserved.
*/

#include <stdio.h>


////////////////////////////////////////////////////////////////////////
// define kernel block size
////////////////////////////////////////////////////////////////////////

#define BLOCK_X 8
#define BLOCK_Y 8
#define BLOCK_Z 8
#define BLOCK_EX (BLOCK_X+2)
#define BLOCK_EY (BLOCK_Y+2)
#define BLOCK_EZ (BLOCK_Z+2)
#define BLOCK_EXY ((BLOCK_EX)*(BLOCK_EY))
#define BLOCK ((BLOCK_EX)*(BLOCK_EY)*(BLOCK_EZ))

#define RR 6


__device__ bool change = false;


__global__ void initialise_state(int NX, int NY, int NZ,
						const unsigned char *d_array,
										int *d_label,
										int *d_state)
{
	// pixel has lower neighbour - 0, is local minimum - 1,
	// some neighbours equal and some higher - 2 or 3

	int i, j, k, ind, NXY,
		min, state,
		nind[6];
	unsigned char minarray;
	bool ncond[6];

	// define global indices
	i = threadIdx.x + blockIdx.x*blockDim.x;
	j = threadIdx.y + blockIdx.y*blockDim.y;
	k = threadIdx.z + blockIdx.z*blockDim.z;

	if (i < NX && j < NY && k < NZ) // && i >= 0 && j >= 0 && k >= 0
	{
		NXY = NX*NY;

		ind = i + j*NX + k*NXY;

		// initialise neighbour indices and conditions
		nind[0] = ind - NXY;	ncond[0] = k > 0;
		nind[1] = ind - NX;		ncond[1] = j > 0;
		nind[2] = ind - 1;		ncond[2] = i > 0;
		nind[3] = ind + 1;		ncond[3] = i + 1 < NX;
		nind[4] = ind + NX;		ncond[4] = j + 1 < NY;
		nind[5] = ind + NXY;	ncond[5] = k + 1 < NZ;

		// define initial state
		minarray = d_array[ind];
		min = ind;
		state = 1; // assume local minimum

		for (int n = 0; n < 6; ++n)
			if (ncond[n] && d_array[nind[n]] <= minarray)
			{
				minarray = d_array[nind[n]];
				min = nind[n];
			}
		if (minarray < d_array[ind]) // lower neighbour
			state = 0;
		else if (min > ind) // equal neighbour with larger index, no lower neighbour
			state = 2;
		else if (min < ind) // equal neighbour with smaller index, no lower neighbour
		{
			state = 3;
			min = ind;
		}

		d_label[ind] = min;
		d_state[ind] = state;
	}
}



__global__ void resolve_plateaux(int NX, int NY, int NZ,
						const unsigned char *d_array,
										int *d_label,
								  const int *d_state,
										int *d_newstate)
{
	// resolve non-minimal plateaux changing states from 2, 3 to 0

	int i, j, k, si, sj, sk, ind, sind, NXY,
		label, state, nstate,
		nind[6], snind[6];
	bool valid,
		ncond[6], sbcond[6];

	__shared__ unsigned char s_array[BLOCK];
	__shared__ int s_state[BLOCK];
	__shared__ bool s_change;

	// define global indices
	i = threadIdx.x + blockIdx.x*blockDim.x;
	j = threadIdx.y + blockIdx.y*blockDim.y;
	k = threadIdx.z + blockIdx.z*blockDim.z;
	si = threadIdx.x + 1;
	sj = threadIdx.y + 1;
	sk = threadIdx.z + 1;

	valid = i < NX && j < NY && k < NZ; // && i >= 0 && j >= 0 && k >= 0

	if (valid)
	{
		NXY = NX*NY;

		ind = i + j*NX + k*NXY;
		sind = si + sj*BLOCK_EX + sk*BLOCK_EXY;

		// initialise neighbour indices and conditions
		nind[0] = ind - NXY;			ncond[0] = k > 0;
		nind[1] = ind - NX;				ncond[1] = j > 0;
		nind[2] = ind - 1;				ncond[2] = i > 0;
		nind[3] = ind + 1;				ncond[3] = i + 1 < NX;
		nind[4] = ind + NX;				ncond[4] = j + 1 < NY;
		nind[5] = ind + NXY;			ncond[5] = k + 1 < NZ;
		snind[0] = sind - BLOCK_EXY;	sbcond[0] = sk == 1;
		snind[1] = sind - BLOCK_EX;		sbcond[1] = sj == 1;
		snind[2] = sind - 1;			sbcond[2] = si == 1;
		snind[3] = sind + 1;			sbcond[3] = si == BLOCK_X;
		snind[4] = sind + BLOCK_EX;		sbcond[4] = sj == BLOCK_Y;
		snind[5] = sind + BLOCK_EXY;	sbcond[5] = sk == BLOCK_Z;

		label = d_label[ind];

		// copy data into shared memory, including extra border
		s_array[sind] = d_array[ind];
		s_state[sind] = d_state[ind];
		for (int n = 0; n < 6; ++n)
			if (sbcond[n] && ncond[n])
			{
				s_array[snind[n]] = d_array[nind[n]];
				s_state[snind[n]] = d_state[nind[n]];
			}
	}

	do
	{
		__syncthreads();
		s_change = false;
		__syncthreads();

		if (valid)
		{
			state = s_state[sind];

			if (state >= 2 || state < 0)
			{
				for (int n = 0; n < 6; ++n)
				{
					nstate = s_state[snind[n]] - 1;
					if (ncond[n] && nstate < 0 && s_array[snind[n]] == s_array[sind] && (state >= 2 || state < nstate))
					{
						state = nstate;
						label = nind[n];
					}
				}
				// if state changed update shared and global change variables
				if (state != s_state[sind])
				{
					s_change = true;
					change = true;
				}
			}
		}

		__syncthreads();
		if (valid)
			s_state[sind] = state;
	} while (s_change);

	if (valid)
	{
		d_label[ind] = label;
		d_newstate[ind] = s_state[sind];
	}
}



__global__ void propagate_labels(int NX, int NY, int NZ,
								int *d_label)
{
	// propagate labels uphill updating current from its label

	int i, j, k, ind,
		newlab, oldlab;

	// define global indices
	i = threadIdx.x + blockIdx.x*blockDim.x;
	j = threadIdx.y + blockIdx.y*blockDim.y;
	k = threadIdx.z + blockIdx.z*blockDim.z;

	if (i < NX && j < NY && k < NZ) // && i >= 0 && j >= 0 && k >= 0
	{
		ind = i + j*NX + k*NX*NY;

		oldlab = ind;
		newlab = d_label[ind];
		for (int n = 0; newlab != oldlab && n < RR; ++n)
		{
			oldlab = newlab;
			newlab = d_label[oldlab];
		}
		d_label[ind] = newlab;
		if (newlab != oldlab)
			change = true;
	}
}



__global__ void unify_plateau_labels(int NX, int NY, int NZ,
									  int *d_label,
								const int *d_state)
{
	// unify multiple labels within minimal plateaux

	int i, j, k, ind, NXY,
		indlab, neilab,
		nind[6];
	bool ncond[6];

	// define global indices
	i = threadIdx.x + blockIdx.x*blockDim.x;
	j = threadIdx.y + blockIdx.y*blockDim.y;
	k = threadIdx.z + blockIdx.z*blockDim.z;

	if (i < NX && j < NY && k < NZ) // && i >= 0 && j >= 0 && k >= 0
	{
		NXY = NX*NY;

		ind = i + j*NX + k*NXY;

		if (d_state[ind] >= 2)
		{
			// initialise neighbour indices and conditions
			nind[0] = ind - NXY;	ncond[0] = k > 0;
			nind[1] = ind - NX;		ncond[1] = j > 0;
			nind[2] = ind - 1;		ncond[2] = i > 0;
			nind[3] = ind + 1;		ncond[3] = i + 1 < NX;
			nind[4] = ind + NX;		ncond[4] = j + 1 < NY;
			nind[5] = ind + NXY;	ncond[5] = k + 1 < NZ;

			for (int n = 0; n < 6; ++n)
				if (ncond[n] && d_state[nind[n]] >= 2)
				{
					indlab = d_label[ind];
					neilab = d_label[nind[n]];
					while (d_label[indlab] != d_label[neilab])
					{
						atomicMin(&d_label[indlab], d_label[neilab]);
						atomicMin(&d_label[neilab], d_label[indlab]);
						change = true;
					}
				}
		}
		while (d_label[ind] != d_label[d_label[ind]])
			d_label[ind] = d_label[d_label[ind]];
	}
}




// 'h_' prefix - CPU (host) memory space
unsigned char *h_array;
int *h_label;

int	NX = 1, NY = 1, NZ = 1, NS;


void mywatershed()
{
	// 'h_' prefix - CPU (host) memory space
	int bx, by, bz;
//	int counter;
	bool h_change;

	// 'd_' prefix - GPU (device) memory space
	unsigned char *d_array;
	int *d_label;
	int *d_state, *d_new, *d_temp;


	// initialise CUDA timing
	float milli;
	cudaEvent_t overallstart, stop;
	cudaEventCreate(&overallstart);
//	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// start the overall timer
	cudaEventRecord(overallstart);

	// allocate memory for arrays
	cudaMalloc((void **)&d_array,	NS);
	cudaMalloc((void **)&d_label,	sizeof(int)*NS);
	cudaMalloc((void **)&d_state,	sizeof(int)*NS);
	cudaMalloc((void **)&d_new,		sizeof(int)*NS);


	// copy array to device
//	cudaEventRecord(start);
	cudaMemcpy(d_array, h_array, NS, cudaMemcpyHostToDevice);
//	cudaEventRecord(stop);
//	cudaEventSynchronize(stop);
//	cudaEventElapsedTime(&milli, start, stop);
//	printf("\nCopy array to device: %.3f (ms) \n", milli);

	// Set up the execution configuration
	bx = 1 + (NX-1)/BLOCK_X;
	by = 1 + (NY-1)/BLOCK_Y;
	bz = 1 + (NZ-1)/BLOCK_Z;

	dim3 dimGrid(bx,by,bz);
	dim3 dimBlock(BLOCK_X,BLOCK_Y,BLOCK_Z);

//	printf("\n dimGrid = %d %d %d \n",dimGrid.x,dimGrid.y,dimGrid.z);
//	printf(" dimBlock = %d %d %d \n",dimBlock.x,dimBlock.y,dimBlock.z);


	// Execute GPU kernel to initialise state
//	cudaEventRecord(start);

	initialise_state<<<dimGrid, dimBlock>>>(NX, NY, NZ, d_array, d_label, d_state);

//	cudaEventRecord(stop);
//	cudaEventSynchronize(stop);
//	cudaEventElapsedTime(&milli, start, stop);
//	printf("initialise_state: %.3f (ms) \n", milli);


	// Execute GPU kernel to resolve plateaux
//	cudaEventRecord(start);

//	counter = 0;
	h_change = true;
	while (h_change)
	{
//		++counter;
		h_change = false;
		cudaMemcpyToSymbol(change, &h_change, sizeof(h_change));

		resolve_plateaux<<<dimGrid, dimBlock>>>(NX, NY, NZ, d_array, d_label, d_state, d_new);

		d_temp = d_state; d_state = d_new; d_new = d_temp;
		cudaMemcpyFromSymbol(&h_change, change, sizeof(change));
	}

//	cudaEventRecord(stop);
//	cudaEventSynchronize(stop);
//	cudaEventElapsedTime(&milli, start, stop);
//	printf("resolve_plateaux: %.3f (ms) \n", milli);
//	printf("resolve_plateaux number of iterations = %d\n", counter);


	// Execute GPU kernel to propagate labels (run the CA iterations)
//	cudaEventRecord(start);

//	counter = 0;
	h_change = true;
	while (h_change)
	{
//		++counter;
		h_change = false;
		cudaMemcpyToSymbol(change, &h_change, sizeof(h_change));

		propagate_labels<<<dimGrid, dimBlock>>>(NX, NY, NZ, d_label);

		cudaMemcpyFromSymbol(&h_change, change, sizeof(change));
	}

//	cudaEventRecord(stop);
//	cudaEventSynchronize(stop);
//	cudaEventElapsedTime(&milli, start, stop);
//	printf("propagate_labels: %.3f (ms) \n", milli);
//	printf("propagate_labels number of iterations = %d\n", counter);


	// Execute GPU kernel to unify minimal plateau labels
//	cudaEventRecord(start);

//	counter = 0;
	h_change = true;
	while (h_change)
	{
//		++counter;
		h_change = false;
		cudaMemcpyToSymbol(change, &h_change, sizeof(h_change));

		unify_plateau_labels<<<dimGrid, dimBlock>>>(NX, NY, NZ, d_label, d_state);

		cudaMemcpyFromSymbol(&h_change, change, sizeof(change));
	}

//	cudaEventRecord(stop);
//	cudaEventSynchronize(stop);
//	cudaEventElapsedTime(&milli, start, stop);
//	printf("unify_plateau_labels: %.3f (ms) \n", milli);
//	printf("unify_plateau_labels number of iterations = %d\n", counter);


	// Read back GPU results
//	cudaEventRecord(start);
	cudaMemcpy(h_label, d_label, sizeof(int)*NS, cudaMemcpyDeviceToHost);
//	cudaEventRecord(stop);
//	cudaEventSynchronize(stop);
//	cudaEventElapsedTime(&milli, start, stop);
//	printf("\nCopy label to host: %.3f (ms) \n", milli);


	// Release GPU memory
	cudaFree(d_array);
	cudaFree(d_label);
	cudaFree(d_state);
	cudaFree(d_new);

	// stop the overall timer
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milli, overallstart, stop);
	printf("\n\nOverall execution time: %.3f (ms) \n", milli);
}



int main(int argc, const char **argv)
{
	NX = 10;
	NY = 10;
	NS = NX*NY*NZ;

	// allocate memory for label array
	h_label = (int *)malloc(sizeof(int)*NS);
	h_array = (unsigned char *)malloc(NS);

	for (int i = 0; i < NS; ++i)
		if (i == 11 || i == 18 || i == 81 || i == 88)
			h_array[i] = 0;
		else
			h_array[i] = 1;

	mywatershed();

	for (int j = 0; j < NY; ++j)
	{
		for (int i = 0; i < NX; ++i)
			printf("%d ", h_label[i + j*NX]);
		printf("\n");
	}

	// Release CPU memory
	free(h_array);
	free(h_label);

	return 0;
}
