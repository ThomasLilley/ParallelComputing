__kernel void add(__global const int* A, __global const int* B, __global int* C) {
	int id = get_global_id(0);
	C[id] = A[id] + B[id];
}

__kernel void minimum(__global const int* A, __global const int* B, __global int* C) {
	int id = get_global_id(0);
	int minima = 100;
	int minimb = 100;

	barrier(CLK_GLOBAL_MEM_FENCE);

	for(int i = 0; i <  936554; i++)
{ 
		if (A[id] < minima) {
			minima = B[id];
		}
}
	barrier(CLK_GLOBAL_MEM_FENCE);

	for (int i = 0; i <  936554; i++)
	{
		if (B[id] < minimb) {
			minimb = B[id];
		}
	}

	barrier(CLK_GLOBAL_MEM_FENCE);

	if (minima < minimb){ 
		C[id] = minimb;
	}
	else{ 
		C[id] = minima;
	}

}

__kernel void maximum(__global const int* A, __global const int* B, __global int* C) {
	int id = get_global_id(0);
		int maxima = 100;
		int maximb = 100;

		barrier(CLK_GLOBAL_MEM_FENCE);

		for (int i = 0; i < 936554; i++)
		{
			if (A[id] > maxima) {
				maxima = A[id];
			}
		}
		barrier(CLK_GLOBAL_MEM_FENCE);

		for (int i = 0; i < 936554; i++)
		{
			if (B[id] > maximb) {
				maximb = B[id];
			}
		}

		barrier(CLK_GLOBAL_MEM_FENCE);

		if (maxima < maximb) {
			C[id] = maximb;
		}
		else {
			C[id] = maxima;
		}

}
