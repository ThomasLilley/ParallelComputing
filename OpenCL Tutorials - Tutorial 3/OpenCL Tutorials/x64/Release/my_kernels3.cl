//flexible step reduce 
__kernel void reduce_add_2(__global const int* A, __global int* B) {
	int id = get_global_id(0);
	int N = get_global_size(0);

	B[id] = A[id];

	barrier(CLK_GLOBAL_MEM_FENCE);

	for (int i = 1; i < N; i *= 2) { //i is a stride
		if (!(id % (i * 2)) && ((id + i) < N))
			B[id] += B[id + i];

		barrier(CLK_GLOBAL_MEM_FENCE);
	}
}

__kernel void average_1(__global const int* A, __global int* B, __global int* C) {
	int id = get_global_id(0);
	int N = get_global_size(0);

	for (int i = 1; i < N; i++) {
		C[0] += (A[id + i] + B[id + i]);

		barrier(CLK_GLOBAL_MEM_FENCE); //wait for all threads to copy
	}
}