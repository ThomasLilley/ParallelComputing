__kernel void add(__global const int* A, __global const int* B, __global int* C) {
	int id = get_global_id(0);
	C[id] = A[id] + B[id];
}

__kernel void minimum(__global const int* A, __global const int* B, __global int* C) {
	int id = get_global_id(0);

	barrier(CLK_GLOBAL_MEM_FENCE);

	if (A[id] > B[id]) {
		C[id] = B[id];
	}
	else {
		C[id] = A[id];
	}

}

__kernel void maximum(__global const int* A, __global const int* B, __global int* C) {
	int id = get_global_id(0);

	if (A[id] > B[id]) {
		C[id] = A[id];
	}
	else {
		C[id] = B[id];
	}

}
