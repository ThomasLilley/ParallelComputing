__kernel void add(__global const int* A, __global const int* B, __global int* C) {
	int id = get_global_id(0);
	//Add vector A to vector B in parallel and return the added vector to C
	C[id] = A[id] + B[id];
}

__kernel void minimum(__global const int* A, __global const int* B, __global int* C) {
	int id = get_global_id(0);

	//compare all of vector A with vector B and place the minimum values from each into vector C
	if (A[id] > B[id]) {
		C[id] = B[id];
	}
	else {
		C[id] = A[id];
	}

}

__kernel void maximum(__global const int* A, __global const int* B, __global int* C) {
	int id = get_global_id(0);

	//compare all of vector A with vector B and place the maximum values from each into vector C

	if (A[id] > B[id]) {
		C[id] = A[id];
	}
	else {
		C[id] = B[id];
	}

}
