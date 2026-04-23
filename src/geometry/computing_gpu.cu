/*
 * TriDist.cu
 *
 *  Created on: Oct 22, 2022
 *      Author: teng
 */

#include <cuda_runtime.h>
#include "geometry.h"
#include "cuda_util.cuh"

#include <cub/block/block_reduce.cuh>
#include <cub/cub.cuh>

#include <future>

#define THID threadIdx.x
#define GLTHID (blockIdx.x * blockDim.x + threadIdx.x)
#define WARP_SIZE 32
#define WARPID (THID >> 5)
#define GLWARPID (GLTHID >> 5)
#define LANEID (THID & 31)

namespace tdbase
{

	GD gd; 

	cudaStream_t refine_streams[2];
	cudaEvent_t memcpy_done[2];


	static inline size_t align_up(size_t x, size_t a) {
		return (x + a - 1) & ~(a - 1);
	}

	void initialize_refine_stream() {
		cudaStreamCreate(&refine_streams[0]);
		cudaStreamCreate(&refine_streams[1]);

		cudaEventCreate(&memcpy_done[0]);
		cudaEventCreate(&memcpy_done[1]);
	}

	void * allocate_pinned_memory(size_t size)
	{
		char* h_ptr;
		cudaHostAlloc((void**)&h_ptr, size, cudaHostAllocDefault);
		return h_ptr;
	}

	void free_pinned_memory(void *ptr)
	{
		cudaFreeHost(ptr);
	}

	void synchronize_gpu(){
		check_execution();
		CUDA_SAFE_CALL(cudaDeviceSynchronize());
	}

	void synchronize_stream(int stream_id){
		cudaStreamSynchronize(refine_streams[stream_id]);
	}

	void synchronize_event(int stream_id) {
		cudaEventSynchronize(memcpy_done[stream_id]);
	}

	void inclusive_scan(const size_t* d_in,
							size_t*       d_out, 
							size_t        N)
	{
		cudaMemset(d_out, 0, sizeof(size_t));

		void*  d_temp = nullptr;
		size_t temp_bytes = 0;

		cub::DeviceScan::InclusiveSum(d_temp, temp_bytes, d_in, d_out + 1, (int)N);

		cudaMalloc(&d_temp, temp_bytes);

		cub::DeviceScan::InclusiveSum(d_temp, temp_bytes, d_in, d_out + 1, (int)N);

		cudaFree(d_temp);
	}

	__global__ void fill_count_and_prefix_kernel(int* valid_voxel_cnt, 
													size_t* valid_voxel_prefix,
													const size_t* chunk_count,
													const size_t* chunk_valid_prefix,
													size_t pair_begin,
													size_t chunk_pair_num
												)
	{
		size_t i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= chunk_pair_num) return;

		size_t global_valid_base = valid_voxel_prefix[pair_begin];

		valid_voxel_cnt[pair_begin + i] = chunk_count[i];
		valid_voxel_prefix[pair_begin + i + 1] = global_valid_base + chunk_valid_prefix[i + 1];
	}

	__global__ void evaluate_knn_kernel(
		float* min_dist,
		float* max_dist,
		int* prefix,
		int* confirmed,
		int* status,
		int target_count,
		int K
	)
	{
		int target = blockIdx.x;
		if (target >= target_count) return;

		int begin = prefix[target];
		int end   = prefix[target+1];
		int m = end - begin;

		int cand_left = K - confirmed[target];

		for (int i = threadIdx.x; i < m; i += blockDim.x)
		{
			int gi = begin + i;

			int sure = 0;
			int maybe = 0;

			for (int j = 0; j < m; j++)
			{
				if (i == j) continue;

				int gj = begin + j;

				if (min_dist[gi] >= max_dist[gj])
					sure++;

				if (!(max_dist[gi] <= min_dist[gj]))
					maybe++;
			}

			if (maybe < cand_left)
				status[gi] = 1;      // CONFIRMED
			else if (sure >= cand_left)
				status[gi] = 2;      // REMOVE
			else
				status[gi] = 0;      // KEEP
		}
	}

	/*
	 *
	 * get the closest points between segments
	 *
	 * */
	__device__ inline void
	SegPoints(float VEC[3],
			  float X[3], float Y[3],			  // closest points
			  const float P[3], const float A[3], // seg 1 origin, vector
			  const float Q[3], const float B[3]) // seg 2 origin, vector
	{
		float T[3], A_dot_A, B_dot_B, A_dot_B, A_dot_T, B_dot_T;
		float TMP[3];

		VmV_d(T, Q, P);
		A_dot_A = VdotV_d(A, A);
		B_dot_B = VdotV_d(B, B);
		A_dot_B = VdotV_d(A, B);
		A_dot_T = VdotV_d(A, T);
		B_dot_T = VdotV_d(B, T);
		assert(A_dot_A != 0 && B_dot_B != 0);

		// t parameterizes ray P,A
		// u parameterizes ray Q,B

		float t, u;

		// compute t for the closest point on ray P,A to
		// ray Q,B

		float denom = A_dot_A * B_dot_B - A_dot_B * A_dot_B;
		if (denom == 0)
		{
			t = 0;
		}
		else
		{
			t = (A_dot_T * B_dot_B - B_dot_T * A_dot_B) / denom;
		}

		// find u for point on ray Q,B closest to point at t
		if (B_dot_B == 0)
		{
			u = 0;
		}
		else
		{
			u = (t * A_dot_B - B_dot_T) / B_dot_B;
		}

		// if u is on segment Q,B, t and u correspond to
		// closest points, otherwise, recompute and
		// clamp t
		if (u <= 0)
		{
			VcV_d(Y, Q);
			if (A_dot_A == 0)
			{
				t = 0;
			}
			else
			{
				t = A_dot_T / A_dot_A;
			}

			if (t <= 0)
			{
				VcV_d(X, P);
				VmV_d(VEC, Q, P);
			}
			else if (t >= 1)
			{
				VpV_d(X, P, A);
				VmV_d(VEC, Q, X);
			}
			else
			{
				VpVxS_d(X, P, A, t);
				VcrossV_d(TMP, T, A);
				VcrossV_d(VEC, A, TMP);
			}
		}
		else if (u >= 1)
		{
			VpV_d(Y, Q, B);
			if (A_dot_A == 0)
			{
				t = 0;
			}
			else
			{
				t = (A_dot_B + A_dot_T) / A_dot_A;
			}

			if (t <= 0)
			{
				VcV_d(X, P);
				VmV_d(VEC, Y, P);
			}
			else if (t >= 1)
			{
				VpV_d(X, P, A);
				VmV_d(VEC, Y, X);
			}
			else
			{
				VpVxS_d(X, P, A, t);
				VmV_d(T, Y, P);
				VcrossV_d(TMP, T, A);
				VcrossV_d(VEC, A, TMP);
			}
		}
		else
		{ // on segment

			VpVxS_d(Y, Q, B, u);

			if (t <= 0)
			{
				VcV_d(X, P);
				VcrossV_d(TMP, T, B);
				VcrossV_d(VEC, B, TMP);
			}
			else if (t >= 1)
			{
				VpV_d(X, P, A);
				VmV_d(T, Q, X);
				VcrossV_d(TMP, T, B);
				VcrossV_d(VEC, B, TMP);
			}
			else
			{ // 0<=t<=1
				VpVxS_d(X, P, A, t);
				VcrossV_d(VEC, A, B);
				if (VdotV_d(VEC, T) < 0)
				{
					VxS_d(VEC, VEC, -1);
				}
			}
		}
	}

	/*
	 * check the segments of a triangle to see if the closest
	 * points can be found on a segment pair, which covers
	 * almost all cases
	 * */
	__device__ inline float
	TriDist_seg(const float *S, const float *T,
				bool &shown_disjoint, bool &closest_find)
	{

		// closest points
		float P[3];
		float Q[3];

		// some temporary vectors
		float V[3];
		float Z[3];
		// Compute vectors along the 6 sides
		float Sv[3][3], Tv[3][3];

		VmV_d(Sv[0], S + 3, S);
		VmV_d(Sv[1], S + 6, S + 3);
		VmV_d(Sv[2], S, S + 6);

		VmV_d(Tv[0], T + 3, T);
		VmV_d(Tv[1], T + 6, T + 3);
		VmV_d(Tv[2], T, T + 6);

		// For each edge pair, the vector connecting the closest points
		// of the edges defines a slab (parallel planes at head and tail
		// enclose the slab). If we can show that the off-edge vertex of
		// each triangle is outside of the slab, then the closest points
		// of the edges are the closest points for the triangles.
		// Even if these tests fail, it may be helpful to know the closest
		// points found, and whether the triangles were shown disjoint

		float mindd = DBL_MAX; // Set first minimum safely high
		float VEC[3];
		for (int i = 0; i < 3; i++)
		{
			for (int j = 0; j < 3; j++)
			{
				// Find closest points on edges i & j, plus the
				// vector (and distance squared) between these points
				SegPoints(VEC, P, Q, S + i * 3, Sv[i], T + j * 3, Tv[j]);
				VmV_d(V, Q, P);
				float dd = VdotV_d(V, V);
				if (dd <= mindd)
				{
					mindd = dd;

					// Verify this closest point pair for the segment pairs with minimum distance
					VmV_d(Z, S + ((i + 2) % 3) * 3, P);
					float a = VdotV_d(Z, VEC);
					VmV_d(Z, T + ((j + 2) % 3) * 3, Q);
					float b = VdotV_d(Z, VEC);

					// the closest distance of segment pairs is the closest distance of the two triangles
					if ((a <= 0) && (b >= 0))
					{
						closest_find = true;
						return sqrt(mindd);
					}

					// otherwise, check the other cases
					// we can use the side product of this calculation
					// to judge whether two triangle joint or not
					float p = VdotV_d(V, VEC);
					if (a < 0)
						a = 0;
					if (b > 0)
						b = 0;
					if ((p - a + b) > 0)
						shown_disjoint = true;
				}
			}
		}

		// not sure is the case
		closest_find = false;
		return mindd;
	}

	/*
	 * any other cases for triangle distance calculation besides the
	 * closest points reside on segments
	 *
	 * */
	__device__ inline float
	TriDist_other(const float *S, const float *T, bool &shown_disjoint)
	{

		// closest points
		float P[3];
		float Q[3];

		// some temporary vectors
		float V[3];
		float Z[3];
		// Compute vectors along the 6 sides
		float Sv[3][3], Tv[3][3];

		VmV_d(Sv[0], S + 3, S);
		VmV_d(Sv[1], S + 6, S + 3);
		VmV_d(Sv[2], S, S + 6);

		VmV_d(Tv[0], T + 3, T);
		VmV_d(Tv[1], T + 6, T + 3);
		VmV_d(Tv[2], T, T + 6);

		// First check for case 1

		float Sn[3], Snl;
		VcrossV_d(Sn, Sv[0], Sv[1]); // Compute normal to S triangle
		Snl = VdotV_d(Sn, Sn);		 // Compute square of length of normal

		// If cross product is long enough,

		if (Snl > 1e-15)
		{
			// Get projection lengths of T points

			float Tp[3];

			VmV_d(V, S, T);
			Tp[0] = VdotV_d(V, Sn);

			VmV_d(V, S, T + 3);
			Tp[1] = VdotV_d(V, Sn);

			VmV_d(V, S, T + 6);
			Tp[2] = VdotV_d(V, Sn);

			// If Sn is a separating direction,
			// find point with smallest projection

			int point = -1;
			if ((Tp[0] > 0) && (Tp[1] > 0) && (Tp[2] > 0))
			{
				if (Tp[0] < Tp[1])
					point = 0;
				else
					point = 1;
				if (Tp[2] < Tp[point])
					point = 2;
			}
			else if ((Tp[0] < 0) && (Tp[1] < 0) && (Tp[2] < 0))
			{
				if (Tp[0] > Tp[1])
					point = 0;
				else
					point = 1;
				if (Tp[2] > Tp[point])
					point = 2;
			}

			// If Sn is a separating direction,

			if (point >= 0)
			{
				shown_disjoint = true;
				// Test whether the point found, when projected onto the
				// other triangle, lies within the face.

				VmV_d(V, T + point * 3, S);
				VcrossV_d(Z, Sn, Sv[0]);
				if (VdotV_d(V, Z) > 0)
				{
					VmV_d(V, T + point * 3, S + 3);
					VcrossV_d(Z, Sn, Sv[1]);
					if (VdotV_d(V, Z) > 0)
					{
						VmV_d(V, T + point * 3, S + 6);
						VcrossV_d(Z, Sn, Sv[2]);
						if (VdotV_d(V, Z) > 0)
						{
							// T[point] passed the test - it's a closest point for
							// the T triangle; the other point is on the face of S

							VpVxS_d(P, T + point * 3, Sn, Tp[point] / Snl);
							VcV_d(Q, T + point * 3);
							return sqrt(VdistV2_d(P, Q));
						}
					}
				}
			}
		}

		float Tn[3], Tnl;
		VcrossV_d(Tn, Tv[0], Tv[1]);
		Tnl = VdotV_d(Tn, Tn);

		if (Tnl > 1e-15)
		{

			float Sp[3];
			VmV_d(V, T, S);
			Sp[0] = VdotV_d(V, Tn);

			VmV_d(V, T, S + 3);
			Sp[1] = VdotV_d(V, Tn);

			VmV_d(V, T, S + 6);
			Sp[2] = VdotV_d(V, Tn);

			int point = -1;
			if ((Sp[0] > 0) &&
				(Sp[1] > 0) && (Sp[2] > 0))
			{
				if (Sp[0] < Sp[1])
					point = 0;
				else
					point = 1;
				if (Sp[2] < Sp[point])
					point = 2;
			}
			else if ((Sp[0] < 0) &&
					 (Sp[1] < 0) && (Sp[2] < 0))
			{
				if (Sp[0] > Sp[1])
					point = 0;
				else
					point = 1;
				if (Sp[2] > Sp[point])
					point = 2;
			}

			if (point >= 0)
			{
				shown_disjoint = true;

				VmV_d(V, S + 3 * point, T);
				VcrossV_d(Z, Tn, Tv[0]);
				if (VdotV_d(V, Z) > 0)
				{
					VmV_d(V, S + 3 * point, T + 3);
					VcrossV_d(Z, Tn, Tv[1]);
					if (VdotV_d(V, Z) > 0)
					{
						VmV_d(V, S + 3 * point, T + 6);
						VcrossV_d(Z, Tn, Tv[2]);
						if (VdotV_d(V, Z) > 0)
						{
							VcV_d(P, S + 3 * point);
							VpVxS_d(Q, S + 3 * point, Tn, Sp[point] / Tnl);
							return sqrt(VdistV2_d(P, Q));
						}
					}
				}
			}
		}

		// not the case
		return -1;
	}

	//--------------------------------------------------------------------------
	// TriDist()
	//
	// Computes the closest points on two triangles, and returns the
	// distance between them.
	//
	// S and T are the triangles, stored tri[point][dimension].
	//
	// If the triangles are disjoint, P and Q give the closest points of
	// S and T respectively. However, if the triangles overlap, P and Q
	// are basically a random pair of points from the triangles, not
	// coincident points on the intersection of the triangles, as might
	// be expected.
	//--------------------------------------------------------------------------

	__device__ float
	TriDist_kernel(const float *S, const float *T)
	{
		bool shown_disjoint = false;
		bool closest_find = false;
		float mindd_seg = TriDist_seg(S, T, shown_disjoint, closest_find);
		if (closest_find)
		{ // the closest points are one segments, simply return
			return mindd_seg;
		}
		else
		{
			// No edge pairs contained the closest points.
			// either:
			// 1. one of the closest points is a vertex, and the
			//    other point is interior to a face.
			// 2. the triangles are overlapping.
			// 3. an edge of one triangle is parallel to the other's face. If
			//    cases 1 and 2 are not true, then the closest points from the 9
			//    edge pairs checks above can be taken as closest points for the
			//    triangles.
			// 4. possibly, the triangles were degenerate.  When the
			//    triangle points are nearly colinear or coincident, one
			//    of above tests might fail even though the edges tested
			//    contain the closest points.
			float mindd_other = TriDist_other(S, T, shown_disjoint);
			if (mindd_other != -1)
			{ // is the case
				return mindd_other;
			}
		}

		// Case 1 can't be shown.
		// If one of these tests showed the triangles disjoint,
		// we assume case 3 or 4, otherwise we conclude case 2,
		// that the triangles overlap.
		if (shown_disjoint)
		{
			return sqrt(mindd_seg);
		}
		else
		{
			return 0;
		}
	}

	__device__ static float atomicMin(float *address, float val)
	{
		int *address_as_i = (int *)address;
		int old = *address_as_i, assumed;
		do
		{
			assumed = old;
			old = ::atomicCAS(address_as_i, assumed,
							  __float_as_int(::fminf(val, __int_as_float(assumed))));
		} while (assumed != old);
		return __int_as_float(old);
	}

	__global__ void TriDist_cuda(const float *data, const size_t *offset_size, const float *hausdorff, result_container *dist, uint32_t tri_offset_1, uint32_t tri_offset_2_start)
	{
		// which batch
		int pair_id = blockIdx.x;
		int tri_offset_2 = threadIdx.x + tri_offset_2_start;

		if (tri_offset_1 >= offset_size[pair_id * 4 + 1])
		{
			return;
		}
		if (tri_offset_2 >= offset_size[pair_id * 4 + 3])
		{
			return;
		}
		size_t obj_offset1 = offset_size[pair_id * 4];
		size_t obj_offset2 = offset_size[pair_id * 4 + 2];

		const float *cur_S = data + 9 * (obj_offset1 + tri_offset_1);
		const float *cur_T = data + 9 * (obj_offset2 + tri_offset_2);
		float dd = TriDist_kernel(cur_S, cur_T);

		float phdist1 = *(hausdorff + 2 * (obj_offset1 + tri_offset_1));
		float phdist2 = *(hausdorff + 2 * (obj_offset2 + tri_offset_2));
		float hdist1 = *(hausdorff + 2 * (obj_offset1 + tri_offset_1) + 1);
		float hdist2 = *(hausdorff + 2 * (obj_offset2 + tri_offset_2) + 1);

		float low_dist = max(dd - phdist1 - phdist2, (float)0.0);
		float high_dist = dd + hdist1 + hdist2;

		if (tri_offset_1 == 0 && tri_offset_2 == 0)
		{
			dist[pair_id].distance = dd;
			dist[pair_id].p1 = tri_offset_1;
			dist[pair_id].p2 = tri_offset_2;
			dist[pair_id].min_dist = low_dist;
			dist[pair_id].max_dist = high_dist;
			return;
		}

		atomicMin(&(dist[pair_id].distance), dd);

		if (dist[pair_id].distance == dd)
		{
			dist[pair_id].p1 = tri_offset_1;
			dist[pair_id].p2 = tri_offset_2;
		}

		atomicMin(&(dist[pair_id].min_dist), low_dist);
		atomicMin(&(dist[pair_id].max_dist), high_dist);

		//	printf("%d %d:\t%f %f | %d %d:\t%f %f | %f %f %f | %f %f\n", obj_offset1, tri_offset_1, phdist1, hdist1, obj_offset2, tri_offset_2, phdist2, hdist2, low_dist, high_dist, dd, dist[pair_id].min_dist, dist[pair_id].max_dist);
	}

	/**
	 * Warp-level optimization.
	 */
	__global__ void TriDist_cuda_op_warp_level(const float *data, const uint32_t *offset_size, const float *hausdorff,
											   result_container *dist, uint32_t max_size_1, uint32_t tri_offset_2_start)
	{
		// which batch
		int pair_id = GLWARPID;
		int tri_offset_2 = LANEID + tri_offset_2_start;

		for (int tri_offset_1 = 0; tri_offset_1 < max_size_1; tri_offset_1++)
		{

			if (tri_offset_1 >= offset_size[pair_id * 4 + 1])
			{
				continue;
			}
			if (tri_offset_2 >= offset_size[pair_id * 4 + 3])
			{
				continue;
			}
			uint32_t obj_offset1 = offset_size[pair_id * 4];
			uint32_t obj_offset2 = offset_size[pair_id * 4 + 2];

			const float *cur_S = data + 9 * (obj_offset1 + tri_offset_1);
			const float *cur_T = data + 9 * (obj_offset2 + tri_offset_2);
			float dd = TriDist_kernel(cur_S, cur_T);

			float phdist1 = *(hausdorff + 2 * (obj_offset1 + tri_offset_1));
			float phdist2 = *(hausdorff + 2 * (obj_offset2 + tri_offset_2));
			float hdist1 = *(hausdorff + 2 * (obj_offset1 + tri_offset_1) + 1);
			float hdist2 = *(hausdorff + 2 * (obj_offset2 + tri_offset_2) + 1);

			float low_dist = max(dd - phdist1 - phdist2, (float)0.0);
			float high_dist = dd + hdist1 + hdist2;

			// if(tri_offset_1==0&&tri_offset_2==0){
			// 	dist[pair_id].distance = dd;
			// 	dist[pair_id].p1 = tri_offset_1;
			// 	dist[pair_id].p2 = tri_offset_2;
			// 	dist[pair_id].min_dist = low_dist;
			// 	dist[pair_id].max_dist = high_dist;
			// 	// return;
			// }

			atomicMin(&(dist[pair_id].distance), dd);

			// if(dist[pair_id].distance == dd){
			// 	dist[pair_id].p1 = tri_offset_1;
			// 	dist[pair_id].p2 = tri_offset_2;
			// }

			atomicMin(&(dist[pair_id].min_dist), low_dist);
			atomicMin(&(dist[pair_id].max_dist), high_dist);
		}

		//	printf("%d %d:\t%f %f | %d %d:\t%f %f | %f %f %f | %f %f\n", obj_offset1, tri_offset_1, phdist1, hdist1, obj_offset2, tri_offset_2, phdist2, hdist2, low_dist, high_dist, dd, dist[pair_id].min_dist, dist[pair_id].max_dist);
	}

	/**
	 * Some improvements:
	 * 1. kernel fusion 12s -> 10.8s
	 * 2. use shared memory for atomicMin 10.8s -> 4.5s ?? (really? check results later!!!)
	 */
	__global__ void TriDist_cuda_op_shm(const float *data, const uint32_t *offset_size,
										const float *hausdorff, result_container *dist, uint32_t max_size_1, uint32_t tri_offset_start_2)
	{
		int pair_id = blockIdx.x;

		int tri_offset_2 = threadIdx.x + tri_offset_start_2;

		__shared__ float sh_dd;
		__shared__ float sh_min_dist;
		__shared__ float sh_max_dist;

		if (THID == 0)
		{
			sh_dd = FLT_MAX;
			sh_min_dist = FLT_MAX;
			sh_max_dist = FLT_MAX;
		}
		__syncthreads();

		for (int tri_offset_1 = 0; tri_offset_1 < max_size_1; tri_offset_1++)
		{

			if (tri_offset_1 >= offset_size[pair_id * 4 + 1])
			{
				continue;
			}
			if (tri_offset_2 >= offset_size[pair_id * 4 + 3])
			{
				continue;
			}
			uint32_t obj_offset1 = offset_size[pair_id * 4];
			uint32_t obj_offset2 = offset_size[pair_id * 4 + 2];

			const float *cur_S = data + 9 * (obj_offset1 + tri_offset_1);
			const float *cur_T = data + 9 * (obj_offset2 + tri_offset_2);
			float dd = TriDist_kernel(cur_S, cur_T);

			float phdist1 = *(hausdorff + 2 * (obj_offset1 + tri_offset_1));
			float phdist2 = *(hausdorff + 2 * (obj_offset2 + tri_offset_2));
			float hdist1 = *(hausdorff + 2 * (obj_offset1 + tri_offset_1) + 1);
			float hdist2 = *(hausdorff + 2 * (obj_offset2 + tri_offset_2) + 1);

			float low_dist = max(dd - phdist1 - phdist2, (float)0.0);
			float high_dist = dd + hdist1 + hdist2;

			atomicMin(&sh_dd, dd);
			atomicMin(&sh_min_dist, low_dist);
			atomicMin(&sh_max_dist, high_dist);
		}

		if (THID == 0)
		{
			dist[pair_id].distance = min(sh_dd, dist[pair_id].distance);
			dist[pair_id].min_dist = min(sh_min_dist, dist[pair_id].min_dist);
			dist[pair_id].max_dist = min(sh_max_dist, dist[pair_id].max_dist);
		}
	}

	__device__ __forceinline__ float atomicMinFloat(float* addr, float value) {
		int* addr_as_i = (int*)addr;
		int old = *addr_as_i, assumed;
		do {
			assumed = old;
			old = atomicCAS(addr_as_i, assumed,
				__float_as_int(fminf(value, __int_as_float(assumed))));
		} while (assumed != old);
		return __int_as_float(old);
	}

	/**
	 * Some improvements:
	 * 1. kernel fusion 12s -> 10.8s
	 * 2. use shared memory for atomicMin 10.8s -> 4.5s ?? (really? check results later!!!)
	 * 3. map ``tri_offset_1'' to threads, loop unrolling 4.5s -> 0.6s ??
	 */
	__global__ void TriDist_cuda_op_shm_flat(const float *data, const uint32_t *offset_size,
											 const float *hausdorff, result_container *dist, int start)
	{
		int pair_id = blockIdx.x;

		int max1 = offset_size[pair_id * 4 + 1];
		int max2 = offset_size[pair_id * 4 + 3];

		__shared__ float sh_dd;
		__shared__ float sh_min_dist;
		__shared__ float sh_max_dist;

		if (THID == 0)
		{
			sh_dd = FLT_MAX;
			sh_min_dist = FLT_MAX;
			sh_max_dist = FLT_MAX;
		}
		__syncthreads();

		int tri_offset_1 = (THID + start) / max2;
		int tri_offset_2 = (THID + start) % max2;

		if (tri_offset_1 < max1 && tri_offset_2 < max2)
		{
			uint32_t obj_offset1 = offset_size[pair_id * 4];
			uint32_t obj_offset2 = offset_size[pair_id * 4 + 2];

			const float *cur_S = data + 9 * (obj_offset1 + tri_offset_1);
			const float *cur_T = data + 9 * (obj_offset2 + tri_offset_2);
			float dd = TriDist_kernel(cur_S, cur_T);

			float phdist1 = *(hausdorff + 2 * (obj_offset1 + tri_offset_1));
			float phdist2 = *(hausdorff + 2 * (obj_offset2 + tri_offset_2));
			float hdist1 = *(hausdorff + 2 * (obj_offset1 + tri_offset_1) + 1);
			float hdist2 = *(hausdorff + 2 * (obj_offset2 + tri_offset_2) + 1);

			float low_dist = max(dd - phdist1 - phdist2, (float)0.0);
			float high_dist = dd + hdist1 + hdist2;

			atomicMinFloat(&sh_dd, dd);
			atomicMinFloat(&sh_min_dist, low_dist);
			atomicMinFloat(&sh_max_dist, high_dist);
		}
		__syncthreads();

		if (THID == 0)
		{
			dist[pair_id].distance = min(dist[pair_id].distance, sh_dd);
			dist[pair_id].min_dist = min(dist[pair_id].min_dist, sh_min_dist);
			dist[pair_id].max_dist = min(dist[pair_id].max_dist, sh_max_dist);
		}
	}

	__device__ __forceinline__
	int find_pair_from_prefix(const int* prefix, int pair_num, int t)
	{
		int l = 0, r = pair_num;
		while (l < r) {
			int m = (l + r) >> 1;
			if (prefix[m + 1] <= t)
				l = m + 1;
			else
				r = m;
		}
		return l;
	}

	__global__
	void TriDist_cuda_op_shm_flat_gpt_flatten(
		const float* data,
		const uint32_t* offset_size,
		const float* hausdorff,
		result_container* dist,
		const int* prefix,     // prefix sum
		int pair_num,
		int total_tasks
	) {
		int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
		int stride     = blockDim.x * gridDim.x;

		for (int t = global_tid; t < total_tasks; t += stride) {

			int pair_id = find_pair_from_prefix(prefix, pair_num, t);
			int local_id = t - prefix[pair_id];

			int max1 = offset_size[pair_id * 4 + 1];
			int max2 = offset_size[pair_id * 4 + 3];

			int tri_offset_1 = local_id / max2;
			int tri_offset_2 = local_id % max2;

			if (tri_offset_1 >= max1 || tri_offset_2 >= max2) continue;

			uint32_t obj_offset1 = offset_size[pair_id * 4];
			uint32_t obj_offset2 = offset_size[pair_id * 4 + 2];

			const float* cur_S = data + 9 * (obj_offset1 + tri_offset_1);
			const float* cur_T = data + 9 * (obj_offset2 + tri_offset_2);

			
			float dd = TriDist_kernel(cur_S, cur_T);

			float phdist1 = hausdorff[2 * (obj_offset1 + tri_offset_1)];
			float phdist2 = hausdorff[2 * (obj_offset2 + tri_offset_2)];
			float hdist1  = hausdorff[2 * (obj_offset1 + tri_offset_1) + 1];
			float hdist2  = hausdorff[2 * (obj_offset2 + tri_offset_2) + 1];

			float low_dist  = fmaxf(dd - phdist1 - phdist2, 0.0f);
			float high_dist = dd + hdist1 + hdist2;

			atomicMinFloat(&dist[pair_id].distance, dd);
			atomicMinFloat(&dist[pair_id].min_dist, low_dist);
			atomicMinFloat(&dist[pair_id].max_dist, high_dist);
		}
	}
	
	__device__ __forceinline__
	float warpMin(unsigned mask, float v) {
		for (int offset = 16; offset > 0; offset >>= 1) {
			float other = __shfl_down_sync(mask, v, offset);
			v = fminf(v, other);
		}
		return v;
	}


	__global__ 
	void TriDist_cuda_op_shm_flat_gpt(
		const float *data, 
		const uint32_t *offset_size,
		const float *hausdorff,
		result_container *dist,
		int start,
		int *d_location,
		size_t *d_prefix_obj,
		float *d_out_min,
		float *d_out_max,
		float *d_min_min_dist,
		float *d_min_max_dist,
		size_t *d_valid_voxel_prefix,
		int *d_valid_voxel_pairs,
		int lod
	)
	{
		int pair_id = blockIdx.x;
		int tid = THID;

		__shared__ float sh_dd[512];
		__shared__ float sh_min_dist[512];
		__shared__ float sh_max_dist[512];

		int max1 = offset_size[pair_id * 4 + 1];
		int max2 = offset_size[pair_id * 4 + 3];

		float local_dd       = FLT_MAX;
		float local_min_dist = FLT_MAX;
		float local_max_dist = FLT_MAX;

		int tri_id = start + tid;

		if (tri_id < max1 * max2)
		{
			int tri_offset_1 = tri_id / max2;
			int tri_offset_2 = tri_id % max2;

			if (tri_offset_1 < max1 && tri_offset_2 < max2)
			{
				uint32_t obj_offset1 = offset_size[pair_id * 4];
				uint32_t obj_offset2 = offset_size[pair_id * 4 + 2];

				const float *cur_S = data + 9 * (obj_offset1 + tri_offset_1);
				const float *cur_T = data + 9 * (obj_offset2 + tri_offset_2);

				float dd = TriDist_kernel(cur_S, cur_T);

				float phdist1 = hausdorff[2 * (obj_offset1 + tri_offset_1)];
				float phdist2 = hausdorff[2 * (obj_offset2 + tri_offset_2)];
				float hdist1  = hausdorff[2 * (obj_offset1 + tri_offset_1) + 1];
				float hdist2  = hausdorff[2 * (obj_offset2 + tri_offset_2) + 1];

				float low_dist  = max(dd - phdist1 - phdist2, 0.0f);
				float high_dist = dd + hdist1 + hdist2;

				local_dd       = dd;
				local_min_dist = low_dist;
				local_max_dist = high_dist;
			}
		}

		sh_dd[tid]       = local_dd;
		sh_min_dist[tid] = local_min_dist;
		sh_max_dist[tid] = local_max_dist;
		__syncthreads();


		// // ========================== Reduction 1 ===================
		// if (threadIdx.x == 0) {
		// 	float dd  = sh_dd[0];
		// 	float mn  = sh_min_dist[0];
		// 	float mx  = sh_max_dist[0];

		// 	for (int i = 1; i < blockDim.x; ++i) {
		// 		dd = fminf(dd, sh_dd[i]);
		// 		mn = fminf(mn, sh_min_dist[i]);
		// 		mx = fminf(mx, sh_max_dist[i]);
		// 	}
		// 	sh_dd[0]       = dd;
		// 	sh_min_dist[0] = mn;
		// 	sh_max_dist[0] = mx;
		// }

		// ========================== Reduction 2 ===================
		for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
			if (tid < stride && tid + stride < blockDim.x) {
				sh_dd[tid]       = fminf(sh_dd[tid],       sh_dd[tid + stride]);
				sh_min_dist[tid] = fminf(sh_min_dist[tid], sh_min_dist[tid + stride]);
				sh_max_dist[tid] = fminf(sh_max_dist[tid], sh_max_dist[tid + stride]);
			}
			__syncthreads();
		}

		if (tid == 0)
		{
			dist[pair_id].distance = fminf(dist[pair_id].distance, sh_dd[0]);
			dist[pair_id].min_dist = fminf(dist[pair_id].min_dist, sh_min_dist[0]);
			dist[pair_id].max_dist = fminf(dist[pair_id].max_dist, sh_max_dist[0]);
		}

		// =============== (NO NEED) Update voxel-level distance at the same time ============

		// if (tid == 0) {
		// 	int obj_x   = d_location[3*pair_id];
		// 	int obj_y   = d_location[3*pair_id+1];
		// 	int voxel_p = d_location[3*pair_id+2];
		// 	int voxel_idx = d_prefix_obj[obj_y] + voxel_p;


		// 	if (lod < 100) {
		// 		d_out_min[voxel_idx] = fminf(d_out_min[voxel_idx], sh_min_dist[0]);
		// 		d_out_max[voxel_idx] = fminf(d_out_max[voxel_idx], sh_max_dist[0]);
		// 		atomicMinFloat(&d_min_min_dist[obj_y], d_out_min[voxel_idx]);
		// 		atomicMinFloat(&d_min_max_dist[obj_y], d_out_max[voxel_idx]);
		// 	} else {
		// 		d_out_min[voxel_idx] = sh_dd[0];
		// 		d_out_max[voxel_idx] = sh_dd[0];
		// 		atomicMinFloat(&d_min_min_dist[obj_y], sh_dd[0]);
		// 		atomicMinFloat(&d_min_max_dist[obj_y], sh_dd[0]);
		// 	}
		// }
	}

	__global__ 
	void TriDist_cuda_op_shm_flat_fused(
		const float *data, 
		const size_t *offset_size,
		const float *hausdorff,
		result_container *dist,
		int *d_location,
		size_t *d_prefix_obj,
		float *d_out_min,
		float *d_out_max,
		float *d_min_min_dist,
		float *d_min_max_dist,
		size_t *d_valid_voxel_prefix,
		int *d_valid_voxel_pairs,
		int lod
	)
	{
		size_t pair_id = blockIdx.x;
		int tid     = threadIdx.x;

		__shared__ float sh_dd[512];
		__shared__ float sh_min_dist[512];
		__shared__ float sh_max_dist[512];

		size_t max1 = offset_size[pair_id * 4 + 1];
		size_t max2 = offset_size[pair_id * 4 + 3];

		size_t total_tasks = max1 * max2;

		float local_dd       = FLT_MAX;
		float local_min_dist = FLT_MAX;
		float local_max_dist = FLT_MAX;

		size_t obj_offset1 = offset_size[pair_id * 4];
		size_t obj_offset2 = offset_size[pair_id * 4 + 2];

		// ===== stride over ALL tasks =====
		for (size_t tri_id = tid; tri_id < total_tasks; tri_id += blockDim.x)
		{
			int tri_offset_1 = tri_id / max2;
			int tri_offset_2 = tri_id % max2;

			const float *cur_S = data + 9 * (obj_offset1 + tri_offset_1);
			const float *cur_T = data + 9 * (obj_offset2 + tri_offset_2);

			float dd = TriDist_kernel(cur_S, cur_T);

			float phdist1 = hausdorff[2 * (obj_offset1 + tri_offset_1)];
			float phdist2 = hausdorff[2 * (obj_offset2 + tri_offset_2)];
			float hdist1  = hausdorff[2 * (obj_offset1 + tri_offset_1) + 1];
			float hdist2  = hausdorff[2 * (obj_offset2 + tri_offset_2) + 1];

			float low_dist  = max(dd - phdist1 - phdist2, 0.0f);
			float high_dist = dd + hdist1 + hdist2;

			local_dd       = fminf(local_dd, dd);
			local_min_dist = fminf(local_min_dist, low_dist);
			local_max_dist = fminf(local_max_dist, high_dist);
		}

		sh_dd[tid]       = local_dd;
		sh_min_dist[tid] = local_min_dist;
		sh_max_dist[tid] = local_max_dist;
		__syncthreads();

		// ===== block reduction =====
		for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
		{
			if (tid < stride)
			{
				sh_dd[tid]       = fminf(sh_dd[tid],       sh_dd[tid + stride]);
				sh_min_dist[tid] = fminf(sh_min_dist[tid], sh_min_dist[tid + stride]);
				sh_max_dist[tid] = fminf(sh_max_dist[tid], sh_max_dist[tid + stride]);
			}
			__syncthreads();
		}

		if (tid == 0)
		{
			dist[pair_id].distance = sh_dd[0];
			dist[pair_id].min_dist = sh_min_dist[0];
			dist[pair_id].max_dist = sh_max_dist[0];
		}
	}

	/**
	 * Only used for ablation study, do not used this
	 */
	__global__
	void TriDist_cuda_op_atomic(
		const float *data,
		const size_t *offset_size,
		const float *hausdorff,
		result_container *dist,
		float *g_dd,
		float *g_min_dist,
		float *g_max_dist
	)
	{
		int pair_id = blockIdx.x;
		int tid     = threadIdx.x;

		// Initialize once per block
		if (tid == 0) {
			g_dd[pair_id]       = FLT_MAX;
			g_min_dist[pair_id] = FLT_MAX;
			g_max_dist[pair_id] = FLT_MAX;
		}
		__syncthreads();

		int max1 = offset_size[pair_id * 4 + 1];
		int max2 = offset_size[pair_id * 4 + 3];
		int total_tasks = max1 * max2;

		size_t obj_offset1 = offset_size[pair_id * 4];
		size_t obj_offset2 = offset_size[pair_id * 4 + 2];


		float local_dd       = FLT_MAX;
		float local_min_dist = FLT_MAX;
		float local_max_dist = FLT_MAX;

		for (int tri_id = tid; tri_id < total_tasks; tri_id += blockDim.x)
		{
			int tri_offset_1 = tri_id / max2;
			int tri_offset_2 = tri_id % max2;

			const float *cur_S = data + 9 * (obj_offset1 + tri_offset_1);
			const float *cur_T = data + 9 * (obj_offset2 + tri_offset_2);

			float dd = TriDist_kernel(cur_S, cur_T);

			float phdist1 = hausdorff[2 * (obj_offset1 + tri_offset_1)];
			float phdist2 = hausdorff[2 * (obj_offset2 + tri_offset_2)];
			float hdist1  = hausdorff[2 * (obj_offset1 + tri_offset_1) + 1];
			float hdist2  = hausdorff[2 * (obj_offset2 + tri_offset_2) + 1];

			float low_dist  = fmaxf(dd - phdist1 - phdist2, 0.0f);
			float high_dist = dd + hdist1 + hdist2;

			local_dd       = fminf(local_dd, dd);
			local_min_dist = fminf(local_min_dist, low_dist);
			local_max_dist = fminf(local_max_dist, high_dist);
		}

		atomicMinFloat(&g_dd[pair_id],       local_dd);
		atomicMinFloat(&g_min_dist[pair_id], local_min_dist);
		atomicMinFloat(&g_max_dist[pair_id], local_max_dist);

		__syncthreads();

		if (tid == 0)
		{
			dist[pair_id].distance = g_dd[pair_id];
			dist[pair_id].min_dist = g_min_dist[pair_id];
			dist[pair_id].max_dist = g_max_dist[pair_id];
		}
	}

	__global__ 
	void update_partial_voxel_level_distance(
		const int *d_status,
		const int *d_prefix_obj,
		const float *d_out_min,
		const float *d_min_max_dist,
		const int *d_valid_voxel_prefix,
		const int *d_valid_voxel_pairs,
		int *d_new_count
	)
	{
		__shared__ int sh_count[1024]; 

		int pair_idx = blockIdx.x;
		int tid = threadIdx.x;

		if (d_status[pair_idx] != UNDECIDED) return;
		
		int begin = d_valid_voxel_prefix[pair_idx];
		int end   = d_valid_voxel_prefix[pair_idx+1];

    	int local_count = 0;

		for (int cur_idx = begin + tid; cur_idx < end; cur_idx += blockDim.x)
		{
			int t = d_valid_voxel_pairs[3 * cur_idx + 2];

			int offset = d_prefix_obj[pair_idx];
			float mindist = d_out_min[offset + t];

			if (mindist <= d_min_max_dist[pair_idx])
			{
				local_count++;
			}
		}
		
		sh_count[tid] = local_count;
		__syncthreads();

		// reduction for count
		if (tid == 0) {
			for (int i = 1; i < blockDim.x; ++i) {
				sh_count[0] += sh_count[i];
			}
		}

		if (tid == 0)
		{
			d_new_count[pair_idx] = sh_count[0];
		}
	}

	__global__ void record_valid_voxel_pairs_from_old(
		const int *d_status,
		const int *d_old_valid_prefix,
		const int *d_old_valid_pairs,
		const int *d_prefix_obj,           
		const float *d_out_min,      
		const float *d_min_max_dist, 
		const int *d_new_valid_prefix,  
		int *d_new_valid_pairs
	)
	{
		__shared__ int sh_count[1024];
		__shared__ int sh_offset[1024];

		int pair_idx = blockIdx.x;
		int tid = threadIdx.x;

		if (d_status[pair_idx] != UNDECIDED) return;

		float threshold = d_min_max_dist[pair_idx];

		int old_begin = d_old_valid_prefix[pair_idx];
		int old_end   = d_old_valid_prefix[pair_idx + 1];
		int old_count = old_end - old_begin;

		int voxel_base = d_prefix_obj[pair_idx];
		int out_base   = d_new_valid_prefix[pair_idx] * 3;

		// -------------------------------
		// Phase 1: count per thread
		// -------------------------------
		int local_count = 0;
		for (int k = tid; k < old_count; k += blockDim.x)
		{
			int cur = old_begin + k;
			int t   = d_old_valid_pairs[3 * cur + 2];  // voxel_pair_id

			if (d_out_min[voxel_base + t] <= threshold)
			{
				local_count++;
			}
		}

		sh_count[tid] = local_count;
		__syncthreads();

		// -------------------------------
		// Phase 2: block prefix sum
		// -------------------------------
		int offset = 0;
		for (int i = 0; i < tid; ++i)
		{
			offset += sh_count[i];
		}
		sh_offset[tid] = offset;
		__syncthreads();

		// -------------------------------
		// Phase 3: write new valid voxel pairs
		// -------------------------------
		int write_pos = out_base + sh_offset[tid] * 3;
		int local_write_count = 0;

		for (int k = tid; k < old_count; k += blockDim.x)
		{
			int cur = old_begin + k;
			int t   = d_old_valid_pairs[3 * cur + 2];

			if (d_out_min[voxel_base + t] <= threshold)
			{
				int pos = write_pos + local_write_count * 3;
				d_new_valid_pairs[pos]     = d_old_valid_pairs[3 * cur];
				d_new_valid_pairs[pos + 1] = d_old_valid_pairs[3 * cur + 1];
				d_new_valid_pairs[pos + 2] = t;
				local_write_count++;
			}
		}
	}


	__device__ __forceinline__ int find_pair_id(uint32_t tid, const uint32_t *prefix_num, int pair_num)
	{
		int low = 0;
		int high = pair_num; // Assuming prefix_num has pair_num + 1 elements
		int ans = 0;

		while (low <= high)
		{
			int mid = low + (high - low) / 2;
			if (prefix_num[mid] <= tid)
			{
				ans = mid;
				low = mid + 1;
			}
			else
			{
				high = mid - 1;
			}
		}
		return ans;
	}

	__global__ void TriDist_cuda_op_balanced_wkld(const float *data, const uint32_t *offset_size,
												  const float *hausdorff, result_container *dist, uint32_t *prefix_num, int pair_num)
	{
		uint32_t global_tid = blockIdx.x * blockDim.x + threadIdx.x;

		// 总 task 数
		uint32_t total_tasks = prefix_num[pair_num];
		if (global_tid >= total_tasks)
			return;

		int pair_id = find_pair_id(global_tid, prefix_num, pair_num);

		uint32_t local_task = global_tid - prefix_num[pair_id];

		uint32_t max1 = offset_size[pair_id * 4 + 1];
		uint32_t max2 = offset_size[pair_id * 4 + 3];

		// 从 local_task 反解二维 index
		uint32_t tri_offset_1 = local_task / max2;
		uint32_t tri_offset_2 = local_task % max2;

		if (tri_offset_1 >= max1 || tri_offset_2 >= max2)
			return;

		uint32_t obj_offset1 = offset_size[pair_id * 4];
		uint32_t obj_offset2 = offset_size[pair_id * 4 + 2];

		const float *cur_S = data + 9 * (obj_offset1 + tri_offset_1);
		const float *cur_T = data + 9 * (obj_offset2 + tri_offset_2);
		float dd = TriDist_kernel(cur_S, cur_T);

		float phdist1 = *(hausdorff + 2 * (obj_offset1 + tri_offset_1));
		float phdist2 = *(hausdorff + 2 * (obj_offset2 + tri_offset_2));
		float hdist1 = *(hausdorff + 2 * (obj_offset1 + tri_offset_1) + 1);
		float hdist2 = *(hausdorff + 2 * (obj_offset2 + tri_offset_2) + 1);

		float low_dist = max(dd - phdist1 - phdist2, (float)0.0);
		float high_dist = dd + hdist1 + hdist2;

		// if(tri_offset_1==0&&tri_offset_2==0){
		// 	dist[pair_id].distance = dd;
		// 	dist[pair_id].p1 = tri_offset_1;
		// 	dist[pair_id].p2 = tri_offset_2;
		// 	dist[pair_id].min_dist = low_dist;
		// 	dist[pair_id].max_dist = high_dist;
		// 	// return;
		// }

		atomicMin(&(dist[pair_id].distance), dd);

		// if(dist[pair_id].distance == dd){
		// 	dist[pair_id].p1 = tri_offset_1;
		// 	dist[pair_id].p2 = tri_offset_2;
		// }

		atomicMin(&(dist[pair_id].min_dist), low_dist);
		atomicMin(&(dist[pair_id].max_dist), high_dist);
	}

	__global__ void TriInt_cuda(const float *data, const uint32_t *offset_size,
								const float *hausdorff, result_container *results,
								uint32_t tri_offset_1, uint32_t tri_offset_2_start)
	{
		// which batch
		int pair_id = blockIdx.x;
		int tri_offset_2 = threadIdx.x + tri_offset_2_start;

		if (tri_offset_1 >= offset_size[pair_id * 4 + 1])
		{
			return;
		}
		if (tri_offset_2 >= offset_size[pair_id * 4 + 3])
		{
			return;
		}

		// determined
		if (results[pair_id].intersected)
		{
			return;
		}

		// if (results[pair_id].min_dist < 100000000 && results[pair_id].min_dist > 0) {
		//	if(pair_id == 1)
		//	printf("teng %d %d %f\n",tri_offset_1, tri_offset_2, results[pair_id].min_dist);
		//	return;
		// }

		uint32_t obj_offset1 = offset_size[pair_id * 4];
		uint32_t obj_offset2 = offset_size[pair_id * 4 + 2];

		const float *cur_S = data + 9 * (obj_offset1 + tri_offset_1);
		const float *cur_T = data + 9 * (obj_offset2 + tri_offset_2);
		float dd = TriDist_kernel(cur_S, cur_T);

		float phdist1 = *(hausdorff + 2 * (obj_offset1 + tri_offset_1));
		float phdist2 = *(hausdorff + 2 * (obj_offset2 + tri_offset_2));
		float hdist1 = *(hausdorff + 2 * (obj_offset1 + tri_offset_1) + 1);
		float hdist2 = *(hausdorff + 2 * (obj_offset2 + tri_offset_2) + 1);

		float low_dist = max(dd - phdist1 - phdist2, (float)0.0);
		float high_dist = dd + hdist1 + hdist2;

		if (tri_offset_1 == 0 && tri_offset_2 == 0)
		{
			results[pair_id].distance = dd;
			results[pair_id].p1 = tri_offset_1;
			results[pair_id].p2 = tri_offset_2;
			results[pair_id].min_dist = low_dist;
			results[pair_id].max_dist = high_dist;
			return;
		}

		atomicMin(&(results[pair_id].distance), dd);

		if (results[pair_id].distance == dd)
		{
			results[pair_id].p1 = tri_offset_1;
			results[pair_id].p2 = tri_offset_2;
		}

		atomicMin(&(results[pair_id].min_dist), low_dist);
		atomicMin(&(results[pair_id].max_dist), high_dist);
		if (results[pair_id].max_dist == 0.0)
		{
			results[pair_id].intersected = true;
		}

		//	printf("%d %d:\t%f %f | %d %d:\t%f %f | %f %f %f | %f %f\n", obj_offset1, tri_offset_1, phdist1, hdist1, obj_offset2, tri_offset_2, phdist2, hdist2, low_dist, high_dist, dd, dist[pair_id].min_dist, dist[pair_id].max_dist);
	}

	//__global__
	// void TriInt_cuda(const float *data, const float *hausdorff, const uint32_t *offset_size, result_container *intersect, uint32_t cur_offset_1, uint32_t cur_offset_2_start){
	//
	//	int batch_id = blockIdx.x;
	//	uint32_t cur_offset_2 = threadIdx.x+cur_offset_2_start;
	//
	//	if(cur_offset_1>=offset_size[batch_id*4+1]){
	//		return;
	//	}
	//	if(cur_offset_2>=offset_size[batch_id*4+3]){
	//		return;
	//	}
	//
	//	// determined
	//	if(intersect[batch_id].intersected){
	//		return;
	//	}
	//
	//	uint32_t offset1 = offset_size[batch_id*4];
	//	uint32_t offset2 = offset_size[batch_id*4+2];
	//	const float *cur_S = data+9*(offset1+cur_offset_1);
	//	const float *cur_T = data+9*(offset2+cur_offset_2);
	//
	//	float dd = TriDist_kernel(cur_S, cur_T);
	//	if(dd==0.0){
	//		intersect[batch_id].intersected = 1;
	//		intersect[batch_id].p1 = cur_offset_1;
	//		intersect[batch_id].p2 = cur_offset_2;
	//		return;
	//	}
	//
	//	// otherwise, check if the two polyhedrons can intersect
	//	if(hausdorff){
	//		const float phdist1 = *(hausdorff+2*(offset1+cur_offset_1));
	//		const float phdist2 = *(hausdorff+2*(offset2+cur_offset_2));
	//		dd = dd - phdist1 - phdist2;
	//	}
	//	atomicMin(&intersect[batch_id].distance, dd);
	//}

	/**
	 *
	 * Kernel for compute max and min distance between voxel pairs
	 *
	 * range aab::distance(const aab &b){
		range ret;
		ret.maxdist = 0;
		ret.mindist = 0;
		float tmp1 = 0;
		float tmp2 = 0;
		for(int i=0;i<3;i++){
			tmp1 = low[i]-b.high[i];
			tmp2 = high[i]-b.low[i];
			ret.maxdist += (tmp1+tmp2)*(tmp1+tmp2)/4;
			if(tmp2<0){
				ret.mindist += tmp2*tmp2;
			}else if(tmp1>0){
				ret.mindist += tmp1*tmp1;
			}
		}
		ret.mindist = sqrt(ret.mindist);
		ret.maxdist = sqrt(ret.maxdist);
		return ret;
	}
	 */

	__device__ __forceinline__ void voxel_dist(const float *v1,
											   const float *v2,
											   float &mindist,
											   float &maxdist)
	{
		mindist = 0.0f;
		maxdist = 0.0f;

#pragma unroll
		for (int i = 0; i < 3; i++)
		{
			float tmp1 = v1[i] - v2[i + 3];
			float tmp2 = v1[i + 3] - v2[i];

			float mid = 0.5f * (tmp1 + tmp2);
			maxdist += mid * mid;

			if (tmp2 < 0.0f)
				mindist += tmp2 * tmp2;
			else if (tmp1 > 0.0f)
				mindist += tmp1 * tmp1;
		}

		mindist = sqrtf(mindist);
		maxdist = sqrtf(maxdist);
	}

	__device__ __forceinline__ void voxel_dist_core(const float *v1,
                                           const float *v2,
                                           float &mindist,
                                           float &maxdist)
	{
		mindist = 0.0f;
		maxdist = 0.0f;

		#pragma unroll
		for (int i = 0; i < 3; i++)
		{
			float tmp1 = v1[i] - v2[i + 3];
			float tmp2 = v1[i + 3] - v2[i];

			if (tmp2 < 0.0f)
				mindist += tmp2 * tmp2;
			else if (tmp1 > 0.0f)
				mindist += tmp1 * tmp1;

			float dc = v1[i + 6] - v2[i + 6];
			maxdist += dc * dc;
		}

		mindist = sqrtf(mindist);
		maxdist = sqrtf(maxdist);
	}

	__global__ void voxel_pair_dist(float *voxel1, int n1, float *voxel2, int n2, float *out_min, float *out_max)
	{

		int total = n1 * n2;
		if (GLTHID >= total)
			return;

		int i = GLTHID / n2;
		int j = GLTHID % n2;

		const float *v1 = voxel1 + i * 6;
		const float *v2 = voxel2 + j * 6;

		float mindist, maxdist;
		voxel_dist(v1, v2, mindist, maxdist);

		out_min[GLTHID] = mindist;
		out_max[GLTHID] = maxdist;
	}

	__global__ void voxel_pair_dist_all(
		int v1,
		const int *candidates,
		const int *prefix,
		DeviceVoxels dv1,
		DeviceVoxels dv2,
		float *out_min,
		float *out_max,
		int base)
	{
		int pair_id = blockIdx.x;
		int tid = THID + base;

		int w2 = candidates[pair_id];
		int cnt1 = dv1.d_voxel_count[v1];
		int cnt2 = dv2.d_voxel_count[w2];

		if (tid >= cnt1 * cnt2)
			return;

		int i = tid / cnt2; // voxel index in v1
		int j = tid % cnt2; // voxel index in candidate v2

		const float *v1_voxel = dv1.d_all_voxels + (dv1.d_voxel_offset[v1] + i) * 6;
		const float *v2_voxel = dv2.d_all_voxels + (dv2.d_voxel_offset[w2] + j) * 6;

		float mindist, maxdist;
		voxel_dist(v1_voxel, v2_voxel, mindist, maxdist);

		int out_idx = prefix[pair_id] + tid;
		out_min[out_idx] = mindist;
		out_max[out_idx] = maxdist;
	}

	void allocateVoxels(vector<Voxel *> &voxels, float *&d_wrapper_voxels)
	{
		vector<float> wrapper_voxels(voxels.size() * 9);
		CUDA_SAFE_CALL(cudaMalloc(&d_wrapper_voxels, voxels.size() * 9 * sizeof(float)));

		for (int i = 0; i < voxels.size(); ++i)
		{
			wrapper_voxels[i * 9 + 0] = voxels[i]->low[0];
			wrapper_voxels[i * 9 + 1] = voxels[i]->low[1];
			wrapper_voxels[i * 9 + 2] = voxels[i]->low[2];
			wrapper_voxels[i * 9 + 3] = voxels[i]->high[0];
			wrapper_voxels[i * 9 + 4] = voxels[i]->high[1];
			wrapper_voxels[i * 9 + 5] = voxels[i]->high[2];

			wrapper_voxels[i * 9 + 6] = voxels[i]->core[0];
			wrapper_voxels[i * 9 + 7] = voxels[i]->core[1];
			wrapper_voxels[i * 9 + 8] = voxels[i]->core[2];
		}
		CUDA_SAFE_CALL(cudaMemcpy(d_wrapper_voxels, wrapper_voxels.data(), voxels.size() * 9 * sizeof(float), cudaMemcpyHostToDevice));
	}

	DeviceVoxels allocateVoxelsForAll(
		size_t tile_size,
		vector<int> &voxel_offset,
		vector<int> &voxel_count,
		vector<float> &all_voxels)
	{
		DeviceVoxels dv;
		dv.tile_size = tile_size;

		// -------------------------------
		// 1. Allocate device memory
		// -------------------------------
		CUDA_SAFE_CALL(cudaMalloc(&dv.d_all_voxels, all_voxels.size() * sizeof(float)));
		CUDA_SAFE_CALL(cudaMalloc(&dv.d_voxel_count, voxel_count.size() * sizeof(int)));
		CUDA_SAFE_CALL(cudaMalloc(&dv.d_voxel_offset, voxel_offset.size() * sizeof(int)));

		std::cout << "GPU data size = " << all_voxels.size() * sizeof(float) + 2 * voxel_count.size() * sizeof(int) << std::endl;

		// -------------------------------
		// 2. Copy data to device
		// -------------------------------
		CUDA_SAFE_CALL(cudaMemcpy(dv.d_all_voxels, all_voxels.data(),
								  all_voxels.size() * sizeof(float),
								  cudaMemcpyHostToDevice));

		CUDA_SAFE_CALL(cudaMemcpy(dv.d_voxel_count, voxel_count.data(),
								  voxel_count.size() * sizeof(int),
								  cudaMemcpyHostToDevice));

		CUDA_SAFE_CALL(cudaMemcpy(dv.d_voxel_offset, voxel_offset.data(),
								  voxel_offset.size() * sizeof(int),
								  cudaMemcpyHostToDevice));

		// -------------------------------
		// 3. Store host copies
		// -------------------------------
		dv.h_voxel_count = voxel_count;
		dv.h_voxel_offset = voxel_offset;

		return dv;
	}

	/** Deprecated
	vector<float *> allocateVoxelsForAll(size_t tile_size, Tile *tile)
	{
		vector<float *> d_voxels(tile_size);

		for(int i=0;i<tile_size;i++){
			HiMesh_Wrapper *wrapper = tile->get_mesh_wrapper(i);
			vector<float> wrapper_voxels(wrapper->voxels.size() * 6);

			for(int j=0; j<wrapper->voxels.size(); ++j) {
				wrapper_voxels[j*6]   = wrapper->voxels[j]->low[0];
				wrapper_voxels[j*6+1] = wrapper->voxels[j]->low[1];
				wrapper_voxels[j*6+2] = wrapper->voxels[j]->low[2];
				wrapper_voxels[j*6+3] = wrapper->voxels[j]->high[0];
				wrapper_voxels[j*6+4] = wrapper->voxels[j]->high[1];
				wrapper_voxels[j*6+5] = wrapper->voxels[j]->high[2];
			}
			CUDA_SAFE_CALL(cudaMalloc(&d_voxels[i], wrapper_voxels.size() * sizeof(float)));
			CUDA_SAFE_CALL(cudaMemcpy(d_voxels[i], wrapper_voxels.data(), wrapper_voxels.size() * sizeof(float), cudaMemcpyHostToDevice));
		}
		return d_voxels;
	}
	*/

	void compute_voxel_pair_distance(float *v1, int len_v1,
									 float *v2, int len_v2,
									 vector<float> &out_min, vector<float> &out_max)
	{
		int length = len_v1 * len_v2;
		int threadsPerBlk = 128;
		int numBlk = (length + threadsPerBlk - 1) / threadsPerBlk;

		float *d_out_min, *d_out_max;
		CUDA_SAFE_CALL(cudaMalloc(&d_out_min, length * sizeof(float)));
		CUDA_SAFE_CALL(cudaMalloc(&d_out_max, length * sizeof(float)));

		voxel_pair_dist<<<numBlk, threadsPerBlk>>>(v1, len_v1,
												   v2, len_v2,
												   d_out_min, d_out_max);
		check_execution();
		cudaDeviceSynchronize();

		CUDA_SAFE_CALL(cudaMemcpy(out_min.data(), d_out_min, length * sizeof(float), cudaMemcpyDeviceToHost));
		CUDA_SAFE_CALL(cudaMemcpy(out_max.data(), d_out_max, length * sizeof(float), cudaMemcpyDeviceToHost));

		CUDA_SAFE_CALL(cudaFree(d_out_min));
		CUDA_SAFE_CALL(cudaFree(d_out_max));
	}

	void compute_voxel_pair_distance(
		int v1,
		vector<pair<int, range>> &candidates,
		const DeviceVoxels &dv1,
		const DeviceVoxels &dv2,
		vector<float> &h_min,
		vector<float> &h_max)
	{
		int pair_num = candidates.size();

		/* -------------------------------
		 * 1. 构造 prefix + candidate index
		 * ------------------------------- */
		vector<int> h_candidates(pair_num);
		vector<int> h_prefix(pair_num + 1, 0);

		int max_pair_voxels = 0;

		for (int i = 0; i < pair_num; ++i)
		{
			int w2 = candidates[i].first;
			h_candidates[i] = w2;

			int cnt2 = dv2.h_voxel_count[w2];
			int cnt1 = dv1.h_voxel_count[v1];

			h_prefix[i + 1] = h_prefix[i] + cnt1 * cnt2;

			max_pair_voxels = max(max_pair_voxels, cnt1 * cnt2);
		}

		int total_pairs = h_prefix[pair_num];

		/* -------------------------------
		 * 2. Device allocations
		 * ------------------------------- */
		int *d_candidates = nullptr;
		int *d_prefix = nullptr;
		CUDA_SAFE_CALL(cudaMalloc(&d_candidates, pair_num * sizeof(int)));
		CUDA_SAFE_CALL(cudaMalloc(&d_prefix, (pair_num + 1) * sizeof(int)));

		CUDA_SAFE_CALL(cudaMemcpy(d_candidates, h_candidates.data(), pair_num * sizeof(int), cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(d_prefix, h_prefix.data(), (pair_num + 1) * sizeof(int), cudaMemcpyHostToDevice));

		float *d_out_min = nullptr, *d_out_max = nullptr;
		CUDA_SAFE_CALL(cudaMalloc(&d_out_min, total_pairs * sizeof(float)));
		CUDA_SAFE_CALL(cudaMalloc(&d_out_max, total_pairs * sizeof(float)));

		/* -------------------------------
		 * 3. Kernel launch (tile v2 voxels)
		 * ------------------------------- */
		constexpr int MAX_DIM = 1024;

		for (int base = 0; base < max_pair_voxels; base += MAX_DIM)
		{
			int dim = min(MAX_DIM, max_pair_voxels - base);
			voxel_pair_dist_all<<<pair_num, dim>>>(
				v1,
				d_candidates,
				d_prefix,
				dv1,
				dv2,
				d_out_min,
				d_out_max,
				base);
		}
		CUDA_SAFE_CALL(cudaDeviceSynchronize());

		h_min.resize(total_pairs);
		h_max.resize(total_pairs);
		cudaMemcpy(h_min.data(), d_out_min, total_pairs * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_max.data(), d_out_max, total_pairs * sizeof(float), cudaMemcpyDeviceToHost);

		cudaFree(d_candidates);
		cudaFree(d_prefix);
		cudaFree(d_out_min);
		cudaFree(d_out_max);
	}

	__global__ void voxel_pair_dist_kernel_for_all(
		DeviceVoxels dv1,
		DeviceVoxels dv2,
		const int *d_obj1,
		const int *d_obj2,
		const size_t *d_prefix_obj,
		float *d_out_min,
		float *d_out_max,
		float *d_min_max_dist,
		float *d_min_min_dist,
		int *d_count
	)
	{
		__shared__ float sh_min_max[1024];
		__shared__ float sh_min_min[1024];
		__shared__ int sh_count[1024];

		int pair_idx = blockIdx.x;
		int tid = threadIdx.x;

		int obj1_id = d_obj1[pair_idx];
		int obj2_id = d_obj2[pair_idx];

		int n1 = dv1.d_voxel_count[obj1_id];
		int n2 = dv2.d_voxel_count[obj2_id];
		int total_pairs = n1 * n2;

		float local_min_max = FLT_MAX;
		float local_min_min = FLT_MAX;

		// stride over voxel pairs
		for (int t = tid; t < total_pairs; t += blockDim.x)
		{

			int i = t / n2;
			int j = t % n2;

			const float *v1 = dv1.d_all_voxels + (dv1.d_voxel_offset[obj1_id] + i) * 6;
			const float *v2 = dv2.d_all_voxels + (dv2.d_voxel_offset[obj2_id] + j) * 6;

			float mindist, maxdist;
			voxel_dist(v1, v2, mindist, maxdist);

			size_t offset = d_prefix_obj[pair_idx];
			size_t idx = offset + t;

			d_out_min[idx] = mindist;
			d_out_max[idx] = maxdist;

			local_min_max = fminf(local_min_max, maxdist);
			local_min_min = fminf(local_min_min, mindist);
		}

		sh_min_max[tid] = local_min_max;
		sh_min_min[tid] = local_min_min;
		__syncthreads();

		// reduction for min
		for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
			if (tid < stride && tid + stride < blockDim.x) {
				sh_min_max[tid] = fminf(sh_min_max[tid], sh_min_max[tid + stride]);
				sh_min_min[tid] = fminf(sh_min_min[tid], sh_min_min[tid + stride]);
			}
			__syncthreads();
		}

		if (tid == 0)
		{
			d_min_max_dist[pair_idx] = sh_min_max[0];
			d_min_min_dist[pair_idx] = sh_min_min[0];
		}

		__syncthreads();

		// ---------- PASS 2: count mindist < sh_min_max[0] ----------
		int local_count = 0;

		for (int t = tid; t < total_pairs; t += blockDim.x)
		{

			size_t offset = d_prefix_obj[pair_idx];
			float mindist = d_out_min[offset + t];

			if (mindist < sh_min_max[0])
			{
				local_count++;
			}
		}

		sh_count[tid] = local_count;
		__syncthreads();

		// Reduction for sum
		for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
			if (tid < stride && tid + stride < blockDim.x) {
				sh_count[tid] += sh_count[tid + stride];
			}
			__syncthreads();
		}

		if (tid == 0)
		{
			d_count[pair_idx] = sh_count[0];
		}
	}

	__global__ void voxel_pair_dist_kernel_for_all_aggregate_distance(
		DeviceVoxels dv1,
		DeviceVoxels dv2,
		const int *d_obj1,
		const int *d_obj2,
		float *d_min_max_dist,
		float *d_min_min_dist
	)
	{
		__shared__ float sh_min_max[1024];
		__shared__ float sh_min_min[1024];

		int pair_idx = blockIdx.x;
		int tid = threadIdx.x;

		int obj1_id = d_obj1[pair_idx];
		int obj2_id = d_obj2[pair_idx];

		int n1 = dv1.d_voxel_count[obj1_id];
		int n2 = dv2.d_voxel_count[obj2_id];
		int total_pairs = n1 * n2;

		float local_min_max = FLT_MAX;
		float local_min_min = FLT_MAX;

		// stride over voxel pairs
		for (int t = tid; t < total_pairs; t += blockDim.x)
		{
			int i = t / n2;
			int j = t % n2;

			const float *v1 = dv1.d_all_voxels + (dv1.d_voxel_offset[obj1_id] + i) * 9;
			const float *v2 = dv2.d_all_voxels + (dv2.d_voxel_offset[obj2_id] + j) * 9;

			float mindist, maxdist;
			voxel_dist_core(v1, v2, mindist, maxdist);

			local_min_max = fminf(local_min_max, maxdist);
			local_min_min = fminf(local_min_min, mindist);
		}

		sh_min_max[tid] = local_min_max;
		sh_min_min[tid] = local_min_min;
		__syncthreads();

		// reduction for min
		for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
			if (tid < stride && tid + stride < blockDim.x) {
				sh_min_max[tid] = fminf(sh_min_max[tid], sh_min_max[tid + stride]);
				sh_min_min[tid] = fminf(sh_min_min[tid], sh_min_min[tid + stride]);
			}
			__syncthreads();
		}

		if (tid == 0)
		{
			d_min_max_dist[pair_idx] = sh_min_max[0];
			d_min_min_dist[pair_idx] = sh_min_min[0];
		}
	}

	__global__ void voxel_pair_dist_kernel_for_all_streaming(
		DeviceVoxels dv1,
		DeviceVoxels dv2,
		const int *d_obj1,
		const int *d_obj2,
		const size_t *d_prefix_obj,
		float *d_out_min,
		float *d_out_max,
		float *d_min_max_dist,
		float *d_min_min_dist,
		size_t *d_count,
		int global_pair_offset
	)
	{
		__shared__ float sh_min_max[1024];
		__shared__ float sh_min_min[1024];
		__shared__ int sh_count[1024];

		int pair_idx = blockIdx.x + global_pair_offset;
		int tid = threadIdx.x;

		int obj1_id = d_obj1[pair_idx];
		int obj2_id = d_obj2[pair_idx];

		int n1 = dv1.d_voxel_count[obj1_id];
		int n2 = dv2.d_voxel_count[obj2_id];
		int total_pairs = n1 * n2;

		float local_min_max = FLT_MAX;
		float local_min_min = FLT_MAX;

		size_t chunk_base = d_prefix_obj[global_pair_offset];

		size_t global_offset = d_prefix_obj[pair_idx];
		
		size_t local_offset = global_offset - chunk_base;

		// stride over voxel pairs
		for (int t = tid; t < total_pairs; t += blockDim.x)
		{

			int i = t / n2;
			int j = t % n2;

			const float *v1 = dv1.d_all_voxels + (dv1.d_voxel_offset[obj1_id] + i) * 9;
			const float *v2 = dv2.d_all_voxels + (dv2.d_voxel_offset[obj2_id] + j) * 9;

			float mindist, maxdist;
			voxel_dist_core(v1, v2, mindist, maxdist);

			size_t idx = local_offset + t;

			d_out_min[idx] = mindist;
			d_out_max[idx] = maxdist;

			local_min_max = fminf(local_min_max, maxdist);
			local_min_min = fminf(local_min_min, mindist);
		}

		sh_min_max[tid] = local_min_max;
		sh_min_min[tid] = local_min_min;
		__syncthreads();

		// reduction for min
		for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
			if (tid < stride && tid + stride < blockDim.x) {
				sh_min_max[tid] = fminf(sh_min_max[tid], sh_min_max[tid + stride]);
				sh_min_min[tid] = fminf(sh_min_min[tid], sh_min_min[tid + stride]);
			}
			__syncthreads();
		}

		if (tid == 0)
		{
			d_min_max_dist[pair_idx] = sh_min_max[0];
			d_min_min_dist[pair_idx] = sh_min_min[0];
		}

		__syncthreads();

		// ---------- PASS 2: count mindist < sh_min_max[0] ----------
		int local_count = 0;

		for (int t = tid; t < total_pairs; t += blockDim.x)
		{
			float mindist = d_out_min[local_offset + t];

			if (mindist <= sh_min_max[0])
			{
				local_count++;
			}
		}

		sh_count[tid] = local_count;
		__syncthreads();

		// Reduction for sum
		for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
			if (tid < stride && tid + stride < blockDim.x) {
				sh_count[tid] += sh_count[tid + stride];
			}
			__syncthreads();
		}

		if (tid == 0)
		{
			d_count[blockIdx.x] = sh_count[0];
		}
	}

	__global__ void voxel_pair_dist_kernel_for_all_streaming_knn(
		DeviceVoxels dv1,
		DeviceVoxels dv2,
		const int *d_obj1,
		const int *d_obj2,
		const size_t *d_prefix_obj,
		float *d_out_min,
		float *d_out_max,
		float *d_min_max_dist,
		float *d_min_min_dist,
		size_t *d_count,
		int global_pair_offset,
		int *d_confirmed_after,
		int *d_status_after,
		int K
	)
	{
		__shared__ int sh_count[1024];

		int pair_idx = blockIdx.x + global_pair_offset;
		int tid = threadIdx.x;

		int obj1_id = d_obj1[pair_idx];
		int obj2_id = d_obj2[pair_idx];

		if (d_confirmed_after[obj1_id] == K) return;
		if (d_status_after[pair_idx] != 0) return;

		int n1 = dv1.d_voxel_count[obj1_id];
		int n2 = dv2.d_voxel_count[obj2_id];
		int total_pairs = n1 * n2;

		size_t chunk_base = d_prefix_obj[global_pair_offset];

		size_t global_offset = d_prefix_obj[pair_idx];
		
		size_t local_offset = global_offset - chunk_base;

		// stride over voxel pairs
		for (int t = tid; t < total_pairs; t += blockDim.x)
		{
			int i = t / n2;
			int j = t % n2;

			const float *v1 = dv1.d_all_voxels + (dv1.d_voxel_offset[obj1_id] + i) * 9;
			const float *v2 = dv2.d_all_voxels + (dv2.d_voxel_offset[obj2_id] + j) * 9;

			float mindist, maxdist;
			voxel_dist_core(v1, v2, mindist, maxdist);

			size_t idx = local_offset + t;

			d_out_min[idx] = mindist;
			d_out_max[idx] = maxdist;
		}

		__syncthreads();

		// ---------- PASS 2: count mindist < sh_min_max[0] ----------
		int local_count = 0;

		for (int t = tid; t < total_pairs; t += blockDim.x)
		{
			float mindist = d_out_min[local_offset + t];

			if (mindist <= d_min_max_dist[pair_idx])
			{
				local_count++;
			}
		}


		sh_count[tid] = local_count;
		__syncthreads();

		// Reduction for sum
		for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
			if (tid < stride && tid + stride < blockDim.x) {
				sh_count[tid] += sh_count[tid + stride];
			}
			__syncthreads();
		}

		if (tid == 0)
		{
			d_count[blockIdx.x] = sh_count[0];
		}
	}

	__global__ void voxel_pair_dist_kernel_for_all_streaming_within(
		DeviceVoxels dv1,
		DeviceVoxels dv2,
		const int *d_obj1,
		const int *d_obj2,
		const size_t *d_prefix_obj,
		float *d_out_min,
		float *d_out_max,
		float *d_min_max_dist,
		float *d_min_min_dist,
		size_t *d_count,
		int global_pair_offset,
		float within_distance
	)
	{
		__shared__ float sh_min_max[1024];
		__shared__ float sh_min_min[1024];
		__shared__ int sh_count[1024];

		int pair_idx = blockIdx.x + global_pair_offset;
		int tid = threadIdx.x;

		int obj1_id = d_obj1[pair_idx];
		int obj2_id = d_obj2[pair_idx];

		int n1 = dv1.d_voxel_count[obj1_id];
		int n2 = dv2.d_voxel_count[obj2_id];
		int total_pairs = n1 * n2;

		float local_min_max = FLT_MAX;
		float local_min_min = FLT_MAX;

		size_t chunk_base = d_prefix_obj[global_pair_offset];

		size_t global_offset = d_prefix_obj[pair_idx];
		
		size_t local_offset = global_offset - chunk_base;

		// stride over voxel pairs
		for (int t = tid; t < total_pairs; t += blockDim.x)
		{

			int i = t / n2;
			int j = t % n2;

			const float *v1 = dv1.d_all_voxels + (dv1.d_voxel_offset[obj1_id] + i) * 9;
			const float *v2 = dv2.d_all_voxels + (dv2.d_voxel_offset[obj2_id] + j) * 9;

			float mindist, maxdist;
			voxel_dist_core(v1, v2, mindist, maxdist);

			size_t idx = local_offset + t;

			d_out_min[idx] = mindist;
			d_out_max[idx] = maxdist;

			local_min_max = fminf(local_min_max, maxdist);
			local_min_min = fminf(local_min_min, mindist);
		}

		sh_min_max[tid] = local_min_max;
		sh_min_min[tid] = local_min_min;
		__syncthreads();

		// reduction for min
		for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
			if (tid < stride && tid + stride < blockDim.x) {
				sh_min_max[tid] = fminf(sh_min_max[tid], sh_min_max[tid + stride]);
				sh_min_min[tid] = fminf(sh_min_min[tid], sh_min_min[tid + stride]);
			}
			__syncthreads();
		}

		if (tid == 0)
		{
			d_min_max_dist[pair_idx] = sh_min_max[0];
			d_min_min_dist[pair_idx] = sh_min_min[0];
		}

		__syncthreads();

		// ---------- PASS 2: count mindist < sh_min_max[0] ----------
		int local_count = 0;


		if (d_min_max_dist[pair_idx] > within_distance && 
			d_min_min_dist[pair_idx] <= within_distance) 
		{
			for (int t = tid; t < total_pairs; t += blockDim.x)
			{
				float mindist = d_out_min[local_offset + t];

				if (mindist <= sh_min_max[0] &&
					mindist <= within_distance)
				{
					local_count++;
				}
			}
		}

		sh_count[tid] = local_count;
		__syncthreads();

		// Reduction for sum
		for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
			if (tid < stride && tid + stride < blockDim.x) {
				sh_count[tid] += sh_count[tid + stride];
			}
			__syncthreads();
		}

		if (tid == 0)
		{
			d_count[blockIdx.x] = sh_count[0];
		}
	}

	__global__ void record_valid_voxel_pairs(
		DeviceVoxels dv1,
		DeviceVoxels dv2,
		const int *d_obj1,
		const int *d_obj2,
		const size_t *d_prefix_obj,
		const float *d_out_min,			// voxel-level distance
		const float *d_out_max,			// voxel-level distance
		const float *d_min_max_dist,	// object-level distance
		const size_t *d_valid_prefix,
		int *d_valid_pairs,
		float *d_valid_pairs_dist
	)
	{
		__shared__ int sh_count[1024];
		__shared__ int sh_offset[1024];

		int pair_idx = blockIdx.x;
		int tid = threadIdx.x;

		int obj1 = d_obj1[pair_idx];
		int obj2 = d_obj2[pair_idx];

		int n1 = dv1.d_voxel_count[obj1];
		int n2 = dv2.d_voxel_count[obj2];
		int total_pairs = n1 * n2;

		float threshold = d_min_max_dist[pair_idx];

		size_t voxel_base = d_prefix_obj[pair_idx];

		size_t out_base = d_valid_prefix[pair_idx] * 3;

		size_t out_base_2 = d_valid_prefix[pair_idx] * 2;


		// -------------------------------
		// Phase 1: count per thread
		// -------------------------------
		int local_count = 0;
		for (int t = tid; t < total_pairs; t += blockDim.x)
		{
			if (d_out_min[voxel_base + t] <= threshold)
			{
				local_count++;
			}
		}

		sh_count[tid] = local_count;
		__syncthreads();

		// -------------------------------
		// Phase 2: block prefix sum
		// -------------------------------
		int offset = 0;
		for (int i = 0; i < tid; ++i)
		{
			offset += sh_count[i];
		}
		sh_offset[tid] = offset;
		__syncthreads();

		// -------------------------------
		// Phase 3: write voxel IDs
		// -------------------------------
		int write_pos = sh_offset[tid];  // Start at this thread's offset

		for (int t = tid; t < total_pairs; t += blockDim.x)
		{
			if (d_out_min[voxel_base + t] <= threshold)
			{
				int v1 = t / n2;
				int v2 = t % n2;

				size_t idx = out_base + write_pos * 3;
				d_valid_pairs[idx] = v1;
				d_valid_pairs[idx + 1] = v2;
				d_valid_pairs[idx + 2] = t;

				// idx = out_base_2 + write_pos * 2;
				// d_valid_pairs_dist[idx] = d_out_min[voxel_base + t];
				// d_valid_pairs_dist[idx + 1] = d_out_max[voxel_base + t];

				write_pos++;  // Increment within this thread's allocation
			}
		}
	}

	__global__ void fill_valid_voxel_distance(
		const size_t* d_prefix_obj,      // 全局 voxel prefix
		const size_t* d_valid_prefix,    // valid voxel prefix（按 pair）
		const int*    d_valid_pairs,     // [v1, v2, t]
		const float*  d_out_min,
		const float*  d_out_max,
		float*        d_valid_pairs_dist
	)
	{
		int pair_idx = blockIdx.x;
		int tid      = threadIdx.x;

		size_t valid_begin = d_valid_prefix[pair_idx];
		size_t valid_end   = d_valid_prefix[pair_idx + 1];

		size_t voxel_base  = d_prefix_obj[pair_idx];

		for (size_t k = valid_begin + tid;
			k < valid_end;
			k += blockDim.x)
		{
			int t = d_valid_pairs[3 * k + 2];

			size_t global_idx = voxel_base + t;

			d_valid_pairs_dist[2 * k]     = d_out_min[global_idx];
			d_valid_pairs_dist[2 * k + 1] = d_out_max[global_idx];
		}
	}

	__global__ void fill_valid_voxel_distance_streaming(
		const size_t* d_prefix_obj,          
		const size_t* d_chunk_valid_prefix,  
		const int*    d_valid_pairs,      
		const float*  d_out_min, 
		const float*  d_out_max,
		float*        d_valid_pairs_dist,
		int           global_pair_offset
	)
	{
		int local_pair_idx  = blockIdx.x;
		int global_pair_idx = local_pair_idx + global_pair_offset;
		int tid = threadIdx.x;

		size_t valid_begin = d_chunk_valid_prefix[local_pair_idx];
		size_t valid_end = d_chunk_valid_prefix[local_pair_idx + 1];

		size_t chunk_base = d_prefix_obj[global_pair_offset];
		size_t global_voxel_base = d_prefix_obj[global_pair_idx];
		size_t local_voxel_base = global_voxel_base - chunk_base;

		for (size_t k = valid_begin + tid; k < valid_end; k += blockDim.x)
		{
			int t = d_valid_pairs[3 * k + 2];
			size_t local_idx = local_voxel_base + t;
			d_valid_pairs_dist[2 * k] = d_out_min[local_idx];
			d_valid_pairs_dist[2 * k + 1] = d_out_max[local_idx];
		}
	}

	__global__ void record_valid_voxel_pairs_streaming(
		DeviceVoxels dv1,
		DeviceVoxels dv2,
		const int *d_obj1,
		const int *d_obj2,
		const size_t *d_prefix_obj,
		const float *d_out_min,			// voxel-level min distance
		const float *d_out_max,			// voxel-level max distance
		const float *d_min_max_dist,	// object-level distance
		const size_t *d_chunk_valid_prefix,
		int *d_valid_pairs,
		float *d_valid_pairs_dist,
		int global_pair_offset  
	)
	{
		__shared__ int sh_count[1024];
		__shared__ int sh_offset[1024];

		int local_pair_idx  = blockIdx.x;
		int global_pair_idx = local_pair_idx + global_pair_offset;
		int tid = threadIdx.x;

		int obj1 = d_obj1[global_pair_idx];
		int obj2 = d_obj2[global_pair_idx];

		int n1 = dv1.d_voxel_count[obj1];
		int n2 = dv2.d_voxel_count[obj2];
		int total_pairs = n1 * n2;


		size_t chunk_base = d_prefix_obj[global_pair_offset]; 

		size_t global_voxel_base = d_prefix_obj[global_pair_idx];

		size_t local_voxel_base = global_voxel_base - chunk_base;

		float threshold = d_min_max_dist[global_pair_idx];

		size_t out_base = d_chunk_valid_prefix[local_pair_idx];

		
		// obtain local_count
		int local_count = 0;

		for (int t = tid; t < total_pairs; t += blockDim.x)
		{
			if (d_out_min[local_voxel_base + t] <= threshold)
				local_count++;
		}

		sh_count[tid] = local_count;
		__syncthreads();

		// inclusive scan
		for (int stride = 1; stride < blockDim.x; stride <<= 1)
		{
			int val = 0;
			if (tid >= stride)
				val = sh_count[tid - stride];

			__syncthreads();
			sh_count[tid] += val;
			__syncthreads();
		}

		// convert to exclusive
		int thread_offset = sh_count[tid] - local_count;
		__syncthreads();

		// write into buffer
		int write_pos = thread_offset;

		for (int t = tid; t < total_pairs; t += blockDim.x)
		{
			if (d_out_min[local_voxel_base + t] <= threshold)
			{
				int v1 = t / n2;
				int v2 = t % n2;

				size_t idx = out_base + write_pos;

				d_valid_pairs[3*idx + 0] = v1;
				d_valid_pairs[3*idx + 1] = v2;
				d_valid_pairs[3*idx + 2] = t;


				// deliberately put in another kernel
				// d_valid_pairs_dist[2*idx + 0] = d_out_min[local_voxel_base + t];
				// d_valid_pairs_dist[2*idx + 1] = d_out_max[local_voxel_base + t];

				write_pos++;
			}
		}
	}

	__global__ void record_valid_voxel_pairs_streaming_knn(
		DeviceVoxels dv1,
		DeviceVoxels dv2,
		const int *d_obj1,
		const int *d_obj2,
		const size_t *d_prefix_obj,
		const float *d_out_min,			// voxel-level min distance
		const float *d_out_max,			// voxel-level max distance
		const float *d_min_max_dist,	// object-level distance
		const size_t *d_chunk_valid_prefix,
		int *d_valid_pairs,
		float *d_valid_pairs_dist,
		int global_pair_offset,
		int *d_confirmed_after,
		int *d_status_after,
		int K
	)
	{
		__shared__ int sh_count[1024];
		__shared__ int sh_offset[1024];

		int local_pair_idx  = blockIdx.x;
		int global_pair_idx = local_pair_idx + global_pair_offset;
		int tid = threadIdx.x;

		int obj1 = d_obj1[global_pair_idx];
		int obj2 = d_obj2[global_pair_idx];

		if (d_confirmed_after[obj1] == K) return;
		if (d_status_after[global_pair_idx] != 0) return;

		int n1 = dv1.d_voxel_count[obj1];
		int n2 = dv2.d_voxel_count[obj2];
		int total_pairs = n1 * n2;


		size_t chunk_base = d_prefix_obj[global_pair_offset]; 

		size_t global_voxel_base = d_prefix_obj[global_pair_idx];

		size_t local_voxel_base = global_voxel_base - chunk_base;

		float threshold = d_min_max_dist[global_pair_idx];

		size_t out_base = d_chunk_valid_prefix[local_pair_idx];

		
		// obtain local_count
		int local_count = 0;

		for (int t = tid; t < total_pairs; t += blockDim.x)
		{
			if (d_out_min[local_voxel_base + t] <= threshold)
				local_count++;
		}

		sh_count[tid] = local_count;
		__syncthreads();

		// inclusive scan
		for (int stride = 1; stride < blockDim.x; stride <<= 1)
		{
			int val = 0;
			if (tid >= stride)
				val = sh_count[tid - stride];

			__syncthreads();
			sh_count[tid] += val;
			__syncthreads();
		}

		// convert to exclusive
		int thread_offset = sh_count[tid] - local_count;
		__syncthreads();

		// write into buffer
		int write_pos = thread_offset;

		for (int t = tid; t < total_pairs; t += blockDim.x)
		{
			if (d_out_min[local_voxel_base + t] <= threshold)
			{
				int v1 = t / n2;
				int v2 = t % n2;

				size_t idx = out_base + write_pos;

				d_valid_pairs[3*idx + 0] = v1;
				d_valid_pairs[3*idx + 1] = v2;
				d_valid_pairs[3*idx + 2] = t;


				// deliberately put in another kernel
				// d_valid_pairs_dist[2*idx + 0] = d_out_min[local_voxel_base + t];
				// d_valid_pairs_dist[2*idx + 1] = d_out_max[local_voxel_base + t];

				write_pos++;
			}
		}
	}

	__global__ void record_valid_voxel_pairs_streaming_within(
		DeviceVoxels dv1,
		DeviceVoxels dv2,
		const int *d_obj1,
		const int *d_obj2,
		const size_t *d_prefix_obj,
		const float *d_out_min,			// voxel-level min distance
		const float *d_out_max,			// voxel-level max distance
		const float *d_min_max_dist,	// object-level distance
		const float *d_min_min_dist,
		const size_t *d_chunk_valid_prefix,
		int *d_valid_pairs,
		float *d_valid_pairs_dist,
		int global_pair_offset,
		float within_distance 
	)
	{
		__shared__ int sh_count[1024];
		__shared__ int sh_offset[1024];

		int local_pair_idx  = blockIdx.x;
		int global_pair_idx = local_pair_idx + global_pair_offset;
		int tid = threadIdx.x;

		int obj1 = d_obj1[global_pair_idx];
		int obj2 = d_obj2[global_pair_idx];

		int n1 = dv1.d_voxel_count[obj1];
		int n2 = dv2.d_voxel_count[obj2];
		int total_pairs = n1 * n2;


		size_t chunk_base = d_prefix_obj[global_pair_offset]; 

		size_t global_voxel_base = d_prefix_obj[global_pair_idx];

		size_t local_voxel_base = global_voxel_base - chunk_base;

		float threshold = d_min_max_dist[global_pair_idx];

		size_t out_base = d_chunk_valid_prefix[local_pair_idx];

		
		// obtain local_count
		int local_count = 0;

		if (d_min_max_dist[global_pair_idx] > within_distance && 
			d_min_min_dist[global_pair_idx] <= within_distance) 
		{
			for (int t = tid; t < total_pairs; t += blockDim.x)
			{
				float mindist = d_out_min[local_voxel_base + t];
				if (mindist <= threshold &&
					mindist <= within_distance)
					local_count++;
			}
		}

		sh_count[tid] = local_count;
		__syncthreads();

		// inclusive scan
		for (int stride = 1; stride < blockDim.x; stride <<= 1)
		{
			int val = 0;
			if (tid >= stride)
				val = sh_count[tid - stride];

			__syncthreads();
			sh_count[tid] += val;
			__syncthreads();
		}

		// convert to exclusive
		int thread_offset = sh_count[tid] - local_count;
		__syncthreads();

		// write into buffer
		int write_pos = thread_offset;

		if (d_min_max_dist[global_pair_idx] > within_distance && 
			d_min_min_dist[global_pair_idx] <= within_distance) 
		{
			for (int t = tid; t < total_pairs; t += blockDim.x)
			{
				float mindist = d_out_min[local_voxel_base + t];

				if (mindist <= threshold &&
					mindist <= within_distance)
				{
					int v1 = t / n2;
					int v2 = t % n2;

					size_t idx = out_base + write_pos;

					d_valid_pairs[3*idx + 0] = v1;
					d_valid_pairs[3*idx + 1] = v2;
					d_valid_pairs[3*idx + 2] = t;

					// deliberately put in another kernel
					// d_valid_pairs_dist[2*idx + 0] = d_out_min[local_voxel_base + t];
					// d_valid_pairs_dist[2*idx + 1] = d_out_max[local_voxel_base + t];

					write_pos++;
				}
			}
		}
	}

	__device__ __forceinline__ int warpReduceSum(int v)
	{
		for (int offset = 16; offset > 0; offset >>= 1)
			v += __shfl_down_sync(0xffffffff, v, offset);
		return v;
	}

	// __global__ void mark_candidate_status(
	// 	const float *min_dists,				// object-level min distance
	// 	const float *max_dists,				// object-level max distance
	// 	const int *obj_pair_prefix,
	// 	int K,
	// 	int obj_num
	// )
	// {
	// 	// Warp-per-object mapping
	// 	int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
	// 	int warp_id    = global_tid >> 5;     // /32
	// 	int lane       = threadIdx.x & 31;    // lane id

	// 	int obj = warp_id;
	// 	if (obj >= obj_num) return;

	// 	int begin = obj_pair_prefix[obj];
	// 	int end   = obj_pair_prefix[obj + 1];
	// 	int cand_cnt = end - begin;

	// 	// Bulk-synchronous: K_left is constant in this kernel
	// 	int K_left = K;

	// 	int local_confirmed = 0;

	// 	// IMPORTANT: must scan ALL candidates (no early break)
	// 	for (int c = 0; c < cand_cnt; ++c)
	// 	{
	// 		int idx_i = begin + c;

	// 		// Carry over decided states
	// 		if (status_before[idx_i] != UNDECIDED)
	// 		{
	// 			if (lane == 0)
	// 				status_after[idx_i] = status_before[idx_i];
	// 			__syncwarp();
	// 			continue;
	// 		}

	// 		float min_i = min_dists[idx_i];
	// 		float max_i = max_dists[idx_i];

	// 		int sure  = 0;
	// 		int maybe = 0;

	// 		// Compare with other UNDECIDED candidates
	// 		for (int j = lane; j < cand_cnt; j += 32)
	// 		{
	// 			if (j == c) continue;

	// 			int idx_j = begin + j;
	// 			if (status_before[idx_j] != UNDECIDED)
	// 				continue;

	// 			float min_j = min_dists[idx_j];
	// 			float max_j = max_dists[idx_j];

	// 			if (max_j <= min_i)        sure++;
	// 			if (!(max_i <= min_j))    maybe++;
	// 		}

	// 		sure  = warpReduceSum(sure);
	// 		maybe = warpReduceSum(maybe);

	// 		if (lane == 0)
	// 		{
	// 			if (maybe < K_left)
	// 			{
	// 				status_after[idx_i] = CONFIRMED;
	// 				local_confirmed++;           // bulk confirmation allowed
	// 			}
	// 			else if (sure >= K_left)
	// 			{
	// 				status_after[idx_i] = REMOVED;
	// 			}
	// 			else
	// 			{
	// 				status_after[idx_i] = UNDECIDED;
	// 			}
	// 		}

	// 		__syncwarp();
	// 	}

	// 	// Write per-object confirmed count (lower bound / bulk result)
	// 	if (lane == 0)
	// 		confirmed_after[obj] = confirmed_before[obj] + local_confirmed;
	// }

	__global__ void mark_candidate_status_warp(
        const float *min_dists,                // object-level min distance
        const float *max_dists,                // object-level max distance
        const int *obj_pair_prefix,
        const int *obj_pair_indices,         // NOT USED!
        int *confirmed_before,
        int *confirmed_after,
        int K,
        int obj_num,
        int *status_before,
        int *status_after
    )
    {
        // Warp-per-object mapping
        int global_tid = blockIdx.x * blockDim.x + threadIdx.x;

        if (global_tid == 0) {

            for (int obj = 0; obj < obj_num; ++obj) {

                int begin = obj_pair_prefix[obj];
                int end   = obj_pair_prefix[obj + 1];
                int cand_cnt = end - begin;
            
                // int K_left = K - confirmed_before[obj];

                // int local_confirmed = 0;

                // IMPORTANT: must scan ALL candidates (no early break)
                for (int c = 0; c < cand_cnt && K > confirmed_after[obj]; ++c)
                {
                    int idx_i = begin + c;

                    // Carry over decided states
                    if (status_after[idx_i] != UNDECIDED)
                    {
                        continue;
                    }

                    float min_i = min_dists[idx_i];
                    float max_i = max_dists[idx_i];

                    int sure  = 0;
                    int maybe = 0;

                    // Compare with other UNDECIDED candidates
                    for (int j = 0; j < cand_cnt; j += 1)
                    {
                        if (j == c) continue;

                        int idx_j = begin + j;
                        if (status_after[idx_j] != UNDECIDED)
                            continue;

                        float min_j = min_dists[idx_j];
                        float max_j = max_dists[idx_j];

                        if (max_j <= min_i)        sure++;
                        if (!(max_i <= min_j))    maybe++;
                    }
                    
                    if (maybe < K - confirmed_after[obj])
                    {
                        status_after[idx_i] = CONFIRMED;
						confirmed_after[obj]++;
                    }
                    else if (sure >= K - confirmed_after[obj])
                    {
                        status_after[idx_i] = REMOVED;
                    }
                    else
                    {
                        status_after[idx_i] = UNDECIDED;
                    }
                }
                // confirmed_after[obj] = confirmed_before[obj] + local_confirmed;
            }
        }
    }

	void mark_candidate_status_cpu(
		const std::vector<float> &min_dists,     // candidate-level
		const std::vector<float> &max_dists,     // candidate-level
		const std::vector<int> &prefix_tile,   // obj -> candidate prefix
		const std::vector<int> &confirmed_before,
		std::vector<int> &confirmed_after,
		int K,
		int obj_num,
		const std::vector<int> &status_before, // UNDECIDED / CONFIRMED / REMOVED
		std::vector<int> &status_after
	)
	{
		// 对每个 object 独立处理（等价 warp-per-object）
		for (int obj = 0; obj < obj_num; ++obj)
		{
			int begin = prefix_tile[obj];
			int end   = prefix_tile[obj + 1];
			int cand_cnt = end - begin;

			int K_left = K - confirmed_before[obj];
			int local_confirmed = 0;

			for (int c = 0; c < cand_cnt && K_left > 0; ++c)
			{
				int idx_i = begin + c;


				if (status_before[idx_i] != UNDECIDED)
				{
					status_after[idx_i] = status_before[idx_i];
					continue;
				}

				float min_i = min_dists[idx_i];
				float max_i = max_dists[idx_i];

				int sure  = 0;
				int maybe = 0;

				// GPU：只和 UNDECIDED 比较
				for (int j = 0; j < cand_cnt; ++j)
				{
					if (j == c) continue;

					int idx_j = begin + j;
					if (status_before[idx_j] != UNDECIDED)
						continue;

					float min_j = min_dists[idx_j];
					float max_j = max_dists[idx_j];

					if (max_j <= min_i)
						sure++;

					if (!(max_i <= min_j))
						maybe++;
				}

				// 完全等价 GPU 判定
				if (maybe < K_left)
				{
					status_after[idx_i] = CONFIRMED;
					local_confirmed++;
					K_left--;
				}
				else if (sure >= K_left)
				{
					status_after[idx_i] = REMOVED;
				}
				else
				{
					status_after[idx_i] = UNDECIDED;
				}
			}

			// GPU 的 confirmed_after 语义（bulk，下界/上界）
			confirmed_after[obj] = confirmed_before[obj] + local_confirmed;
		}
	}



	/**
	 * This is the kernel invoking for k-NN query!!!
	 */
	void compute_voxel_pair_distance_for_all(const DeviceVoxels &dv1,
											 const DeviceVoxels &dv2,
											 vector<int> &prefix_tile,
											 vector<size_t> &prefix_obj,
											 vector<int> &compute_obj_1,
											 vector<int> &compute_obj_2,
											 size_t total_voxel_size,
											 size_t max_voxel_size,
											 vector<float> &h_min,
											 vector<float> &h_max,
											 vector<float> &min_max_dist,
											 vector<float> &min_min_dist,
											 vector<int> &valid_voxel_cnt,
											 vector<size_t> &valid_voxel_prefix,
											 vector<int> &valid_voxel_pairs,
											 int K,
											 vector<int> &confirmed_after,
											 vector<int> &status_after)
	{
		int pair_num = compute_obj_1.size();

		// 1. 分配 device memory
		// int *d_obj1, *d_obj2;
		CUDA_SAFE_CALL(cudaMalloc(&gd.d_obj1, pair_num * sizeof(int)));
		CUDA_SAFE_CALL(cudaMalloc(&gd.d_obj2, pair_num * sizeof(int)));
		CUDA_SAFE_CALL(cudaMemcpy(gd.d_obj1, compute_obj_1.data(), pair_num * sizeof(int), cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(gd.d_obj2, compute_obj_2.data(), pair_num * sizeof(int), cudaMemcpyHostToDevice));

		// int *d_prefix_obj;
		CUDA_SAFE_CALL(cudaMalloc(&gd.d_prefix_obj, (pair_num + 1) * sizeof(size_t)));
		CUDA_SAFE_CALL(cudaMemcpy(gd.d_prefix_obj, prefix_obj.data(), (pair_num + 1) * sizeof(size_t), cudaMemcpyHostToDevice));

		// int *d_prefix_tile;
		CUDA_SAFE_CALL(cudaMalloc(&gd.d_prefix_tile, (dv1.tile_size + 1) * sizeof(int)));
		CUDA_SAFE_CALL(cudaMemcpy(gd.d_prefix_tile, prefix_tile.data(), (dv1.tile_size + 1) * sizeof(int), cudaMemcpyHostToDevice));

		// float *d_out_min, *d_out_max;
		// use unified memory since it can be large
		CUDA_SAFE_CALL(cudaMallocManaged(&gd.d_out_min, total_voxel_size * sizeof(float)));
		CUDA_SAFE_CALL(cudaMallocManaged(&gd.d_out_max, total_voxel_size * sizeof(float)));

		// float *d_min_max_dist, *d_min_min_dist;
		CUDA_SAFE_CALL(cudaMalloc(&gd.d_min_max_dist, pair_num * sizeof(float)));
		CUDA_SAFE_CALL(cudaMalloc(&gd.d_min_min_dist, pair_num * sizeof(float)));

		// int *d_count;
		CUDA_SAFE_CALL(cudaMalloc(&gd.d_count, pair_num * sizeof(int)));

		// allocate memory for the third kernel
		CUDA_SAFE_CALL(cudaMalloc(&gd.d_confirmed_before, dv1.tile_size * sizeof(int)));
		CUDA_SAFE_CALL(cudaMemset(gd.d_confirmed_before, 0, dv1.tile_size * sizeof(int)));

		CUDA_SAFE_CALL(cudaMalloc(&gd.d_confirmed_after, dv1.tile_size * sizeof(int)));
		CUDA_SAFE_CALL(cudaMemset(gd.d_confirmed_after, 0, dv1.tile_size * sizeof(int)));


		// int *d_status_before;
		CUDA_SAFE_CALL(cudaMalloc(&gd.d_status_before, pair_num * sizeof(int)));
		CUDA_SAFE_CALL(cudaMemset(gd.d_status_before, 0, pair_num * sizeof(int))); // all undecided

		// int *d_status_after;
		CUDA_SAFE_CALL(cudaMalloc(&gd.d_status_after, pair_num * sizeof(int)));
		CUDA_SAFE_CALL(cudaMemset(gd.d_status_after, 0, pair_num * sizeof(int)));

		int BLKDIM = min((int)max_voxel_size, 1024);


		// if (max_voxel_size > 1024)
		// {
		// 	exit(-1);
		// }
		voxel_pair_dist_kernel_for_all<<<pair_num, BLKDIM>>>(
			dv1, dv2,
			gd.d_obj1, gd.d_obj2,
			gd.d_prefix_obj,
			gd.d_out_min, gd.d_out_max,
			gd.d_min_max_dist,
			gd.d_min_min_dist,
			gd.d_count
		);
		check_execution();
		CUDA_SAFE_CALL(cudaDeviceSynchronize());

		valid_voxel_cnt.resize(pair_num);
		CUDA_SAFE_CALL(cudaMemcpy(valid_voxel_cnt.data(), gd.d_count, pair_num * sizeof(int), cudaMemcpyDeviceToHost));

		// h_min.resize(total_voxel_size);
		// h_max.resize(total_voxel_size);
		// CUDA_SAFE_CALL(cudaMemcpy(h_min.data(), gd.d_out_min, total_voxel_size * sizeof(float), cudaMemcpyDeviceToHost));
		// CUDA_SAFE_CALL(cudaMemcpy(h_max.data(), gd.d_out_max, total_voxel_size * sizeof(float), cudaMemcpyDeviceToHost));

		min_max_dist.resize(pair_num);
		min_min_dist.resize(pair_num);
		CUDA_SAFE_CALL(cudaMemcpy(min_max_dist.data(), gd.d_min_max_dist, pair_num * sizeof(float), cudaMemcpyDeviceToHost));
		CUDA_SAFE_CALL(cudaMemcpy(min_min_dist.data(), gd.d_min_min_dist, pair_num * sizeof(float), cudaMemcpyDeviceToHost));

		//============ launch second kernel =============
		valid_voxel_prefix.assign(pair_num + 1, 0);
		for (int i = 0; i < pair_num; ++i)
		{
			valid_voxel_prefix[i + 1] = valid_voxel_prefix[i] + valid_voxel_cnt[i];
		}

		size_t total_valid_voxel_pairs = valid_voxel_prefix[pair_num];

		// int *d_valid_voxel_prefix;
		// int *d_valid_voxel_pairs;

		CUDA_SAFE_CALL(cudaMallocManaged(&gd.d_valid_voxel_prefix, (pair_num + 1) * sizeof(size_t)));

		CUDA_SAFE_CALL(cudaMemcpy(gd.d_valid_voxel_prefix, valid_voxel_prefix.data(), (pair_num + 1) * sizeof(size_t), cudaMemcpyHostToDevice));

		CUDA_SAFE_CALL(cudaMallocManaged(&gd.d_valid_voxel_pairs, 3 * total_valid_voxel_pairs * sizeof(int)));

		float *d_valid_pairs_dist;

		CUDA_SAFE_CALL(cudaMallocManaged(&d_valid_pairs_dist, 2 * total_valid_voxel_pairs * sizeof(float)));

		// this step is intended for future use, not for this step!!!!
		record_valid_voxel_pairs<<<pair_num, BLKDIM>>>(
			dv1, dv2,
			gd.d_obj1, gd.d_obj2,
			gd.d_prefix_obj,
			gd.d_out_min,
			gd.d_out_max,
			gd.d_min_max_dist,
			gd.d_valid_voxel_prefix,
			gd.d_valid_voxel_pairs,
			d_valid_pairs_dist
		);
		check_execution();
		CUDA_SAFE_CALL(cudaDeviceSynchronize());

		valid_voxel_pairs.resize(3 * total_valid_voxel_pairs);
		CUDA_SAFE_CALL(cudaMemcpy(valid_voxel_pairs.data(),
								  gd.d_valid_voxel_pairs,
								  3 * total_valid_voxel_pairs * sizeof(int),
								  cudaMemcpyDeviceToHost));

		//=========== launch third kernel ==========

		// int *d_confirmed_before, *d_confirmed_after;
	
		int BLOCK = 256;
		int GRID  = dv1.tile_size;   // 一个 tile 一个 block

		evaluate_knn_kernel<<<GRID, BLOCK>>>(
			gd.d_min_min_dist,
			gd.d_min_max_dist,
			gd.d_prefix_tile,
			gd.d_confirmed_before,
			gd.d_status_after,
			dv1.tile_size,
			K
		);
		check_execution();
		CUDA_SAFE_CALL(cudaDeviceSynchronize());

		// status 是 per pair
		status_after.resize(pair_num);
		CUDA_SAFE_CALL(cudaMemcpy(
			status_after.data(),
			gd.d_status_after,
			pair_num * sizeof(int),
			cudaMemcpyDeviceToHost));

		confirmed_after.resize(dv1.tile_size);
		for (int t = 0; t < dv1.tile_size; ++t)
		{
			int begin = prefix_tile[t];
			int end   = prefix_tile[t + 1];

			int confirmed_cnt = 0;

			for (int i = begin; i < end; ++i)
			{
				if (status_after[i] == 1)  // CONFIRMED
					confirmed_cnt++;
				
				if (confirmed_cnt == K) break;
			}
			confirmed_after[t] = confirmed_cnt;
		}
		

		// do not free temporarily
		// CUDA_SAFE_CALL(cudaFree(d_obj1));
		// CUDA_SAFE_CALL(cudaFree(d_obj2));
		CUDA_SAFE_CALL(cudaFree(gd.d_out_min));
		CUDA_SAFE_CALL(cudaFree(gd.d_out_max));
		// CUDA_SAFE_CALL(cudaFree(d_min_max_dist));
		// CUDA_SAFE_CALL(cudaFree(d_min_min_dist));
		// CUDA_SAFE_CALL(cudaFree(d_count));

		CUDA_SAFE_CALL(cudaFree(gd.d_valid_voxel_prefix));
		CUDA_SAFE_CALL(cudaFree(gd.d_valid_voxel_pairs));
	}

	/**
	 * This is the kernel invoking for k-NN query!!! 
	 * 
	 * Chunked Stream Version
	 */
	void compute_voxel_pair_distance_for_all_streaming(
		const DeviceVoxels &dv1,
		const DeviceVoxels &dv2,
		vector<int> &prefix_tile,
		vector<size_t> &prefix_obj,
		vector<int> &compute_obj_1,
		vector<int> &compute_obj_2,
		size_t total_voxel_size,
		size_t max_voxel_size,
		vector<float> &h_min,
		vector<float> &h_max,
		vector<float> &min_max_dist,
		vector<float> &min_min_dist,
		vector<int> &valid_voxel_cnt,
		vector<size_t> &valid_voxel_prefix,
		vector<int> &valid_voxel_pairs,
		vector<float> &valid_voxel_pairs_dist,
		int K,
		vector<int> &confirmed_after,
		vector<int> &status_after
	)
	{
		int pair_num = compute_obj_1.size();

		CUDA_SAFE_CALL(cudaMalloc(&gd.d_obj1, pair_num * sizeof(int)));
		CUDA_SAFE_CALL(cudaMalloc(&gd.d_obj2, pair_num * sizeof(int)));
		CUDA_SAFE_CALL(cudaMemcpy(gd.d_obj1, compute_obj_1.data(), pair_num * sizeof(int), cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(gd.d_obj2, compute_obj_2.data(), pair_num * sizeof(int), cudaMemcpyHostToDevice));

		// int *d_prefix_obj;
		CUDA_SAFE_CALL(cudaMalloc(&gd.d_prefix_obj, (pair_num + 1) * sizeof(size_t)));
		CUDA_SAFE_CALL(cudaMemcpy(gd.d_prefix_obj, prefix_obj.data(), (pair_num + 1) * sizeof(size_t), cudaMemcpyHostToDevice));

		// int *d_prefix_tile;
		CUDA_SAFE_CALL(cudaMalloc(&gd.d_prefix_tile, (dv1.tile_size + 1) * sizeof(int)));
		CUDA_SAFE_CALL(cudaMemcpy(gd.d_prefix_tile, prefix_tile.data(), (dv1.tile_size + 1) * sizeof(int), cudaMemcpyHostToDevice));

		// float *d_out_min, *d_out_max;
		// We don't allocate all at once 
		// CUDA_SAFE_CALL(cudaMalloc(&gd.d_out_min, total_voxel_size * sizeof(float)));
		// CUDA_SAFE_CALL(cudaMalloc(&gd.d_out_max, total_voxel_size * sizeof(float)));


		CUDA_SAFE_CALL(cudaMalloc(&gd.d_min_max_dist, pair_num * sizeof(float)));
		CUDA_SAFE_CALL(cudaMalloc(&gd.d_min_min_dist, pair_num * sizeof(float)));


		// allocate memory for the third kernel
		CUDA_SAFE_CALL(cudaMalloc(&gd.d_confirmed_before, dv1.tile_size * sizeof(int)));
		CUDA_SAFE_CALL(cudaMemset(gd.d_confirmed_before, 0, dv1.tile_size * sizeof(int)));

		CUDA_SAFE_CALL(cudaMalloc(&gd.d_confirmed_after, dv1.tile_size * sizeof(int)));
		CUDA_SAFE_CALL(cudaMemset(gd.d_confirmed_after, 0, dv1.tile_size * sizeof(int)));

		// int *d_status_before;
		CUDA_SAFE_CALL(cudaMalloc(&gd.d_status_before, pair_num * sizeof(int)));
		CUDA_SAFE_CALL(cudaMemset(gd.d_status_before, 0, pair_num * sizeof(int))); // all undecided

		// int *d_status_after;
		CUDA_SAFE_CALL(cudaMalloc(&gd.d_status_after, pair_num * sizeof(int)));
		CUDA_SAFE_CALL(cudaMemset(gd.d_status_after, 0, pair_num * sizeof(int)));


		size_t GPU_MEMORY_LIMIT = 20 * 1024ULL * 1024 * 1024; // 10GB
		size_t bytes_per_voxel = 2 * sizeof(float);           // min and max
		size_t MAX_CHUNK_VOXELS = GPU_MEMORY_LIMIT / bytes_per_voxel;


		valid_voxel_cnt.assign(pair_num, 0);
		valid_voxel_prefix.assign(pair_num + 1, 0);

		min_max_dist.resize(pair_num);
		min_min_dist.resize(pair_num);

		// valid_voxel_pairs.reserve(total_voxel_size * 0.05 * 3);
		// valid_voxel_pairs_dist.reserve(total_voxel_size * 0.05 * 2);

		valid_voxel_pairs.reserve(1e8 * 3);
		valid_voxel_pairs_dist.reserve(1e8 * 2);


		size_t pair_begin = 0;
		size_t old_size = 0;

		size_t BLKDIM = min(max_voxel_size, (size_t)1024);

		auto tx = std::chrono::high_resolution_clock::now();
		double pure_kernel_time = 0.0;

		while (pair_begin < pair_num)
		{

			auto t0 = std::chrono::high_resolution_clock::now();

			size_t pair_end = pair_begin;
			size_t chunk_voxel_size = 0;

			while (pair_end < pair_num)
			{
				size_t voxel_cnt = prefix_obj[pair_end + 1] - prefix_obj[pair_end];

				if (chunk_voxel_size + voxel_cnt > MAX_CHUNK_VOXELS)
					break;

				chunk_voxel_size += voxel_cnt;
				pair_end++;
			}

			size_t chunk_pair_num = pair_end - pair_begin;

			std::cout << "Chunk pairs: " << pair_begin << " - " << pair_end << std::endl;

			float *d_out_min;
			float *d_out_max;
			size_t *d_count;

			CUDA_SAFE_CALL(cudaMalloc(&d_out_min, chunk_voxel_size * sizeof(float)));

			CUDA_SAFE_CALL(cudaMalloc(&d_out_max, chunk_voxel_size * sizeof(float)));

			CUDA_SAFE_CALL(cudaMalloc(&d_count, chunk_pair_num * sizeof(size_t)));


			auto ta = std::chrono::high_resolution_clock::now();

			voxel_pair_dist_kernel_for_all_streaming<<<chunk_pair_num, BLKDIM>>>(
				dv1, dv2,
				gd.d_obj1, gd.d_obj2,
				gd.d_prefix_obj,
				d_out_min, 
				d_out_max,
				gd.d_min_max_dist,
				gd.d_min_min_dist,
				d_count,
				pair_begin
			);
			check_execution();
			CUDA_SAFE_CALL(cudaDeviceSynchronize());

			auto tb = std::chrono::high_resolution_clock::now();
			pure_kernel_time += std::chrono::duration<double>(tb - ta).count();


			size_t host_offset = prefix_obj[pair_begin];

			vector<size_t> chunk_count(chunk_pair_num);

        	CUDA_SAFE_CALL(cudaMemcpy(chunk_count.data(),
									d_count,
									chunk_pair_num * sizeof(size_t),
									cudaMemcpyDeviceToHost));


			
			
			vector<size_t> chunk_valid_prefix(chunk_pair_num + 1, 0);

			for (size_t i = 0; i < chunk_pair_num; ++i)
			{
				chunk_valid_prefix[i + 1] = chunk_valid_prefix[i] + chunk_count[i];
			}

			size_t chunk_total_valid = chunk_valid_prefix[chunk_pair_num];

			size_t global_valid_base = valid_voxel_prefix[pair_begin];
			for (size_t i = 0; i < chunk_pair_num; ++i)
			{
				valid_voxel_cnt[pair_begin + i] = chunk_count[i];
				valid_voxel_prefix[pair_begin + i + 1] = global_valid_base + chunk_valid_prefix[i + 1];
			}

			int* d_valid_pairs;
			float *d_valid_pairs_dist;
			size_t* d_chunk_prefix;

			CUDA_SAFE_CALL(cudaMalloc(&d_valid_pairs, 3 * chunk_total_valid * sizeof(int)));

			CUDA_SAFE_CALL(cudaMalloc(&d_valid_pairs_dist, 2 * chunk_total_valid * sizeof(float)));

			CUDA_SAFE_CALL(cudaMalloc(&d_chunk_prefix, (chunk_pair_num + 1) * sizeof(size_t)));

			CUDA_SAFE_CALL(cudaMemcpy(d_chunk_prefix,
						chunk_valid_prefix.data(),
						(chunk_pair_num + 1) * sizeof(size_t),
						cudaMemcpyHostToDevice));
		

			ta = std::chrono::high_resolution_clock::now();
			record_valid_voxel_pairs_streaming<<<chunk_pair_num, BLKDIM>>>(
				dv1, dv2,
				gd.d_obj1, gd.d_obj2,
				gd.d_prefix_obj,
				d_out_min,
				d_out_max,
				gd.d_min_max_dist,
				d_chunk_prefix,
				d_valid_pairs,
				d_valid_pairs_dist,
				pair_begin
			);
			check_execution();
			CUDA_SAFE_CALL(cudaDeviceSynchronize());

			fill_valid_voxel_distance_streaming<<<chunk_pair_num, BLKDIM>>>(
				gd.d_prefix_obj,      
				d_chunk_prefix,
				d_valid_pairs,
				d_out_min,
				d_out_max,
				d_valid_pairs_dist,
				pair_begin
			);
			check_execution();
			CUDA_SAFE_CALL(cudaDeviceSynchronize());

			tb = std::chrono::high_resolution_clock::now();
			pure_kernel_time += std::chrono::duration<double>(tb - ta).count();


			auto t1 = std::chrono::high_resolution_clock::now();
			std::cout << "compute = " << std::chrono::duration<double>(t1 - t0).count() << std::endl;


	
			t0 = std::chrono::high_resolution_clock::now();
			size_t old_size = valid_voxel_pairs.size();
			valid_voxel_pairs.resize(old_size + 3 * chunk_total_valid);

			size_t old_size_2 = valid_voxel_pairs_dist.size();
			valid_voxel_pairs_dist.resize(old_size_2 + 2 * chunk_total_valid);


			CUDA_SAFE_CALL(cudaMemcpy(valid_voxel_pairs.data() + old_size,
						d_valid_pairs,
						3 * chunk_total_valid * sizeof(int),
						cudaMemcpyDeviceToHost));

			CUDA_SAFE_CALL(cudaMemcpy(valid_voxel_pairs_dist.data() + old_size_2,
						d_valid_pairs_dist,
						2 * chunk_total_valid * sizeof(float),
						cudaMemcpyDeviceToHost));

			t1 = std::chrono::high_resolution_clock::now();
			std::cout << "memcpy = " << std::chrono::duration<double>(t1 - t0).count() << std::endl;


			CUDA_SAFE_CALL(cudaFree(d_out_min));
			CUDA_SAFE_CALL(cudaFree(d_out_max));
			CUDA_SAFE_CALL(cudaFree(d_count));
			CUDA_SAFE_CALL(cudaFree(d_valid_pairs));
			CUDA_SAFE_CALL(cudaFree(d_valid_pairs_dist));
			CUDA_SAFE_CALL(cudaFree(d_chunk_prefix));

			pair_begin = pair_end;
		}

		std::cout << "total valid voxel = " << valid_voxel_prefix[pair_num] << std::endl;

		auto ty = std::chrono::high_resolution_clock::now();
		std::cout << "while time = " << std::chrono::duration<double>(ty - tx).count() << std::endl;
		std::cout << "pure kernel time = " << pure_kernel_time << std::endl;


		//=========== launch third kernel ==========

		auto t0 = std::chrono::high_resolution_clock::now();

		int BLOCK = 256;
		int GRID  = dv1.tile_size;   // 一个 tile 一个 block

		evaluate_knn_kernel<<<GRID, BLOCK>>>(
			gd.d_min_min_dist,
			gd.d_min_max_dist,
			gd.d_prefix_tile,
			gd.d_confirmed_before,
			gd.d_status_after,
			dv1.tile_size,
			K
		);
		check_execution();
		CUDA_SAFE_CALL(cudaDeviceSynchronize());

		// status 是 per pair
		status_after.resize(pair_num);
		CUDA_SAFE_CALL(cudaMemcpy(
			status_after.data(),
			gd.d_status_after,
			pair_num * sizeof(int),
			cudaMemcpyDeviceToHost));

		confirmed_after.resize(dv1.tile_size);
		for (int t = 0; t < dv1.tile_size; ++t)
		{
			int begin = prefix_tile[t];
			int end   = prefix_tile[t + 1];

			int confirmed_cnt = 0;

			for (int i = begin; i < end; ++i)
			{
				if (status_after[i] == 1)  // CONFIRMED
					confirmed_cnt++;
				
				if (confirmed_cnt == K) break;
			}
			confirmed_after[t] = confirmed_cnt;
		}

		auto t1 = std::chrono::high_resolution_clock::now();
		std::cout << "filtering stage evaluate time = " << std::chrono::duration<double>(t1 - t0).count() << std::endl;
	}

	// void compute_voxel_pair_distance_for_all_streaming_pipeline(
	// 	const DeviceVoxels &dv1,
	// 	const DeviceVoxels &dv2,
	// 	vector<int> &prefix_tile,
	// 	vector<size_t> &prefix_obj,
	// 	vector<int> &compute_obj_1,
	// 	vector<int> &compute_obj_2,
	// 	size_t total_voxel_size,
	// 	size_t max_voxel_size,
	// 	vector<float> &h_min,
	// 	vector<float> &h_max,
	// 	vector<float> &min_max_dist,
	// 	vector<float> &min_min_dist,
	// 	vector<int> &valid_voxel_cnt,
	// 	vector<size_t> &valid_voxel_prefix,
	// 	vector<int> &valid_voxel_pairs,
	// 	vector<float> &valid_voxel_pairs_dist,
	// 	int K,
	// 	vector<int> &confirmed_after,
	// 	vector<int> &status_after
	// ) 
	// {
	// 	int pair_num = compute_obj_1.size();

	// 	CUDA_SAFE_CALL(cudaMalloc(&gd.d_obj1, pair_num * sizeof(int)));
	// 	CUDA_SAFE_CALL(cudaMalloc(&gd.d_obj2, pair_num * sizeof(int)));
	// 	CUDA_SAFE_CALL(cudaMemcpy(gd.d_obj1, compute_obj_1.data(), pair_num * sizeof(int), cudaMemcpyHostToDevice));
	// 	CUDA_SAFE_CALL(cudaMemcpy(gd.d_obj2, compute_obj_2.data(), pair_num * sizeof(int), cudaMemcpyHostToDevice));

	// 	// int *d_prefix_obj;
	// 	CUDA_SAFE_CALL(cudaMalloc(&gd.d_prefix_obj, (pair_num + 1) * sizeof(size_t)));
	// 	CUDA_SAFE_CALL(cudaMemcpy(gd.d_prefix_obj, prefix_obj.data(), (pair_num + 1) * sizeof(size_t), cudaMemcpyHostToDevice));

	// 	// int *d_prefix_tile;
	// 	CUDA_SAFE_CALL(cudaMalloc(&gd.d_prefix_tile, (dv1.tile_size + 1) * sizeof(int)));
	// 	CUDA_SAFE_CALL(cudaMemcpy(gd.d_prefix_tile, prefix_tile.data(), (dv1.tile_size + 1) * sizeof(int), cudaMemcpyHostToDevice));

	// 	// float *d_out_min, *d_out_max;
	// 	// We don't allocate all at once 
	// 	// CUDA_SAFE_CALL(cudaMalloc(&gd.d_out_min, total_voxel_size * sizeof(float)));
	// 	// CUDA_SAFE_CALL(cudaMalloc(&gd.d_out_max, total_voxel_size * sizeof(float)));

	// 	// float *d_min_max_dist, *d_min_min_dist;
	// 	CUDA_SAFE_CALL(cudaMalloc(&gd.d_min_max_dist, pair_num * sizeof(float)));
	// 	CUDA_SAFE_CALL(cudaMalloc(&gd.d_min_min_dist, pair_num * sizeof(float)));

	// 	// int *d_count;
	// 	// CUDA_SAFE_CALL(cudaMalloc(&gd.d_count, pair_num * sizeof(int)));

	// 	// allocate memory for the third kernel
	// 	CUDA_SAFE_CALL(cudaMalloc(&gd.d_confirmed_before, dv1.tile_size * sizeof(int)));
	// 	CUDA_SAFE_CALL(cudaMemset(gd.d_confirmed_before, 0, dv1.tile_size * sizeof(int)));

	// 	CUDA_SAFE_CALL(cudaMalloc(&gd.d_confirmed_after, dv1.tile_size * sizeof(int)));
	// 	CUDA_SAFE_CALL(cudaMemset(gd.d_confirmed_after, 0, dv1.tile_size * sizeof(int)));

	// 	// int *d_status_before;
	// 	CUDA_SAFE_CALL(cudaMalloc(&gd.d_status_before, pair_num * sizeof(int)));
	// 	CUDA_SAFE_CALL(cudaMemset(gd.d_status_before, 0, pair_num * sizeof(int))); // all undecided

	// 	// int *d_status_after;
	// 	CUDA_SAFE_CALL(cudaMalloc(&gd.d_status_after, pair_num * sizeof(int)));
	// 	CUDA_SAFE_CALL(cudaMemset(gd.d_status_after, 0, pair_num * sizeof(int)));


	// 	size_t GPU_MEMORY_LIMIT = 25ULL * 1024 * 1024 * 1024; // 10GB
	// 	size_t bytes_per_voxel = 2 * sizeof(float);           // min and max
	// 	size_t MAX_CHUNK_VOXELS = GPU_MEMORY_LIMIT / bytes_per_voxel;

	// 	size_t MAX_VALID = MAX_CHUNK_VOXELS * 0.05;


	// 	vector<vector<size_t>> global_chunk_valid_prefix;


	// 	valid_voxel_cnt.assign(pair_num, 0);
	// 	valid_voxel_prefix.assign(pair_num + 1, 0);

	// 	min_max_dist.resize(pair_num);
	// 	min_min_dist.resize(pair_num);

	// 	valid_voxel_pairs.reserve(total_voxel_size * 0.05 * 3);
	// 	valid_voxel_pairs_dist.reserve(total_voxel_size * 0.05 * 2);

	// 	cudaHostRegister(valid_voxel_pairs.data(),
    //              total_voxel_size * 0.05 * 3 * sizeof(int),
    //              cudaHostRegisterDefault);

	// 	cudaHostRegister(valid_voxel_pairs_dist.data(),
    //              total_voxel_size * 0.05 * 3 * sizeof(float),
    //              cudaHostRegisterDefault);


	// 	size_t pair_begin = 0;

	// 	cudaStream_t compute_stream;
	// 	cudaStream_t memcpy_stream;

	// 	cudaStreamCreate(&compute_stream);
	// 	cudaStreamCreate(&memcpy_stream);

	// 	cudaEvent_t compute_done_event;
	// 	cudaEventCreate(&compute_done_event);

	// 	cudaEvent_t copy_done_event;
	// 	cudaEventCreate(&copy_done_event);

	// 	// 上一轮的 device 指针
	// 	int *prev_d_valid_pairs = nullptr;
	// 	float *prev_d_valid_pairs_dist = nullptr;

	// 	bool first_iter = true;

	// 	auto tx = std::chrono::high_resolution_clock::now();

	// 	while (pair_begin < pair_num)
	// 	{
	// 		size_t pair_end = pair_begin;
	// 		size_t chunk_voxel_size = 0;

	// 		while (pair_end < pair_num)
	// 		{
	// 			size_t voxel_cnt = prefix_obj[pair_end + 1] - prefix_obj[pair_end];

	// 			if (chunk_voxel_size + voxel_cnt > MAX_CHUNK_VOXELS)
	// 				break;

	// 			chunk_voxel_size += voxel_cnt;
	// 			pair_end++;
	// 		}

	// 		size_t chunk_pair_num = pair_end - pair_begin;

	// 		std::cout << "Chunk pairs: " << pair_begin << " - " << pair_end << std::endl;

	// 		float *d_out_min;
	// 		float *d_out_max;
	// 		size_t *d_count;
			
	// 		cudaMallocAsync(&d_out_min, chunk_voxel_size * sizeof(float), compute_stream);
	// 		cudaMallocAsync(&d_out_max, chunk_voxel_size * sizeof(float), compute_stream);
	// 		cudaMallocAsync(&d_count, chunk_pair_num  * sizeof(size_t), compute_stream);


	// 		voxel_pair_dist_kernel_for_all_streaming<<<
	// 			chunk_pair_num, max_voxel_size, 0, compute_stream>>>(
	// 			dv1, dv2,
	// 			gd.d_obj1, gd.d_obj2,
	// 			gd.d_prefix_obj,
	// 			d_out_min,
	// 			d_out_max,
	// 			gd.d_min_max_dist,
	// 			gd.d_min_min_dist,
	// 			d_count,
	// 			pair_begin
	// 		);

	// 		// std::vector<int> chunk_count(chunk_pair_num);
	// 		// cudaMemcpyAsync(chunk_count.data(),
	// 		// 				d_count,
	// 		// 				chunk_pair_num * sizeof(int),
	// 		// 				cudaMemcpyDeviceToHost,
	// 		// 				compute_stream);

	// 		// cudaStreamSynchronize(compute_stream);
	// 		std::vector<size_t> chunk_valid_prefix(chunk_pair_num + 1, 0);
	// 		// for (size_t i = 0; i < chunk_pair_num; ++i)
	// 		// 	chunk_valid_prefix[i + 1] = chunk_valid_prefix[i] + chunk_count[i];
	// 		// size_t chunk_total_valid = chunk_valid_prefix[chunk_pair_num];

	// 		// use GPU to do prefix sum
	// 		size_t *d_chunk_prefix;
	// 		cudaMallocAsync(&d_chunk_prefix, (chunk_pair_num + 1) * sizeof(size_t), compute_stream);
	// 		exclusive_scan_async(d_count, d_chunk_prefix, chunk_pair_num, compute_stream);
	// 		cudaMemcpyAsync(chunk_valid_prefix.data(),
	// 						d_chunk_prefix,
	// 						(chunk_pair_num + 1) * sizeof(size_t),
	// 						cudaMemcpyDeviceToHost,
	// 						compute_stream
	// 					);


	// 		// global_chunk_valid_prefix.push_back(chunk_valid_prefix);

	// 		int*    d_valid_pairs;
	// 		float*  d_valid_pairs_dist;
	// 		// size_t* d_chunk_prefix;

	// 		// CUDA_SAFE_CALL(cudaMalloc(&d_valid_pairs,
	// 		// 		3 * chunk_total_valid * sizeof(int)));

	// 		// CUDA_SAFE_CALL(cudaMalloc(&d_valid_pairs_dist,
	// 		// 		2 * chunk_total_valid * sizeof(float)));

	// 		cudaMallocAsync(&d_valid_pairs,
	// 				3 * MAX_VALID * sizeof(int), compute_stream);

	// 		cudaMallocAsync(&d_valid_pairs_dist,
	// 				2 * MAX_VALID * sizeof(float), compute_stream);


	// 		// CUDA_SAFE_CALL(cudaMemcpy(d_chunk_prefix,
	// 		// 				chunk_valid_prefix.data(),
	// 		// 				(chunk_pair_num + 1) * sizeof(size_t),
	// 		// 				cudaMemcpyHostToDevice));


	// 		record_valid_voxel_pairs_streaming<<<
	// 			chunk_pair_num, max_voxel_size, 0, compute_stream>>>(
	// 			dv1, dv2,
	// 			gd.d_obj1, gd.d_obj2,
	// 			gd.d_prefix_obj,
	// 			d_out_min,
	// 			d_out_max,
	// 			gd.d_min_max_dist,
	// 			d_chunk_prefix,
	// 			d_valid_pairs,
	// 			d_valid_pairs_dist,
	// 			pair_begin
	// 		);

	// 		fill_valid_voxel_distance_streaming<<<
	// 			chunk_pair_num, max_voxel_size, 0, compute_stream>>>(
	// 			gd.d_prefix_obj,
	// 			d_chunk_prefix,
	// 			d_valid_pairs,
	// 			d_out_min,
	// 			d_out_max,
	// 			d_valid_pairs_dist,
	// 			pair_begin
	// 		);

	// 		// 记录 compute 完成
	// 		cudaEventRecord(compute_done_event, compute_stream);

	// 		// 让 memcpy_stream 等 compute 完成
	// 		cudaStreamWaitEvent(memcpy_stream, compute_done_event, 0);


	// 		if (!first_iter)
	// 		{
	// 			cudaEventSynchronize(copy_done_event);
	// 			cudaFreeAsync(prev_d_valid_pairs, compute_stream);
	// 			cudaFreeAsync(prev_d_valid_pairs_dist, compute_stream);
	// 		}

	// 		auto t0 = std::chrono::high_resolution_clock::now();

	// 		// cudaStreamSynchronize(compute_stream);
	// 		// size_t chunk_total_valid = chunk_valid_prefix[chunk_pair_num];

	// 		size_t old_size = valid_voxel_pairs.size();
	// 		// valid_voxel_pairs.resize(old_size + 3 * MAX_VALID);

	// 		size_t old_size_2 = valid_voxel_pairs_dist.size();
	// 		// valid_voxel_pairs_dist.resize(old_size_2 + 2 * MAX_VALID);
			
			

	// 		cudaMemcpyAsync(valid_voxel_pairs.data() + old_size,
	// 						d_valid_pairs,
	// 						3 * MAX_VALID * sizeof(int),
	// 						cudaMemcpyDeviceToHost,
	// 						memcpy_stream
	// 					);

	// 		cudaMemcpyAsync(valid_voxel_pairs_dist.data() + old_size_2,
	// 						d_valid_pairs_dist,
	// 						2 * MAX_VALID * sizeof(float),
	// 						cudaMemcpyDeviceToHost,
	// 						memcpy_stream
	// 					);
 
	// 		cudaEventRecord(copy_done_event, memcpy_stream);

	// 		prev_d_valid_pairs = d_valid_pairs;
    // 		prev_d_valid_pairs_dist = d_valid_pairs_dist;

	// 		auto t1 = std::chrono::high_resolution_clock::now();
	// 		std::cout << "async copy time = " << std::chrono::duration<double>(t1 - t0).count() << std::endl;

	// 		pair_begin = pair_end;

	// 		first_iter = false;


	// 		cudaFreeAsync(d_out_min, compute_stream);
	// 		cudaFreeAsync(d_out_max, compute_stream);
	// 		cudaFreeAsync(d_count, compute_stream);
	// 		cudaFreeAsync(d_chunk_prefix, compute_stream);
	// 	}

	// 	cudaEventSynchronize(copy_done_event);

	// 	if (prev_d_valid_pairs)
	// 	{
	// 		cudaFree(prev_d_valid_pairs);
	// 		cudaFree(prev_d_valid_pairs_dist);
	// 	}

	// 	auto ty = std::chrono::high_resolution_clock::now();
	// 	std::cout << "while time = " << std::chrono::duration<double>(ty - tx).count() << std::endl;

	// 	cudaStreamSynchronize(memcpy_stream);
	// 	cudaEventDestroy(compute_done_event);
	// 	cudaEventDestroy(copy_done_event);
	// 	cudaStreamDestroy(compute_stream);
	// 	cudaStreamDestroy(memcpy_stream);


	// 	// recover
	// 	size_t global_valid_base = 0;
	// 	size_t global_index = 0;

	// 	for (size_t c = 0; c < global_chunk_valid_prefix.size(); ++c)
	// 	{
	// 		const auto& chunk_prefix = global_chunk_valid_prefix[c];

	// 		size_t chunk_pair_num = chunk_prefix.size() - 1;

	// 		for (size_t i = 0; i < chunk_pair_num; ++i)
	// 		{
	// 			// 从 prefix 反推 count
	// 			size_t local_count = chunk_prefix[i + 1] - chunk_prefix[i];

	// 			valid_voxel_cnt[global_index] = local_count;

	// 			valid_voxel_prefix[global_index + 1] =
	// 				global_valid_base + chunk_prefix[i + 1];

	// 			global_index++;
	// 		}

	// 		// 更新 global base
	// 		global_valid_base += chunk_prefix.back();
	// 	}

	// 	std::cout << "total valid voxel = " << valid_voxel_prefix[pair_num] << std::endl;


	// 	//=========== launch third kernel ==========
	// 	int warps = dv1.tile_size;
	// 	int threads = 256;
	// 	int blocks = (warps * 32 + threads - 1) / threads;

	// 	// 3. launch kernel
	// 	mark_candidate_status<<<blocks, threads>>>(
	// 		gd.d_min_min_dist,
	// 		gd.d_min_max_dist,
	// 		gd.d_prefix_tile,
	// 		gd.d_obj2,
	// 		gd.d_confirmed_before,
	// 		gd.d_confirmed_after,
	// 		K,
	// 		dv1.tile_size,
	// 		gd.d_status_before,
	// 		gd.d_status_after
	// 	);
	// 	check_execution();
	// 	CUDA_SAFE_CALL(cudaDeviceSynchronize());

	// 	confirmed_after.resize(dv1.tile_size);
	// 	CUDA_SAFE_CALL(cudaMemcpy(confirmed_after.data(), gd.d_confirmed_after,
	// 							  dv1.tile_size * sizeof(int), cudaMemcpyDeviceToHost));

	// 	status_after.resize(pair_num);
	// 	CUDA_SAFE_CALL(cudaMemcpy(status_after.data(), gd.d_status_after,
	// 							  pair_num * sizeof(int), cudaMemcpyDeviceToHost));

	// 	int confirm_this_round = 0;
	// 	for (int i = 0; i < dv1.tile_size; ++i)
	// 	{
	// 		if (confirmed_after[i] == K)
	// 		{
	// 			confirm_this_round++;
	// 		}
	// 	}

	// 	std::cout << "******confirm_this_round by GPU = " << confirm_this_round << std::endl;
	// }

	

	void compute_voxel_pair_distance_for_all_streaming_pipeline(
		const DeviceVoxels &dv1,
		const DeviceVoxels &dv2,
		vector<int> &prefix_tile,
		vector<size_t> &prefix_obj,
		vector<int> &compute_obj_1,
		vector<int> &compute_obj_2,
		size_t total_voxel_size,
		size_t max_voxel_size,
		vector<float> &h_min,
		vector<float> &h_max,
		vector<float> &min_max_dist,
		vector<float> &min_min_dist,
		vector<int> &valid_voxel_cnt,
		vector<size_t> &valid_voxel_prefix,
		vector<int>  &valid_voxel_pairs,      			
		vector<float> &valid_voxel_pairs_dist,
		int K,
		vector<int> &confirmed_after,
		vector<int> &status_after
	)
	{
		int pair_num = compute_obj_1.size();

		CUDA_SAFE_CALL(cudaMalloc(&gd.d_obj1, pair_num * sizeof(int)));
		CUDA_SAFE_CALL(cudaMalloc(&gd.d_obj2, pair_num * sizeof(int)));
		CUDA_SAFE_CALL(cudaMemcpy(gd.d_obj1, compute_obj_1.data(), pair_num * sizeof(int), cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(gd.d_obj2, compute_obj_2.data(), pair_num * sizeof(int), cudaMemcpyHostToDevice));

		// int *d_prefix_obj;
		CUDA_SAFE_CALL(cudaMalloc(&gd.d_prefix_obj, (pair_num + 1) * sizeof(size_t)));
		CUDA_SAFE_CALL(cudaMemcpy(gd.d_prefix_obj, prefix_obj.data(), (pair_num + 1) * sizeof(size_t), cudaMemcpyHostToDevice));

		// int *d_prefix_tile;
		CUDA_SAFE_CALL(cudaMalloc(&gd.d_prefix_tile, (dv1.tile_size + 1) * sizeof(int)));
		CUDA_SAFE_CALL(cudaMemcpy(gd.d_prefix_tile, prefix_tile.data(), (dv1.tile_size + 1) * sizeof(int), cudaMemcpyHostToDevice));

		// float *d_out_min, *d_out_max;
		// We don't allocate all at once 
		// CUDA_SAFE_CALL(cudaMalloc(&gd.d_out_min, total_voxel_size * sizeof(float)));
		// CUDA_SAFE_CALL(cudaMalloc(&gd.d_out_max, total_voxel_size * sizeof(float)));

		// float *d_min_max_dist, *d_min_min_dist;
		CUDA_SAFE_CALL(cudaMalloc(&gd.d_min_max_dist, pair_num * sizeof(float)));
		CUDA_SAFE_CALL(cudaMalloc(&gd.d_min_min_dist, pair_num * sizeof(float)));


		// allocate memory for the third kernel
		CUDA_SAFE_CALL(cudaMalloc(&gd.d_confirmed_before, dv1.tile_size * sizeof(int)));
		CUDA_SAFE_CALL(cudaMemset(gd.d_confirmed_before, 0, dv1.tile_size * sizeof(int)));

		CUDA_SAFE_CALL(cudaMalloc(&gd.d_confirmed_after, dv1.tile_size * sizeof(int)));
		CUDA_SAFE_CALL(cudaMemset(gd.d_confirmed_after, 0, dv1.tile_size * sizeof(int)));

		// int *d_status_before;
		CUDA_SAFE_CALL(cudaMalloc(&gd.d_status_before, pair_num * sizeof(int)));
		CUDA_SAFE_CALL(cudaMemset(gd.d_status_before, 0, pair_num * sizeof(int))); // all undecided

		// int *d_status_after;
		CUDA_SAFE_CALL(cudaMalloc(&gd.d_status_after, pair_num * sizeof(int)));
		CUDA_SAFE_CALL(cudaMemset(gd.d_status_after, 0, pair_num * sizeof(int)));


		size_t GPU_MEMORY_LIMIT = 1 * 1024ULL * 1024 * 1024; // 500MB
		size_t bytes_per_voxel = 2 * sizeof(float);           // min and max
		size_t MAX_CHUNK_VOXELS = GPU_MEMORY_LIMIT / bytes_per_voxel;

		const float RATIO = 1;

		size_t MAX_VALID = MAX_CHUNK_VOXELS;

		valid_voxel_cnt.assign(pair_num, 0);
		valid_voxel_prefix.assign(pair_num + 1, 0);

		min_max_dist.resize(pair_num);
		min_min_dist.resize(pair_num);

		valid_voxel_pairs.reserve(total_voxel_size * RATIO * 3);
		valid_voxel_pairs_dist.reserve(total_voxel_size * RATIO * 2);

		cudaHostRegister(valid_voxel_pairs.data(),
                 valid_voxel_pairs.capacity() * sizeof(int),
                 cudaHostRegisterDefault);

		cudaHostRegister(valid_voxel_pairs_dist.data(),
                 valid_voxel_pairs_dist.capacity() * sizeof(float),
                 cudaHostRegisterDefault);

		// valid_voxel_pairs.reserve(1e8 * 3);
		// valid_voxel_pairs_dist.reserve(1e8 * 2);


		size_t pair_begin = 0;
		
		cudaStream_t compute_stream, memcpy_stream;
		cudaStreamCreate(&compute_stream);
		cudaStreamCreate(&memcpy_stream);

		cudaEvent_t compute_done[2], copy_done[2];
		cudaEventCreate(&compute_done[0]);
		cudaEventCreate(&compute_done[1]);
		cudaEventCreate(&copy_done[0]);
		cudaEventCreate(&copy_done[1]);

		int* d_valid_pairs[2];
		float* d_valid_pairs_dist[2];

		cudaMalloc(&d_valid_pairs[0], 3 * MAX_VALID * sizeof(int));
		cudaMalloc(&d_valid_pairs[1], 3 * MAX_VALID * sizeof(int));
		cudaMalloc(&d_valid_pairs_dist[0], 2 * MAX_VALID * sizeof(float));
		cudaMalloc(&d_valid_pairs_dist[1], 2 * MAX_VALID * sizeof(float));


		auto tx = std::chrono::high_resolution_clock::now();

		// size_t STAGING_PAIRS_BYTES = 3 * MAX_VALID * sizeof(int);
		// size_t STAGING_DIST_BYTES  = 2 * MAX_VALID * sizeof(float);

		// int*   h_pairs_stage;
		// float* h_dist_stage;

		// CUDA_SAFE_CALL(cudaMallocHost(&h_pairs_stage, STAGING_PAIRS_BYTES));
		// CUDA_SAFE_CALL(cudaMallocHost(&h_dist_stage,  STAGING_DIST_BYTES));

		auto ty = std::chrono::high_resolution_clock::now();
		std::cout << "Pinned memory time = " << std::chrono::duration<double>(ty - tx).count() << std::endl;

		int *d_valid_voxel_cnt;
		size_t *d_valid_voxel_prefix;

		CUDA_SAFE_CALL(cudaMalloc(&d_valid_voxel_cnt, pair_num * sizeof(int)));
		CUDA_SAFE_CALL(cudaMalloc(&d_valid_voxel_prefix, (pair_num + 1) * sizeof(size_t)));
		CUDA_SAFE_CALL(cudaMemset(d_valid_voxel_cnt, 0, pair_num * sizeof(int)));
		CUDA_SAFE_CALL(cudaMemset(d_valid_voxel_prefix, 0, (pair_num + 1) * sizeof(size_t)));


		float *d_out_min[2], *d_out_max[2];
		size_t *d_count[2];
		for (int i=0; i<2; i++) {
			cudaMalloc(&d_out_min[i], MAX_CHUNK_VOXELS * sizeof(float));
			cudaMalloc(&d_out_max[i], MAX_CHUNK_VOXELS * sizeof(float));
			cudaMalloc(&d_count[i], pair_num * sizeof(size_t));
		}


		int iter = 0;
		size_t prev_chunk_total_valid = 0;
		size_t prev_old_size = 0, prev_old_size_2 = 0;


		std::vector<std::tuple<size_t, size_t, size_t>> chunks;

		while (pair_begin < pair_num)
		{
			size_t pair_end = pair_begin;
			size_t chunk_voxel_size = 0;

			while (pair_end < pair_num)
			{
				size_t voxel_cnt = prefix_obj[pair_end + 1] - prefix_obj[pair_end];

				if (chunk_voxel_size + voxel_cnt > MAX_CHUNK_VOXELS)
					break;

				chunk_voxel_size += voxel_cnt;
				pair_end++;
			}

			if (pair_end == pair_begin) {
				size_t voxel_cnt = prefix_obj[pair_end + 1] - prefix_obj[pair_end];
				chunk_voxel_size = voxel_cnt;
				pair_end++;
			}

			chunks.emplace_back(pair_begin, pair_end, chunk_voxel_size);

			pair_begin = pair_end;
		}

		
		// while (pair_begin < pair_num)
		for (size_t iter = 0; iter < chunks.size(); ++iter)
		{
			int curr = iter % 2;
			int prev = 1 - curr;

			// size_t pair_end = pair_begin;
			// size_t chunk_voxel_size = 0;

			// while (pair_end < pair_num)
			// {
			// 	size_t voxel_cnt = prefix_obj[pair_end + 1] - prefix_obj[pair_end];
			// 	if (chunk_voxel_size + voxel_cnt > MAX_CHUNK_VOXELS)
			// 		break;
			// 	chunk_voxel_size += voxel_cnt;
			// 	pair_end++;
			// }

			// size_t chunk_pair_num = pair_end - pair_begin;

			auto& chunk = chunks[iter];
			pair_begin = std::get<0>(chunk);
			size_t pair_end = std::get<1>(chunk);
			size_t chunk_voxel_size = std::get<2>(chunk);
    		size_t chunk_pair_num = pair_end - pair_begin;

			std::cout << "Chunk pairs: " << pair_begin << " - " << pair_end << std::endl;

			// float *d_out_min, *d_out_max;
			// size_t *d_count;
			// cudaMalloc(&d_out_min, chunk_voxel_size * sizeof(float));
			// cudaMalloc(&d_out_max, chunk_voxel_size * sizeof(float));
			// cudaMalloc(&d_count, chunk_pair_num * sizeof(size_t));
			


			voxel_pair_dist_kernel_for_all_streaming<<<chunk_pair_num, max_voxel_size, 0, compute_stream>>>(
				dv1, dv2,
				gd.d_obj1, gd.d_obj2,
				gd.d_prefix_obj,
				d_out_min[curr],
				d_out_max[curr],
				gd.d_min_max_dist,
				gd.d_min_min_dist,
				d_count[curr],
				pair_begin);

			// check_execution();
			// CUDA_SAFE_CALL(cudaDeviceSynchronize());

			if (iter > 0)
			{
				cudaStreamWaitEvent(memcpy_stream, compute_done[prev], 0);

				cudaMemcpyAsync( valid_voxel_pairs.data() + prev_old_size,
								d_valid_pairs[prev],
								3 * prev_chunk_total_valid * sizeof(int),
								cudaMemcpyDeviceToHost,
								memcpy_stream);

				cudaMemcpyAsync( valid_voxel_pairs_dist.data() + prev_old_size_2,
								d_valid_pairs_dist[prev],
								2 * prev_chunk_total_valid * sizeof(float),
								cudaMemcpyDeviceToHost,
								memcpy_stream);


				cudaEventRecord(copy_done[prev], memcpy_stream);

				// CUDA_SAFE_CALL(cudaStreamSynchronize(memcpy_stream));

				// valid_voxel_pairs.resize(valid_voxel_pairs.size() + 3 * prev_chunk_total_valid);
				// valid_voxel_pairs_dist.resize(valid_voxel_pairs_dist.size() + 2 * prev_chunk_total_valid);

				// std::memcpy(valid_voxel_pairs.data() + prev_old_size,
				// 			h_pairs_stage,
				// 			3 * prev_chunk_total_valid * sizeof(int));

				// std::memcpy(valid_voxel_pairs_dist.data() + prev_old_size_2,
				// 			h_dist_stage,
				// 			2 * prev_chunk_total_valid * sizeof(float));
			}

			cudaStreamSynchronize(compute_stream); 

			// std::vector<size_t> chunk_count(chunk_pair_num);
			// cudaMemcpy(chunk_count.data(), d_count,
			// 		chunk_pair_num * sizeof(size_t),
			// 		cudaMemcpyDeviceToHost);

			// std::vector<size_t> chunk_valid_prefix(chunk_pair_num + 1, 0);
			// for (size_t i = 0; i < chunk_pair_num; ++i)
			// 	chunk_valid_prefix[i + 1] = chunk_valid_prefix[i] + chunk_count[i];

			// size_t chunk_total_valid = chunk_valid_prefix[chunk_pair_num];

			// size_t global_valid_base = valid_voxel_prefix[pair_begin];
			// for (size_t i = 0; i < chunk_pair_num; ++i)
			// {
			// 	valid_voxel_cnt[pair_begin + i] = chunk_count[i];
			// 	valid_voxel_prefix[pair_begin + i + 1] = global_valid_base + chunk_valid_prefix[i + 1];
			// }

			// size_t* d_chunk_prefix;
			// cudaMalloc(&d_chunk_prefix, (chunk_pair_num + 1) * sizeof(size_t));
			// cudaMemcpy(d_chunk_prefix,
			// 		chunk_valid_prefix.data(),
			// 		(chunk_pair_num + 1) * sizeof(size_t),
			// 		cudaMemcpyHostToDevice);

			size_t* d_chunk_prefix;
			CUDA_SAFE_CALL(cudaMalloc(&d_chunk_prefix, (chunk_pair_num + 1) * sizeof(size_t)));
			inclusive_scan(d_count[curr], d_chunk_prefix, chunk_pair_num);

			size_t chunk_total_valid;
			cudaMemcpy(&chunk_total_valid, d_chunk_prefix + chunk_pair_num, sizeof(size_t), cudaMemcpyDeviceToHost);
			
			// Let GPU do this
			fill_count_and_prefix_kernel<<<(chunk_pair_num + 255)/256, 256, 0, compute_stream>>>(d_valid_voxel_cnt,
																								d_valid_voxel_prefix,
																								d_count[curr],
																								d_chunk_prefix,
																								pair_begin,
																								chunk_pair_num
																							);

			cudaStreamWaitEvent(compute_stream, copy_done[curr], 0);

			record_valid_voxel_pairs_streaming<<<chunk_pair_num, max_voxel_size, 0, compute_stream>>>(
				dv1, dv2,
				gd.d_obj1, gd.d_obj2,
				gd.d_prefix_obj,
				d_out_min[curr],
				d_out_max[curr],
				gd.d_min_max_dist,
				d_chunk_prefix,
				d_valid_pairs[curr],
				d_valid_pairs_dist[curr],
				pair_begin);

			fill_valid_voxel_distance_streaming<<<chunk_pair_num, max_voxel_size, 0, compute_stream>>>(
				gd.d_prefix_obj,
				d_chunk_prefix,
				d_valid_pairs[curr],
				d_out_min[curr],
				d_out_max[curr],
				d_valid_pairs_dist[curr],
				pair_begin);

			cudaEventRecord(compute_done[curr], compute_stream);

			// 更新 offset
			prev_old_size += 3 * prev_chunk_total_valid;
			prev_old_size_2 += 2 * prev_chunk_total_valid;
			prev_chunk_total_valid = chunk_total_valid;

			std::cout << "prev_old_size = " << prev_old_size << std::endl;

			// valid_voxel_pairs.resize(prev_old_size + 3 * chunk_total_valid);
			// valid_voxel_pairs_dist.resize(prev_old_size_2 + 2 * chunk_total_valid);

			// pair_begin = pair_end;
			// iter++;

			// cudaFree(d_out_min);
			// cudaFree(d_out_max);
			// cudaFree(d_count);
			// cudaFree(d_chunk_prefix);
		}

		if (iter > 0) {
			int last = (iter - 1) % 2;
		
			// wait for compute 
			cudaStreamWaitEvent(memcpy_stream,
								compute_done[last],
								0);
			// memcpy last round
			cudaMemcpyAsync( valid_voxel_pairs.data() + prev_old_size,
							d_valid_pairs[last],
							3 * prev_chunk_total_valid * sizeof(int),
							cudaMemcpyDeviceToHost,
							memcpy_stream);

			cudaMemcpyAsync( valid_voxel_pairs_dist.data() + prev_old_size_2,
							d_valid_pairs_dist[last],
							2 * prev_chunk_total_valid * sizeof(float),
							cudaMemcpyDeviceToHost,
							memcpy_stream);

			// wait memcpy 
			cudaStreamSynchronize(memcpy_stream);

			// valid_voxel_pairs.resize(valid_voxel_pairs.size() + 3 * prev_chunk_total_valid);
			// valid_voxel_pairs_dist.resize(valid_voxel_pairs_dist.size() + 2 * prev_chunk_total_valid);

			// std::memcpy(valid_voxel_pairs.data() + prev_old_size,
			// 				h_pairs_stage,
			// 				3 * prev_chunk_total_valid * sizeof(int));

			// std::memcpy(valid_voxel_pairs_dist.data() + prev_old_size_2,
			// 				h_dist_stage,
			// 				2 * prev_chunk_total_valid * sizeof(float));
		}


		// CUDA_SAFE_CALL(cudaFreeHost(h_pairs_stage));
		// CUDA_SAFE_CALL(cudaFreeHost(h_dist_stage));
		
		//=========== launch third kernel ==========
		int BLOCK = 256;
		int GRID  = dv1.tile_size;   // one tile per block

		evaluate_knn_kernel<<<GRID, BLOCK>>>(
			gd.d_min_min_dist,
			gd.d_min_max_dist,
			gd.d_prefix_tile,
			gd.d_confirmed_before,
			gd.d_status_after,
			dv1.tile_size,
			K
		);
		check_execution();
		CUDA_SAFE_CALL(cudaDeviceSynchronize());

		// status 是 per pair
		status_after.resize(pair_num);
		CUDA_SAFE_CALL(cudaMemcpy(
			status_after.data(),
			gd.d_status_after,
			pair_num * sizeof(int),
			cudaMemcpyDeviceToHost));

		confirmed_after.resize(dv1.tile_size);
		for (int t = 0; t < dv1.tile_size; ++t)
		{
			int begin = prefix_tile[t];
			int end   = prefix_tile[t + 1];

			int confirmed_cnt = 0;

			for (int i = begin; i < end; ++i)
			{
				if (status_after[i] == 1)  // CONFIRMED
					confirmed_cnt++;
				
				if (confirmed_cnt == K) break;
			}
			confirmed_after[t] = confirmed_cnt;
		}

		CUDA_SAFE_CALL(cudaMemcpy(valid_voxel_cnt.data(), d_valid_voxel_cnt, pair_num * sizeof(int), cudaMemcpyDeviceToHost));
		CUDA_SAFE_CALL(cudaMemcpy(valid_voxel_prefix.data(), d_valid_voxel_prefix, (pair_num + 1) * sizeof(size_t), cudaMemcpyDeviceToHost));

		CUDA_SAFE_CALL(cudaHostUnregister(valid_voxel_pairs.data()));
		CUDA_SAFE_CALL(cudaHostUnregister(valid_voxel_pairs_dist.data()));

		for (int i=0; i<2; i++) {
			cudaFree(&d_out_min[i]);
			cudaFree(&d_out_max[i]);
			cudaFree(&d_count[i]);
		}
	}

	void compute_voxel_pair_distance_for_all_streaming_pipeline_reduceIO(
		const DeviceVoxels &dv1,
		const DeviceVoxels &dv2,
		vector<int> &prefix_tile,
		vector<size_t> &prefix_obj,
		vector<int> &compute_obj_1,
		vector<int> &compute_obj_2,
		size_t total_voxel_size,
		size_t max_voxel_size,
		vector<float> &h_min,
		vector<float> &h_max,
		vector<float> &min_max_dist,
		vector<float> &min_min_dist,
		vector<int> &valid_voxel_cnt,
		vector<size_t> &valid_voxel_prefix,
		vector<int>  &valid_voxel_pairs,      			
		vector<float> &valid_voxel_pairs_dist,
		int K,
		vector<int> &confirmed_after,
		vector<int> &status_after
	) 
	{
		int pair_num = compute_obj_1.size();

		std::cout << "pair_num = " <<  pair_num << std::endl;

		CUDA_SAFE_CALL(cudaMalloc(&gd.d_obj1, pair_num * sizeof(int)));
		CUDA_SAFE_CALL(cudaMalloc(&gd.d_obj2, pair_num * sizeof(int)));
		CUDA_SAFE_CALL(cudaMemcpy(gd.d_obj1, compute_obj_1.data(), pair_num * sizeof(int), cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(gd.d_obj2, compute_obj_2.data(), pair_num * sizeof(int), cudaMemcpyHostToDevice));

		// int *d_prefix_obj;
		CUDA_SAFE_CALL(cudaMalloc(&gd.d_prefix_obj, (pair_num + 1) * sizeof(size_t)));
		CUDA_SAFE_CALL(cudaMemcpy(gd.d_prefix_obj, prefix_obj.data(), (pair_num + 1) * sizeof(size_t), cudaMemcpyHostToDevice));

		// int *d_prefix_tile;
		CUDA_SAFE_CALL(cudaMalloc(&gd.d_prefix_tile, (dv1.tile_size + 1) * sizeof(int)));
		CUDA_SAFE_CALL(cudaMemcpy(gd.d_prefix_tile, prefix_tile.data(), (dv1.tile_size + 1) * sizeof(int), cudaMemcpyHostToDevice));

		// float *d_out_min, *d_out_max;
		// We don't allocate all at once 
		// CUDA_SAFE_CALL(cudaMalloc(&gd.d_out_min, total_voxel_size * sizeof(float)));
		// CUDA_SAFE_CALL(cudaMalloc(&gd.d_out_max, total_voxel_size * sizeof(float)));

		// float *d_min_max_dist, *d_min_min_dist;
		CUDA_SAFE_CALL(cudaMalloc(&gd.d_min_max_dist, pair_num * sizeof(float)));
		CUDA_SAFE_CALL(cudaMalloc(&gd.d_min_min_dist, pair_num * sizeof(float)));


		// allocate memory for the third kernel
		CUDA_SAFE_CALL(cudaMalloc(&gd.d_confirmed_before, dv1.tile_size * sizeof(int)));
		CUDA_SAFE_CALL(cudaMemset(gd.d_confirmed_before, 0, dv1.tile_size * sizeof(int)));

		CUDA_SAFE_CALL(cudaMalloc(&gd.d_confirmed_after, dv1.tile_size * sizeof(int)));
		CUDA_SAFE_CALL(cudaMemset(gd.d_confirmed_after, 0, dv1.tile_size * sizeof(int)));

		// int *d_status_before;
		CUDA_SAFE_CALL(cudaMalloc(&gd.d_status_before, pair_num * sizeof(int)));
		CUDA_SAFE_CALL(cudaMemset(gd.d_status_before, 0, pair_num * sizeof(int))); // all undecided

		// int *d_status_after;
		CUDA_SAFE_CALL(cudaMalloc(&gd.d_status_after, pair_num * sizeof(int)));
		CUDA_SAFE_CALL(cudaMemset(gd.d_status_after, 0, pair_num * sizeof(int)));


		size_t GPU_MEMORY_LIMIT = 5 * 1024ULL * 1024 * 1024; // 500MB
		size_t bytes_per_voxel = 2 * sizeof(float);           // min and max
		size_t MAX_CHUNK_VOXELS = GPU_MEMORY_LIMIT / bytes_per_voxel;

		const float RATIO = 0.07;

		size_t MAX_VALID = MAX_CHUNK_VOXELS;

		valid_voxel_cnt.assign(pair_num, 0);
		valid_voxel_prefix.assign(pair_num + 1, 0);

		min_max_dist.resize(pair_num);
		min_min_dist.resize(pair_num);


		valid_voxel_pairs.reserve(10 * 1e8 * 3);
		valid_voxel_pairs_dist.reserve(10 * 1e8 * 2);

		auto tx = std::chrono::high_resolution_clock::now();

		CUDA_SAFE_CALL(cudaHostRegister(valid_voxel_pairs.data(),
                                valid_voxel_pairs.capacity() * sizeof(int),
                                cudaHostRegisterDefault));

		CUDA_SAFE_CALL(cudaHostRegister(valid_voxel_pairs_dist.data(),
                                valid_voxel_pairs_dist.capacity() * sizeof(float),
                                cudaHostRegisterDefault));
		
		auto ty = std::chrono::high_resolution_clock::now();
		std::cout << "Pinned memory time = " << std::chrono::duration<double>(ty - tx).count() << std::endl;


		int *d_valid_voxel_cnt;
		size_t *d_valid_voxel_prefix;

		CUDA_SAFE_CALL(cudaMalloc(&d_valid_voxel_cnt, pair_num * sizeof(int)));
		CUDA_SAFE_CALL(cudaMalloc(&d_valid_voxel_prefix, (pair_num + 1) * sizeof(size_t)));
		CUDA_SAFE_CALL(cudaMemset(d_valid_voxel_cnt, 0, pair_num * sizeof(int)));
		CUDA_SAFE_CALL(cudaMemset(d_valid_voxel_prefix, 0, (pair_num + 1) * sizeof(size_t)));


		size_t pair_begin = 0;
		
		cudaStream_t compute_stream, memcpy_stream;
		cudaStreamCreate(&compute_stream);
		cudaStreamCreate(&memcpy_stream);

		cudaEvent_t compute_done[2], copy_done[2];
		cudaEventCreate(&compute_done[0]);
		cudaEventCreate(&compute_done[1]);
		cudaEventCreate(&copy_done[0]);
		cudaEventCreate(&copy_done[1]);

		int* d_valid_pairs[2];
		float* d_valid_pairs_dist[2];

		cudaMalloc(&d_valid_pairs[0], 3 * MAX_VALID * sizeof(int));
		cudaMalloc(&d_valid_pairs[1], 3 * MAX_VALID * sizeof(int));
		cudaMalloc(&d_valid_pairs_dist[0], 2 * MAX_VALID * sizeof(float));
		cudaMalloc(&d_valid_pairs_dist[1], 2 * MAX_VALID * sizeof(float));

		size_t BLKDIM = min(max_voxel_size, (size_t)1024);

		// size_t STAGING_PAIRS_BYTES = 3 * MAX_VALID * sizeof(int);
		// size_t STAGING_DIST_BYTES  = 2 * MAX_VALID * sizeof(float);

		// auto tx = std::chrono::high_resolution_clock::now();

		// int*   h_pairs_stage;
		// float* h_dist_stage;

		// CUDA_SAFE_CALL(cudaMallocHost(&h_pairs_stage, STAGING_PAIRS_BYTES));
		// CUDA_SAFE_CALL(cudaMallocHost(&h_dist_stage,  STAGING_DIST_BYTES));

		// auto ty = std::chrono::high_resolution_clock::now();
		// std::cout << "Pinned memory time = " << std::chrono::duration<double>(ty - tx).count() << std::endl;


		voxel_pair_dist_kernel_for_all_aggregate_distance<<<pair_num, BLKDIM>>>(dv1, dv2,
																						gd.d_obj1,
																						gd.d_obj2,
																						gd.d_min_max_dist,
																						gd.d_min_min_dist);
		check_execution();
		CUDA_SAFE_CALL(cudaDeviceSynchronize());

		// launch kNN selection
		int BLOCK = 256;
		int GRID  = dv1.tile_size;   // one tile per block

		evaluate_knn_kernel<<<GRID, BLOCK>>>(
			gd.d_min_min_dist,
			gd.d_min_max_dist,
			gd.d_prefix_tile,
			gd.d_confirmed_before,
			gd.d_status_after,
			dv1.tile_size,
			K
		);
		check_execution();
		CUDA_SAFE_CALL(cudaDeviceSynchronize());

		// status 是 per pair
		status_after.resize(pair_num);
		CUDA_SAFE_CALL(cudaMemcpy(
			status_after.data(),
			gd.d_status_after,
			pair_num * sizeof(int),
			cudaMemcpyDeviceToHost));

		confirmed_after.resize(dv1.tile_size);
		for (int t = 0; t < dv1.tile_size; ++t)
		{
			int begin = prefix_tile[t];
			int end   = prefix_tile[t + 1];

			int confirmed_cnt = 0;

			for (int i = begin; i < end; ++i)
			{
				if (status_after[i] == 1)  // CONFIRMED
					confirmed_cnt++;

				if (confirmed_cnt == K) break;
			}
			confirmed_after[t] = confirmed_cnt;
		}

		size_t finished = 0;
		for (int i=0;i<dv1.tile_size; ++i){
			finished += confirmed_after[i];
		}

		int *d_confirmed_after;
		CUDA_SAFE_CALL(cudaMalloc(&d_confirmed_after, dv1.tile_size * sizeof(int)));
		CUDA_SAFE_CALL(cudaMemcpy(d_confirmed_after, confirmed_after.data(), dv1.tile_size * sizeof(int), cudaMemcpyHostToDevice));

		int iter = 0;
		size_t prev_chunk_total_valid = 0;
		size_t prev_old_size = 0, prev_old_size_2 = 0;

		float *d_out_min, *d_out_max;
		size_t *d_count;
		cudaMalloc(&d_out_min, MAX_CHUNK_VOXELS * sizeof(float));
		cudaMalloc(&d_out_max, MAX_CHUNK_VOXELS * sizeof(float));
		cudaMalloc(&d_count, pair_num * sizeof(size_t));



		std::vector<std::tuple<size_t, size_t, size_t>> chunks;

		while (pair_begin < pair_num)
		{
			size_t pair_end = pair_begin;
			size_t chunk_voxel_size = 0;

			while (pair_end < pair_num)
			{
				size_t voxel_cnt = prefix_obj[pair_end + 1] - prefix_obj[pair_end];

				if (chunk_voxel_size + voxel_cnt > MAX_CHUNK_VOXELS)
					break;

				chunk_voxel_size += voxel_cnt;
				pair_end++;
			}

			if (pair_end == pair_begin) {
				size_t voxel_cnt = prefix_obj[pair_end + 1] - prefix_obj[pair_end];
				chunk_voxel_size = voxel_cnt;
				pair_end++;
			}

			chunks.emplace_back(pair_begin, pair_end, chunk_voxel_size);

			pair_begin = pair_end;
		}


		
		// while (pair_begin < pair_num)
		for (size_t iter = 0; iter < chunks.size(); ++iter)
		{
			int curr = iter % 2;
			int prev = 1 - curr;

			// size_t pair_end = pair_begin;
			// size_t chunk_voxel_size = 0;

			// while (pair_end < pair_num)
			// {
			// 	size_t voxel_cnt = prefix_obj[pair_end + 1] - prefix_obj[pair_end];
			// 	if (chunk_voxel_size + voxel_cnt > MAX_CHUNK_VOXELS)
			// 		break;
			// 	chunk_voxel_size += voxel_cnt;
			// 	pair_end++;
			// }

			// size_t chunk_pair_num = pair_end - pair_begin;

			auto& chunk = chunks[iter];
			pair_begin = std::get<0>(chunk);
			size_t pair_end = std::get<1>(chunk);
			size_t chunk_voxel_size = std::get<2>(chunk);
    		size_t chunk_pair_num = pair_end - pair_begin;

			std::cout << "Chunk pairs: " << pair_begin << " - " << pair_end << std::endl;

			// float *d_out_min, *d_out_max;
			// size_t *d_count;
			// CUDA_SAFE_CALL(cudaMalloc(&d_out_min, chunk_voxel_size * sizeof(float)));
			// CUDA_SAFE_CALL(cudaMalloc(&d_out_max, chunk_voxel_size * sizeof(float)));
			// CUDA_SAFE_CALL(cudaMalloc(&d_count, chunk_pair_num * sizeof(size_t)));
			CUDA_SAFE_CALL(cudaMemset(d_count, 0, chunk_pair_num * sizeof(size_t)));


			voxel_pair_dist_kernel_for_all_streaming_knn<<<chunk_pair_num, BLKDIM, 0, compute_stream>>>(
				dv1, dv2,
				gd.d_obj1, gd.d_obj2,
				gd.d_prefix_obj,
				d_out_min,
				d_out_max,
				gd.d_min_max_dist,
				gd.d_min_min_dist,
				d_count,
				pair_begin,
				d_confirmed_after,
				gd.d_status_after,
				K
			);

			if (iter > 0)
			{
				cudaStreamWaitEvent(memcpy_stream, compute_done[prev], 0);

				cudaMemcpyAsync( valid_voxel_pairs.data() + prev_old_size,
								d_valid_pairs[prev],
								3 * prev_chunk_total_valid * sizeof(int),
								cudaMemcpyDeviceToHost,
								memcpy_stream);

				cudaMemcpyAsync( valid_voxel_pairs_dist.data() + prev_old_size_2,
								d_valid_pairs_dist[prev],
								2 * prev_chunk_total_valid * sizeof(float),
								cudaMemcpyDeviceToHost,
								memcpy_stream);

				cudaEventRecord(copy_done[prev], memcpy_stream);



				// CUDA_SAFE_CALL(cudaStreamSynchronize(memcpy_stream));


				// valid_voxel_pairs.resize(valid_voxel_pairs.size() + 3 * prev_chunk_total_valid);
				// valid_voxel_pairs_dist.resize(valid_voxel_pairs_dist.size() + 2 * prev_chunk_total_valid);

				
				// std::memcpy(valid_voxel_pairs.data() + prev_old_size,
				// 		h_pairs_stage,
				// 		3 * prev_chunk_total_valid * sizeof(int));

				// std::memcpy(valid_voxel_pairs_dist.data() + prev_old_size_2,
				// 		h_dist_stage,
				// 		2 * prev_chunk_total_valid * sizeof(float));

			}


			cudaStreamSynchronize(compute_stream); 

			size_t* d_chunk_prefix;
			CUDA_SAFE_CALL(cudaMalloc(&d_chunk_prefix, (chunk_pair_num + 1) * sizeof(size_t)));
			inclusive_scan(d_count, d_chunk_prefix, chunk_pair_num);

			size_t chunk_total_valid;
			cudaMemcpy(&chunk_total_valid, d_chunk_prefix + chunk_pair_num, sizeof(size_t), cudaMemcpyDeviceToHost);
			
			// Let GPU do this
			fill_count_and_prefix_kernel<<<(chunk_pair_num + 255)/256, 256, 0, compute_stream>>>(d_valid_voxel_cnt,
																								d_valid_voxel_prefix,
																								d_count,
																								d_chunk_prefix,
																								pair_begin,
																								chunk_pair_num
																							);
	
			cudaStreamWaitEvent(compute_stream, copy_done[curr], 0);

			record_valid_voxel_pairs_streaming_knn<<<chunk_pair_num, BLKDIM, 0, compute_stream>>>(
				dv1, dv2,
				gd.d_obj1, gd.d_obj2,
				gd.d_prefix_obj,
				d_out_min,
				d_out_max,
				gd.d_min_max_dist,
				d_chunk_prefix,
				d_valid_pairs[curr],
				d_valid_pairs_dist[curr],
				pair_begin,
				d_confirmed_after,
				gd.d_status_after,
				K
			);

			fill_valid_voxel_distance_streaming<<<chunk_pair_num, BLKDIM, 0, compute_stream>>>(
				gd.d_prefix_obj,
				d_chunk_prefix,
				d_valid_pairs[curr],
				d_out_min,
				d_out_max,
				d_valid_pairs_dist[curr],
				pair_begin);

			cudaEventRecord(compute_done[curr], compute_stream);

			// 更新 offset
			prev_old_size += 3 * prev_chunk_total_valid;
			prev_old_size_2 += 2 * prev_chunk_total_valid;
			prev_chunk_total_valid = chunk_total_valid;

			// std::cout << "prev_old_size = " <<prev_old_size << std::endl;

			// pair_begin = pair_end;
			// iter++;

			// cudaFree(d_out_min);
			// cudaFree(d_out_max);
			// cudaFree(d_count);
			// cudaFree(d_chunk_prefix);
		}

		if (iter > 0) {
			int last = (iter - 1) % 2;
		
			// wait for compute 
			cudaStreamWaitEvent(memcpy_stream,
								compute_done[last],
								0);
			// memcpy last round
			cudaMemcpyAsync( valid_voxel_pairs.data() + prev_old_size,
							d_valid_pairs[last],
							3 * prev_chunk_total_valid * sizeof(int),
							cudaMemcpyDeviceToHost,
							memcpy_stream);

			cudaMemcpyAsync( valid_voxel_pairs_dist.data() + prev_old_size_2,
							d_valid_pairs_dist[last],
							2 * prev_chunk_total_valid * sizeof(float),
							cudaMemcpyDeviceToHost,
							memcpy_stream);

			// wait memcpy 
			cudaStreamSynchronize(memcpy_stream);

			// valid_voxel_pairs.resize(valid_voxel_pairs.size() + 3 * prev_chunk_total_valid);
			// valid_voxel_pairs_dist.resize(valid_voxel_pairs_dist.size() + 2 * prev_chunk_total_valid);

			// std::memcpy(valid_voxel_pairs.data() + prev_old_size,
			// 				h_pairs_stage,
			// 				3 * prev_chunk_total_valid * sizeof(int));

			// std::memcpy(valid_voxel_pairs_dist.data() + prev_old_size_2,
			// 				h_dist_stage,
			// 				2 * prev_chunk_total_valid * sizeof(float));
		}


		// CUDA_SAFE_CALL(cudaFreeHost(h_pairs_stage));
		// CUDA_SAFE_CALL(cudaFreeHost(h_dist_stage));
		
		CUDA_SAFE_CALL(cudaMemcpy(valid_voxel_cnt.data(), d_valid_voxel_cnt, pair_num * sizeof(int), cudaMemcpyDeviceToHost));
		CUDA_SAFE_CALL(cudaMemcpy(valid_voxel_prefix.data(), d_valid_voxel_prefix, (pair_num + 1) * sizeof(size_t), cudaMemcpyDeviceToHost));

		CUDA_SAFE_CALL(cudaHostUnregister(valid_voxel_pairs.data()));
		CUDA_SAFE_CALL(cudaHostUnregister(valid_voxel_pairs_dist.data()));

		cudaFree(d_out_min);
		cudaFree(d_out_max);
		cudaFree(d_count);
	}



	void compute_voxel_pair_distance_for_all_streaming_pipeline_disk(
		const DeviceVoxels &dv1,
		const DeviceVoxels &dv2,
		vector<int> &prefix_tile,
		vector<size_t> &prefix_obj,
		vector<int> &compute_obj_1,
		vector<int> &compute_obj_2,
		size_t total_voxel_size,
		size_t max_voxel_size,
		vector<float> &h_min,
		vector<float> &h_max,
		vector<float> &min_max_dist,
		vector<float> &min_min_dist,
		vector<int> &valid_voxel_cnt,
		vector<size_t> &valid_voxel_prefix,
		vector<int>  &valid_voxel_pairs, 
		vector<float> &valid_voxel_pairs_dist,
		int K,
		vector<int> &confirmed_after,
		vector<int> &status_after
	) 
	{
		int pair_num = compute_obj_1.size();

		CUDA_SAFE_CALL(cudaMalloc(&gd.d_obj1, pair_num * sizeof(int)));
		CUDA_SAFE_CALL(cudaMalloc(&gd.d_obj2, pair_num * sizeof(int)));
		CUDA_SAFE_CALL(cudaMemcpy(gd.d_obj1, compute_obj_1.data(), pair_num * sizeof(int), cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(gd.d_obj2, compute_obj_2.data(), pair_num * sizeof(int), cudaMemcpyHostToDevice));

		// int *d_prefix_obj;
		CUDA_SAFE_CALL(cudaMalloc(&gd.d_prefix_obj, (pair_num + 1) * sizeof(size_t)));
		CUDA_SAFE_CALL(cudaMemcpy(gd.d_prefix_obj, prefix_obj.data(), (pair_num + 1) * sizeof(size_t), cudaMemcpyHostToDevice));

		// int *d_prefix_tile;
		CUDA_SAFE_CALL(cudaMalloc(&gd.d_prefix_tile, (dv1.tile_size + 1) * sizeof(int)));
		CUDA_SAFE_CALL(cudaMemcpy(gd.d_prefix_tile, prefix_tile.data(), (dv1.tile_size + 1) * sizeof(int), cudaMemcpyHostToDevice));

		// float *d_out_min, *d_out_max;
		// We don't allocate all at once 
		// CUDA_SAFE_CALL(cudaMalloc(&gd.d_out_min, total_voxel_size * sizeof(float)));
		// CUDA_SAFE_CALL(cudaMalloc(&gd.d_out_max, total_voxel_size * sizeof(float)));

		// float *d_min_max_dist, *d_min_min_dist;
		CUDA_SAFE_CALL(cudaMalloc(&gd.d_min_max_dist, pair_num * sizeof(float)));
		CUDA_SAFE_CALL(cudaMalloc(&gd.d_min_min_dist, pair_num * sizeof(float)));

		// int *d_count;
		// CUDA_SAFE_CALL(cudaMalloc(&gd.d_count, pair_num * sizeof(int)));

		// allocate memory for the third kernel
		CUDA_SAFE_CALL(cudaMalloc(&gd.d_confirmed_before, dv1.tile_size * sizeof(int)));
		CUDA_SAFE_CALL(cudaMemset(gd.d_confirmed_before, 0, dv1.tile_size * sizeof(int)));

		CUDA_SAFE_CALL(cudaMalloc(&gd.d_confirmed_after, dv1.tile_size * sizeof(int)));
		CUDA_SAFE_CALL(cudaMemset(gd.d_confirmed_after, 0, dv1.tile_size * sizeof(int)));

		// int *d_status_before;
		CUDA_SAFE_CALL(cudaMalloc(&gd.d_status_before, pair_num * sizeof(int)));
		CUDA_SAFE_CALL(cudaMemset(gd.d_status_before, 0, pair_num * sizeof(int))); // all undecided

		// int *d_status_after;
		CUDA_SAFE_CALL(cudaMalloc(&gd.d_status_after, pair_num * sizeof(int)));
		CUDA_SAFE_CALL(cudaMemset(gd.d_status_after, 0, pair_num * sizeof(int)));


		size_t GPU_MEMORY_LIMIT = 1 * 1024ULL * 1024 * 1024; // 500MB
		size_t bytes_per_voxel = 2 * sizeof(float);           // min and max
		size_t MAX_CHUNK_VOXELS = GPU_MEMORY_LIMIT / bytes_per_voxel;

		const float RATIO = 0.07;

		size_t MAX_VALID = MAX_CHUNK_VOXELS;

		valid_voxel_cnt.assign(pair_num, 0);
		valid_voxel_prefix.assign(pair_num + 1, 0);

		min_max_dist.resize(pair_num);
		min_min_dist.resize(pair_num);


		auto tx = std::chrono::high_resolution_clock::now();


		Buffer buffers[2];
		int cur = 0;
		int prev = 1;  
		std::mutex mtx;
		std::condition_variable cv;
		bool write_done = true;
		bool exit_flag = false;
		int write_index = -1;

		std::thread writer([&]() {

			FILE* f_pairs = fopen("pairs.bin", "wb");
			FILE* f_dist  = fopen("dist.bin", "wb");

			while (true) {
				int idx = -1;

				{
					std::unique_lock<std::mutex> lock(mtx);
					cv.wait(lock, [&]() {
						return write_index != -1 || exit_flag;
					});

					if (exit_flag && write_index == -1) break;

					idx = write_index;
					write_index = -1;
					write_done = false;
				}

				Buffer& buf = buffers[idx];

				fwrite(buf.pairs, sizeof(int),   3 * buf.pair_count, f_pairs);
				fwrite(buf.dist,  sizeof(float), 2 * buf.pair_count, f_dist);

				{
					std::lock_guard<std::mutex> lock(mtx);
					write_done = true;
				}

				cv.notify_all();
			}

			fclose(f_pairs);
			fclose(f_dist);
		});



		// TODO: save/flush on disk???
		// valid_voxel_pairs.reserve(total_voxel_size * RATIO * 3);
		// valid_voxel_pairs_dist.reserve(total_voxel_size * RATIO * 2);

		auto ty = std::chrono::high_resolution_clock::now();
		std::cout << "Host vector time = " << std::chrono::duration<double>(ty - tx).count() << std::endl;


		size_t pair_begin = 0;
		
		cudaStream_t compute_stream, memcpy_stream;
		cudaStreamCreate(&compute_stream);
		cudaStreamCreate(&memcpy_stream);

		cudaEvent_t compute_done[2], copy_done[2];
		cudaEventCreate(&compute_done[0]);
		cudaEventCreate(&compute_done[1]);
		cudaEventCreate(&copy_done[0]);
		cudaEventCreate(&copy_done[1]);

		int* d_valid_pairs[2];
		float* d_valid_pairs_dist[2];

		cudaMalloc(&d_valid_pairs[0], 3 * MAX_VALID * sizeof(int));
		cudaMalloc(&d_valid_pairs[1], 3 * MAX_VALID * sizeof(int));
		cudaMalloc(&d_valid_pairs_dist[0], 2 * MAX_VALID * sizeof(float));
		cudaMalloc(&d_valid_pairs_dist[1], 2 * MAX_VALID * sizeof(float));


		tx = std::chrono::high_resolution_clock::now();

		size_t STAGING_PAIRS_BYTES = 3 * MAX_VALID * sizeof(int);
		size_t STAGING_DIST_BYTES  = 2 * MAX_VALID * sizeof(float);

		int*   h_pairs_stage;
		float* h_dist_stage;

		CUDA_SAFE_CALL(cudaMallocHost(&h_pairs_stage, STAGING_PAIRS_BYTES));
		CUDA_SAFE_CALL(cudaMallocHost(&h_dist_stage,  STAGING_DIST_BYTES));

		ty = std::chrono::high_resolution_clock::now();
		std::cout << "Pin time = " << std::chrono::duration<double>(ty - tx).count() << std::endl;


		int iter = 0;
		size_t prev_chunk_total_valid = 0;
		size_t prev_old_size = 0, prev_old_size_2 = 0;

		auto t0 = std::chrono::high_resolution_clock::now();

		while (pair_begin < pair_num)
		{
			int curr = iter % 2;
			int prev = 1 - curr;

			size_t pair_end = pair_begin;
			size_t chunk_voxel_size = 0;

			while (pair_end < pair_num)
			{
				size_t voxel_cnt = prefix_obj[pair_end + 1] - prefix_obj[pair_end];
				if (chunk_voxel_size + voxel_cnt > MAX_CHUNK_VOXELS)
					break;
				chunk_voxel_size += voxel_cnt;
				pair_end++;
			}

			size_t chunk_pair_num = pair_end - pair_begin;

			std::cout << "Chunk pairs: " << pair_begin << " - " << pair_end << std::endl;

			float *d_out_min, *d_out_max;
			size_t *d_count;

			cudaMalloc(&d_out_min, chunk_voxel_size * sizeof(float));
			cudaMalloc(&d_out_max, chunk_voxel_size * sizeof(float));
			cudaMalloc(&d_count, chunk_pair_num * sizeof(size_t));


			voxel_pair_dist_kernel_for_all_streaming<<<chunk_pair_num, max_voxel_size, 0, compute_stream>>>(
				dv1, dv2,
				gd.d_obj1, gd.d_obj2,
				gd.d_prefix_obj,
				d_out_min,
				d_out_max,
				gd.d_min_max_dist,
				gd.d_min_min_dist,
				d_count,
				pair_begin);

			if (iter > 0)
			{
				cudaStreamWaitEvent(memcpy_stream, compute_done[prev], 0);

				cudaMemcpyAsync(h_pairs_stage,											// valid_voxel_pairs.data() + prev_old_size,
								d_valid_pairs[prev],
								3 * prev_chunk_total_valid * sizeof(int),
								cudaMemcpyDeviceToHost,
								memcpy_stream);

				cudaMemcpyAsync(h_dist_stage,											// valid_voxel_pairs_dist.data() + prev_old_size_2,
								d_valid_pairs_dist[prev],
								2 * prev_chunk_total_valid * sizeof(float),
								cudaMemcpyDeviceToHost,
								memcpy_stream);

				cudaEventRecord(copy_done[prev], memcpy_stream);

				CUDA_SAFE_CALL(cudaStreamSynchronize(memcpy_stream));
				
				// save in memory
				valid_voxel_pairs.resize(valid_voxel_pairs.size() + 3 * prev_chunk_total_valid);
				valid_voxel_pairs_dist.resize(valid_voxel_pairs_dist.size() + 2 * prev_chunk_total_valid);

				std::memcpy(valid_voxel_pairs.data() + prev_old_size,
							h_pairs_stage,
							3 * prev_chunk_total_valid * sizeof(int));

				std::memcpy(valid_voxel_pairs_dist.data() + prev_old_size_2,
							h_dist_stage,
							2 * prev_chunk_total_valid * sizeof(float));

				// save in disk 
				// {
				// 	std::unique_lock<std::mutex> lock(mtx);
				// 	cv.wait(lock, [&]() { return write_done; });
				// }

				// Buffer& buf = buffers[cur];
				// buf.pairs = h_pairs_stage;
				// buf.dist  = h_dist_stage;
				// buf.pair_count = prev_chunk_total_valid;
				// {
				// 	std::lock_guard<std::mutex> lock(mtx);
				// 	write_index = cur;
				// 	write_done = false;
				// }
				// cv.notify_one();
				// cur = 1 - cur;
			}

			cudaStreamSynchronize(compute_stream); // prefix 需要

			std::vector<size_t> chunk_count(chunk_pair_num);
			cudaMemcpy(chunk_count.data(), d_count,
					chunk_pair_num * sizeof(size_t),
					cudaMemcpyDeviceToHost);

			std::vector<size_t> chunk_valid_prefix(chunk_pair_num + 1, 0);
			for (size_t i = 0; i < chunk_pair_num; ++i)
				chunk_valid_prefix[i + 1] = chunk_valid_prefix[i] + chunk_count[i];

			size_t chunk_total_valid = chunk_valid_prefix[chunk_pair_num];

			size_t global_valid_base = valid_voxel_prefix[pair_begin];
			for (size_t i = 0; i < chunk_pair_num; ++i)
			{
				valid_voxel_cnt[pair_begin + i] = chunk_count[i];
				valid_voxel_prefix[pair_begin + i + 1] = global_valid_base + chunk_valid_prefix[i + 1];
			}

			size_t* d_chunk_prefix;
			cudaMalloc(&d_chunk_prefix, (chunk_pair_num + 1) * sizeof(size_t));
			cudaMemcpy(d_chunk_prefix,
					chunk_valid_prefix.data(),
					(chunk_pair_num + 1) * sizeof(size_t),
					cudaMemcpyHostToDevice);

			record_valid_voxel_pairs_streaming<<<chunk_pair_num, max_voxel_size, 0, compute_stream>>>(
				dv1, dv2,
				gd.d_obj1, gd.d_obj2,
				gd.d_prefix_obj,
				d_out_min,
				d_out_max,
				gd.d_min_max_dist,
				d_chunk_prefix,
				d_valid_pairs[curr],
				d_valid_pairs_dist[curr],
				pair_begin);

			fill_valid_voxel_distance_streaming<<<chunk_pair_num, max_voxel_size, 0, compute_stream>>>(
				gd.d_prefix_obj,
				d_chunk_prefix,
				d_valid_pairs[curr],
				d_out_min,
				d_out_max,
				d_valid_pairs_dist[curr],
				pair_begin);

			cudaEventRecord(compute_done[curr], compute_stream);

			// 更新 offset
			prev_old_size += 3 * prev_chunk_total_valid;
			prev_old_size_2 += 2 * prev_chunk_total_valid;
			prev_chunk_total_valid = chunk_total_valid;

			// valid_voxel_pairs.resize(prev_old_size + 3 * chunk_total_valid);
			// valid_voxel_pairs_dist.resize(prev_old_size_2 + 2 * chunk_total_valid);

			pair_begin = pair_end;
			iter++;

			cudaFree(d_out_min);
			cudaFree(d_out_max);
			cudaFree(d_count);
			cudaFree(d_chunk_prefix);
		}

		if (iter > 0) {
			int last = (iter - 1) % 2;
		
			// 等最后 compute 完成
			cudaStreamWaitEvent(memcpy_stream,
								compute_done[last],
								0);
			// 启动最后一次 memcpy
			cudaMemcpyAsync(h_pairs_stage,
							d_valid_pairs[last],
							3 * prev_chunk_total_valid * sizeof(int),
							cudaMemcpyDeviceToHost,
							memcpy_stream);

			cudaMemcpyAsync(h_dist_stage,
							d_valid_pairs_dist[last],
							2 * prev_chunk_total_valid * sizeof(float),
							cudaMemcpyDeviceToHost,
							memcpy_stream);

			// 等 memcpy 完成
			cudaStreamSynchronize(memcpy_stream);

			// save in memory
			valid_voxel_pairs.resize(valid_voxel_pairs.size() + 3 * prev_chunk_total_valid);
			valid_voxel_pairs_dist.resize(valid_voxel_pairs_dist.size() + 2 * prev_chunk_total_valid);

			std::memcpy(valid_voxel_pairs.data() + prev_old_size,
							h_pairs_stage,
							3 * prev_chunk_total_valid * sizeof(int));

			std::memcpy(valid_voxel_pairs_dist.data() + prev_old_size_2,
							h_dist_stage,
							2 * prev_chunk_total_valid * sizeof(float));

			// save in disk
			// {
			// 	std::unique_lock<std::mutex> lock(mtx);
			// 	cv.wait(lock, [&]() { return write_done; });
			// }

			// Buffer& buf = buffers[cur];
			// buf.pairs = h_pairs_stage;
			// buf.dist  = h_dist_stage;
			// buf.pair_count = prev_chunk_total_valid;
			// {
			// 	std::lock_guard<std::mutex> lock(mtx);
			// 	write_index = cur;
			// 	write_done = false;
			// }
			// cv.notify_one();
			// cur = 1 - cur;
		}

		{
			std::lock_guard<std::mutex> lock(mtx);
			exit_flag = true;
		}
		cv.notify_all();
		writer.join();

		std::cout << "valid_voxel_pairs = " << valid_voxel_pairs.size()
				<< ", valid_voxel_pairs_dist = " << valid_voxel_pairs_dist.size() << std::endl;


		auto t1 = std::chrono::high_resolution_clock::now();
		std::cout << "while time  = " << std::chrono::duration<double>(t1 - t0).count() << std::endl;


		CUDA_SAFE_CALL(cudaFreeHost(h_pairs_stage));
		CUDA_SAFE_CALL(cudaFreeHost(h_dist_stage));
		
		//=========== launch third kernel ==========
		int BLOCK = 256;
		int GRID  = dv1.tile_size;   // 一个 tile 一个 block

		evaluate_knn_kernel<<<GRID, BLOCK>>>(
			gd.d_min_min_dist,
			gd.d_min_max_dist,
			gd.d_prefix_tile,
			gd.d_confirmed_before,
			gd.d_status_after,
			dv1.tile_size,
			K
		);
		check_execution();
		CUDA_SAFE_CALL(cudaDeviceSynchronize());

		// status 是 per pair
		status_after.resize(pair_num);
		CUDA_SAFE_CALL(cudaMemcpy(
			status_after.data(),
			gd.d_status_after,
			pair_num * sizeof(int),
			cudaMemcpyDeviceToHost));

		confirmed_after.resize(dv1.tile_size);
		for (int t = 0; t < dv1.tile_size; ++t)
		{
			int begin = prefix_tile[t];
			int end   = prefix_tile[t + 1];

			int confirmed_cnt = 0;

			for (int i = begin; i < end; ++i)
			{
				if (status_after[i] == 1)  // CONFIRMED
					confirmed_cnt++;
				
				if (confirmed_cnt == K) break;
			}
			confirmed_after[t] = confirmed_cnt;
		}
	}

	void freeHost(int *valid_voxel_pairs,      			// we want to use pinnedMemory, so we don't use vector for these 2.
				float *valid_voxel_pairs_dist)
	{
		cudaFreeHost(valid_voxel_pairs);
		cudaFreeHost(valid_voxel_pairs_dist);
	}

	void compute_voxel_pair_distance_for_all_within(const DeviceVoxels &dv1,
												const DeviceVoxels &dv2,
												vector<int> &prefix_tile,
												vector<size_t> &prefix_obj, 
												vector<int> &compute_obj_1,
												vector<int> &compute_obj_2,
												size_t total_voxel_size,
												size_t max_voxel_size,
												vector<float> &h_min,
												vector<float> &h_max,
												vector<float> &min_max_dist,
												vector<float> &min_min_dist,
												vector<int> &valid_voxel_cnt,
												vector<size_t> &valid_voxel_prefix,
												vector<int> &valid_voxel_pairs,
												vector<float> &valid_voxel_pairs_dist
												)
	{
		int pair_num = compute_obj_1.size();

		// 1. 分配 device memory
		// int *d_obj1, *d_obj2;
		CUDA_SAFE_CALL(cudaMalloc(&gd.d_obj1, pair_num * sizeof(int)));
		CUDA_SAFE_CALL(cudaMalloc(&gd.d_obj2, pair_num * sizeof(int)));
		CUDA_SAFE_CALL(cudaMemcpy(gd.d_obj1, compute_obj_1.data(), pair_num * sizeof(int), cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(gd.d_obj2, compute_obj_2.data(), pair_num * sizeof(int), cudaMemcpyHostToDevice));

		// int *d_prefix_obj;
		CUDA_SAFE_CALL(cudaMalloc(&gd.d_prefix_obj, (pair_num + 1) * sizeof(size_t)));
		CUDA_SAFE_CALL(cudaMemcpy(gd.d_prefix_obj, prefix_obj.data(), (pair_num + 1) * sizeof(size_t), cudaMemcpyHostToDevice));

		// int *d_prefix_tile;
		CUDA_SAFE_CALL(cudaMalloc(&gd.d_prefix_tile, (dv1.tile_size + 1) * sizeof(int)));
		CUDA_SAFE_CALL(cudaMemcpy(gd.d_prefix_tile, prefix_tile.data(), (dv1.tile_size + 1) * sizeof(int), cudaMemcpyHostToDevice));

		// float *d_out_min, *d_out_max;
		CUDA_SAFE_CALL(cudaMallocManaged(&gd.d_out_min, total_voxel_size * sizeof(float)));
		CUDA_SAFE_CALL(cudaMallocManaged(&gd.d_out_max, total_voxel_size * sizeof(float)));

		// float *d_min_max_dist, *d_min_min_dist;
		CUDA_SAFE_CALL(cudaMalloc(&gd.d_min_max_dist, pair_num * sizeof(float)));
		CUDA_SAFE_CALL(cudaMalloc(&gd.d_min_min_dist, pair_num * sizeof(float)));

		// int *d_count;
		CUDA_SAFE_CALL(cudaMalloc(&gd.d_count, pair_num * sizeof(int)));

		if (max_voxel_size > 1024)
		{
			exit(-1);
		}

		voxel_pair_dist_kernel_for_all<<<pair_num, max_voxel_size>>>(
			dv1, dv2,
			gd.d_obj1, gd.d_obj2,
			gd.d_prefix_obj,
			gd.d_out_min, gd.d_out_max,
			gd.d_min_max_dist,
			gd.d_min_min_dist,
			gd.d_count);
		CUDA_SAFE_CALL(cudaDeviceSynchronize());

		valid_voxel_cnt.resize(pair_num);
		CUDA_SAFE_CALL(cudaMemcpy(valid_voxel_cnt.data(), gd.d_count, pair_num * sizeof(int), cudaMemcpyDeviceToHost));

		// h_min.resize(total_voxel_size);
		// h_max.resize(total_voxel_size);
		// CUDA_SAFE_CALL(cudaMemcpy(h_min.data(), gd.d_out_min, total_voxel_size * sizeof(float), cudaMemcpyDeviceToHost));
		// CUDA_SAFE_CALL(cudaMemcpy(h_max.data(), gd.d_out_max, total_voxel_size * sizeof(float), cudaMemcpyDeviceToHost));

		min_max_dist.resize(pair_num);
		min_min_dist.resize(pair_num);
		CUDA_SAFE_CALL(cudaMemcpy(min_max_dist.data(), gd.d_min_max_dist, pair_num * sizeof(float), cudaMemcpyDeviceToHost));
		CUDA_SAFE_CALL(cudaMemcpy(min_min_dist.data(), gd.d_min_min_dist, pair_num * sizeof(float), cudaMemcpyDeviceToHost));

		//============ launch second kernel =============
		valid_voxel_prefix.assign(pair_num + 1, 0);
		for (int i = 0; i < pair_num; ++i)
		{
			valid_voxel_prefix[i + 1] = valid_voxel_prefix[i] + valid_voxel_cnt[i];
		}

		size_t total_valid_voxel_pairs = valid_voxel_prefix[pair_num];

		// int *d_valid_voxel_prefix;
		// int *d_valid_voxel_pairs;

		CUDA_SAFE_CALL(cudaMallocManaged(&gd.d_valid_voxel_prefix, (pair_num + 1) * sizeof(size_t)));
		
		CUDA_SAFE_CALL(cudaMemcpy(gd.d_valid_voxel_prefix, valid_voxel_prefix.data(), (pair_num + 1) * sizeof(size_t), cudaMemcpyHostToDevice));

		CUDA_SAFE_CALL(cudaMallocManaged(&gd.d_valid_voxel_pairs, 3 * total_valid_voxel_pairs * sizeof(int)));

		float *d_valid_pairs_dist;

		CUDA_SAFE_CALL(cudaMallocManaged(&d_valid_pairs_dist, 2 * total_valid_voxel_pairs * sizeof(float)));

		// this step is intended for future use, not for this step!!!!
		record_valid_voxel_pairs<<<pair_num, max_voxel_size>>>(
			dv1, dv2,
			gd.d_obj1, gd.d_obj2,
			gd.d_prefix_obj,
			gd.d_out_min,
			gd.d_out_max,
			gd.d_min_max_dist,
			gd.d_valid_voxel_prefix,
			gd.d_valid_voxel_pairs,
			d_valid_pairs_dist
		);
		CUDA_SAFE_CALL(cudaDeviceSynchronize());

		fill_valid_voxel_distance<<<pair_num, max_voxel_size>>>(
			gd.d_prefix_obj,      		// voxel prefix
			gd.d_valid_voxel_prefix,   	// valid voxel prefix
			gd.d_valid_voxel_pairs,     // [v1, v2, t]
			gd.d_out_min,
			gd.d_out_max,
			d_valid_pairs_dist
		);
		CUDA_SAFE_CALL(cudaDeviceSynchronize());

		valid_voxel_pairs.resize(3 * total_valid_voxel_pairs);
		CUDA_SAFE_CALL(cudaMemcpy(valid_voxel_pairs.data(),
								  gd.d_valid_voxel_pairs,
								  3 * total_valid_voxel_pairs * sizeof(int),
								  cudaMemcpyDeviceToHost));
		
		valid_voxel_pairs_dist.resize(2 * total_valid_voxel_pairs);

		CUDA_SAFE_CALL(cudaMemcpy(valid_voxel_pairs_dist.data(),
						d_valid_pairs_dist,
						2 * total_valid_voxel_pairs * sizeof(float),
						cudaMemcpyDeviceToHost));

		// do not free temporarily
		// CUDA_SAFE_CALL(cudaFree(d_obj1));
		// CUDA_SAFE_CALL(cudaFree(d_obj2));
		CUDA_SAFE_CALL(cudaFree(gd.d_out_min));
		CUDA_SAFE_CALL(cudaFree(gd.d_out_max));
		// CUDA_SAFE_CALL(cudaFree(d_min_max_dist));
		// CUDA_SAFE_CALL(cudaFree(d_min_min_dist));
		// CUDA_SAFE_CALL(cudaFree(d_count));
	}

	void compute_voxel_pair_distance_for_all_within_streaming(const DeviceVoxels &dv1,
												const DeviceVoxels &dv2,
												vector<int> &prefix_tile,
												vector<size_t> &prefix_obj, 
												vector<int> &compute_obj_1,
												vector<int> &compute_obj_2,
												size_t total_voxel_size,
												size_t max_voxel_size,
												vector<float> &h_min,
												vector<float> &h_max,
												vector<float> &min_max_dist,
												vector<float> &min_min_dist,
												vector<int> &valid_voxel_cnt,
												vector<size_t> &valid_voxel_prefix,
												vector<int> &valid_voxel_pairs,
												vector<float> &valid_voxel_pairs_dist
	)
	{
		int pair_num = compute_obj_1.size();

		CUDA_SAFE_CALL(cudaMalloc(&gd.d_obj1, pair_num * sizeof(int)));
		CUDA_SAFE_CALL(cudaMalloc(&gd.d_obj2, pair_num * sizeof(int)));
		CUDA_SAFE_CALL(cudaMemcpy(gd.d_obj1, compute_obj_1.data(), pair_num * sizeof(int), cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(gd.d_obj2, compute_obj_2.data(), pair_num * sizeof(int), cudaMemcpyHostToDevice));

		// int *d_prefix_obj;
		CUDA_SAFE_CALL(cudaMalloc(&gd.d_prefix_obj, (pair_num + 1) * sizeof(size_t)));
		CUDA_SAFE_CALL(cudaMemcpy(gd.d_prefix_obj, prefix_obj.data(), (pair_num + 1) * sizeof(size_t), cudaMemcpyHostToDevice));

		// int *d_prefix_tile;
		CUDA_SAFE_CALL(cudaMalloc(&gd.d_prefix_tile, (dv1.tile_size + 1) * sizeof(int)));
		CUDA_SAFE_CALL(cudaMemcpy(gd.d_prefix_tile, prefix_tile.data(), (dv1.tile_size + 1) * sizeof(int), cudaMemcpyHostToDevice));

		// float *d_out_min, *d_out_max;
		// We don't allocate all at once 
		// CUDA_SAFE_CALL(cudaMalloc(&gd.d_out_min, total_voxel_size * sizeof(float)));
		// CUDA_SAFE_CALL(cudaMalloc(&gd.d_out_max, total_voxel_size * sizeof(float)));

		// float *d_min_max_dist, *d_min_min_dist;
		CUDA_SAFE_CALL(cudaMalloc(&gd.d_min_max_dist, pair_num * sizeof(float)));
		CUDA_SAFE_CALL(cudaMalloc(&gd.d_min_min_dist, pair_num * sizeof(float)));

		// int *d_count;
		// CUDA_SAFE_CALL(cudaMalloc(&gd.d_count, pair_num * sizeof(int)));



		size_t GPU_MEMORY_LIMIT = 20ULL * 1024 * 1024 * 1024; // 10GB
		size_t bytes_per_voxel = 2 * sizeof(float);           // min and max
		size_t MAX_CHUNK_VOXELS = GPU_MEMORY_LIMIT / bytes_per_voxel;


		valid_voxel_cnt.assign(pair_num, 0);
		valid_voxel_prefix.assign(pair_num + 1, 0);

		min_max_dist.resize(pair_num);
		min_min_dist.resize(pair_num);

		valid_voxel_pairs.reserve(1e8 * 3);
		valid_voxel_pairs_dist.reserve(1e8 * 2);



		size_t pair_begin = 0;

		while (pair_begin < pair_num)
		{
			size_t pair_end = pair_begin;
			size_t chunk_voxel_size = 0;

			while (pair_end < pair_num)
			{
				size_t voxel_cnt = prefix_obj[pair_end + 1] - prefix_obj[pair_end];

				if (chunk_voxel_size + voxel_cnt > MAX_CHUNK_VOXELS)
					break;

				chunk_voxel_size += voxel_cnt;
				pair_end++;
			}

			size_t chunk_pair_num = pair_end - pair_begin;

			std::cout << "Chunk pairs: " << pair_begin << " - " << pair_end << std::endl;

			float *d_out_min;
			float *d_out_max;
			size_t *d_count;

			CUDA_SAFE_CALL(cudaMalloc(&d_out_min, chunk_voxel_size * sizeof(float)));

			CUDA_SAFE_CALL(cudaMalloc(&d_out_max, chunk_voxel_size * sizeof(float)));

			CUDA_SAFE_CALL(cudaMalloc(&d_count, chunk_pair_num * sizeof(size_t)));


			voxel_pair_dist_kernel_for_all_streaming<<<chunk_pair_num, max_voxel_size>>>(
				dv1, dv2,
				gd.d_obj1, gd.d_obj2,
				gd.d_prefix_obj,
				d_out_min, 
				d_out_max,
				gd.d_min_max_dist,
				gd.d_min_min_dist,
				d_count,
				pair_begin
			);
			check_execution();
			CUDA_SAFE_CALL(cudaDeviceSynchronize());


			size_t host_offset = prefix_obj[pair_begin];

			vector<size_t> chunk_count(chunk_pair_num);

        	CUDA_SAFE_CALL(cudaMemcpy(chunk_count.data(),
									d_count,
									chunk_pair_num * sizeof(size_t),
									cudaMemcpyDeviceToHost));


			vector<size_t> chunk_valid_prefix(chunk_pair_num + 1, 0);

			for (size_t i = 0; i < chunk_pair_num; ++i)
			{
				chunk_valid_prefix[i + 1] = chunk_valid_prefix[i] + chunk_count[i];
			}

			size_t chunk_total_valid = chunk_valid_prefix[chunk_pair_num];

			int* d_valid_pairs;
			float *d_valid_pairs_dist;
			size_t* d_chunk_prefix;

			std::cout << "chunk_total_valid = " << chunk_total_valid << std::endl;

			CUDA_SAFE_CALL(cudaMalloc(&d_valid_pairs, 3 * chunk_total_valid * sizeof(int)));

			CUDA_SAFE_CALL(cudaMalloc(&d_valid_pairs_dist, 2 * chunk_total_valid * sizeof(float)));

			CUDA_SAFE_CALL(cudaMalloc(&d_chunk_prefix, (chunk_pair_num + 1) * sizeof(size_t)));

			CUDA_SAFE_CALL(cudaMemcpy(d_chunk_prefix,
						chunk_valid_prefix.data(),
						(chunk_pair_num + 1) * sizeof(size_t),
						cudaMemcpyHostToDevice));
		

			record_valid_voxel_pairs_streaming<<<chunk_pair_num, max_voxel_size>>>(
				dv1, dv2,
				gd.d_obj1, gd.d_obj2,
				gd.d_prefix_obj,
				d_out_min,
				d_out_max,
				gd.d_min_max_dist,
				d_chunk_prefix,
				d_valid_pairs,
				d_valid_pairs_dist,
				pair_begin
			);
			check_execution();
			CUDA_SAFE_CALL(cudaDeviceSynchronize());

			fill_valid_voxel_distance_streaming<<<chunk_pair_num, max_voxel_size>>>(
				gd.d_prefix_obj,      
				d_chunk_prefix,
				d_valid_pairs,
				d_out_min,
				d_out_max,
				d_valid_pairs_dist,
				pair_begin
			);
			check_execution();
			CUDA_SAFE_CALL(cudaDeviceSynchronize());

			size_t old_size = valid_voxel_pairs.size();
			valid_voxel_pairs.resize(old_size + 3 * chunk_total_valid);

			size_t old_size_2 = valid_voxel_pairs_dist.size();
			valid_voxel_pairs_dist.resize(old_size_2 + 2 * chunk_total_valid);

			CUDA_SAFE_CALL(cudaMemcpy(valid_voxel_pairs.data() + old_size,
						d_valid_pairs,
						3 * chunk_total_valid * sizeof(int),
						cudaMemcpyDeviceToHost));

			CUDA_SAFE_CALL(cudaMemcpy(valid_voxel_pairs_dist.data() + old_size_2,
						d_valid_pairs_dist,
						2 * chunk_total_valid * sizeof(float),
						cudaMemcpyDeviceToHost));

			size_t global_valid_base = valid_voxel_prefix[pair_begin];

			for (size_t i = 0; i < chunk_pair_num; ++i)
			{
				valid_voxel_cnt[pair_begin + i] = chunk_count[i];
				valid_voxel_prefix[pair_begin + i + 1] = global_valid_base + chunk_valid_prefix[i + 1];
			}

			CUDA_SAFE_CALL(cudaFree(d_out_min));
			CUDA_SAFE_CALL(cudaFree(d_out_max));
			CUDA_SAFE_CALL(cudaFree(d_count));
			CUDA_SAFE_CALL(cudaFree(d_valid_pairs));
			CUDA_SAFE_CALL(cudaFree(d_valid_pairs_dist));
			CUDA_SAFE_CALL(cudaFree(d_chunk_prefix));

			pair_begin = pair_end;
		}
	}

	void compute_voxel_pair_distance_for_all_within_streaming_pipeline(const DeviceVoxels &dv1,
												const DeviceVoxels &dv2,
												vector<int> &prefix_tile,
												vector<size_t> &prefix_obj, 
												vector<int> &compute_obj_1,
												vector<int> &compute_obj_2,
												size_t total_voxel_size,
												size_t max_voxel_size,
												vector<float> &h_min,
												vector<float> &h_max,
												vector<float> &min_max_dist,
												vector<float> &min_min_dist,
												vector<int> &valid_voxel_cnt,
												vector<size_t> &valid_voxel_prefix,
												vector<int> &valid_voxel_pairs,
												vector<float> &valid_voxel_pairs_dist,
												float within_distance
	)
	{
		int pair_num = compute_obj_1.size();

		CUDA_SAFE_CALL(cudaMalloc(&gd.d_obj1, pair_num * sizeof(int)));
		CUDA_SAFE_CALL(cudaMalloc(&gd.d_obj2, pair_num * sizeof(int)));
		CUDA_SAFE_CALL(cudaMemcpy(gd.d_obj1, compute_obj_1.data(), pair_num * sizeof(int), cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(gd.d_obj2, compute_obj_2.data(), pair_num * sizeof(int), cudaMemcpyHostToDevice));

		CUDA_SAFE_CALL(cudaMalloc(&gd.d_prefix_obj, (pair_num + 1) * sizeof(size_t)));
		CUDA_SAFE_CALL(cudaMemcpy(gd.d_prefix_obj, prefix_obj.data(), (pair_num + 1) * sizeof(size_t), cudaMemcpyHostToDevice));

		CUDA_SAFE_CALL(cudaMalloc(&gd.d_prefix_tile, (dv1.tile_size + 1) * sizeof(int)));
		CUDA_SAFE_CALL(cudaMemcpy(gd.d_prefix_tile, prefix_tile.data(), (dv1.tile_size + 1) * sizeof(int), cudaMemcpyHostToDevice));

		// float *d_out_min, *d_out_max;
		// We don't allocate all at once 
		// CUDA_SAFE_CALL(cudaMalloc(&gd.d_out_min, total_voxel_size * sizeof(float)));
		// CUDA_SAFE_CALL(cudaMalloc(&gd.d_out_max, total_voxel_size * sizeof(float)));

		// float *d_min_max_dist, *d_min_min_dist;
		CUDA_SAFE_CALL(cudaMalloc(&gd.d_min_max_dist, pair_num * sizeof(float)));
		CUDA_SAFE_CALL(cudaMalloc(&gd.d_min_min_dist, pair_num * sizeof(float)));


		size_t GPU_MEMORY_LIMIT = 1 * 1024ULL * 1024 * 1024; // 500MB
		size_t bytes_per_voxel = 2 * sizeof(float);           // min and max
		size_t MAX_CHUNK_VOXELS = GPU_MEMORY_LIMIT / bytes_per_voxel;

		const float RATIO = 0.07;

		size_t MAX_VALID = MAX_CHUNK_VOXELS;

		valid_voxel_cnt.assign(pair_num, 0);
		valid_voxel_prefix.assign(pair_num + 1, 0);

		int *d_valid_voxel_cnt;
		size_t *d_valid_voxel_prefix;

		CUDA_SAFE_CALL(cudaMalloc(&d_valid_voxel_cnt, pair_num * sizeof(int)));
		CUDA_SAFE_CALL(cudaMalloc(&d_valid_voxel_prefix, (pair_num + 1) * sizeof(size_t)));
		CUDA_SAFE_CALL(cudaMemset(d_valid_voxel_cnt, 0, pair_num * sizeof(int)));
		CUDA_SAFE_CALL(cudaMemset(d_valid_voxel_prefix, 0, (pair_num + 1) * sizeof(size_t)));


		min_max_dist.resize(pair_num);
		min_min_dist.resize(pair_num);

		// valid_voxel_pairs.reserve(total_voxel_size * RATIO * 3);
		// valid_voxel_pairs_dist.reserve(total_voxel_size * RATIO * 2);

		valid_voxel_pairs.reserve(1e9 * 3);
		valid_voxel_pairs_dist.reserve(1e9 * 2);

		auto tx = std::chrono::high_resolution_clock::now();

		CUDA_SAFE_CALL(cudaHostRegister(valid_voxel_pairs.data(),
                                valid_voxel_pairs.capacity() * sizeof(int),
                                cudaHostRegisterDefault));

		CUDA_SAFE_CALL(cudaHostRegister(valid_voxel_pairs_dist.data(),
                                valid_voxel_pairs_dist.capacity() * sizeof(float),
                                cudaHostRegisterDefault));
		
		auto ty = std::chrono::high_resolution_clock::now();
		std::cout << "Pinned memory time = " << std::chrono::duration<double>(ty - tx).count() << std::endl;

		size_t BLKDIM = min(max_voxel_size, (size_t)1024);


		size_t pair_begin = 0;
		
		cudaStream_t compute_stream, memcpy_stream;
		cudaStreamCreate(&compute_stream);
		cudaStreamCreate(&memcpy_stream);

		cudaEvent_t compute_done[2], copy_done[2];
		cudaEventCreate(&compute_done[0]);
		cudaEventCreate(&compute_done[1]);
		cudaEventCreate(&copy_done[0]);
		cudaEventCreate(&copy_done[1]);

		int* d_valid_pairs[2];
		float* d_valid_pairs_dist[2];

		cudaMalloc(&d_valid_pairs[0], 3 * MAX_VALID * sizeof(int));
		cudaMalloc(&d_valid_pairs[1], 3 * MAX_VALID * sizeof(int));
		cudaMalloc(&d_valid_pairs_dist[0], 2 * MAX_VALID * sizeof(float));
		cudaMalloc(&d_valid_pairs_dist[1], 2 * MAX_VALID * sizeof(float));

		float *d_out_min, *d_out_max;
		size_t *d_count;

		cudaMalloc(&d_out_min, MAX_CHUNK_VOXELS * sizeof(float));
		cudaMalloc(&d_out_max, MAX_CHUNK_VOXELS * sizeof(float));
		cudaMalloc(&d_count, pair_num * sizeof(size_t));


		// auto tx = std::chrono::high_resolution_clock::now();

		// size_t STAGING_PAIRS_BYTES = 3 * MAX_VALID * sizeof(int);
		// size_t STAGING_DIST_BYTES  = 2 * MAX_VALID * sizeof(float);

		// int*   h_pairs_stage;
		// float* h_dist_stage;

		// CUDA_SAFE_CALL(cudaMallocHost(&h_pairs_stage, STAGING_PAIRS_BYTES));
		// CUDA_SAFE_CALL(cudaMallocHost(&h_dist_stage,  STAGING_DIST_BYTES));

		// auto ty = std::chrono::high_resolution_clock::now();
		// std::cout << "Pinned memory time = " << std::chrono::duration<double>(ty - tx).count() << std::endl;


		int iter = 0;
		size_t prev_chunk_total_valid = 0;
		size_t prev_old_size = 0, prev_old_size_2 = 0;

		
		while (pair_begin < pair_num)
		{
			int curr = iter % 2;
			int prev = 1 - curr;

			size_t pair_end = pair_begin;
			size_t chunk_voxel_size = 0;

			while (pair_end < pair_num)
			{
				size_t voxel_cnt = prefix_obj[pair_end + 1] - prefix_obj[pair_end];
				if (chunk_voxel_size + voxel_cnt > MAX_CHUNK_VOXELS)
					break;
				chunk_voxel_size += voxel_cnt;
				pair_end++;
			}

			size_t chunk_pair_num = pair_end - pair_begin;

			std::cout << "Chunk pairs: " << pair_begin << " - " << pair_end << std::endl;

			// float *d_out_min, *d_out_max;
			// size_t *d_count;

			// cudaMalloc(&d_out_min, chunk_voxel_size * sizeof(float));
			// cudaMalloc(&d_out_max, chunk_voxel_size * sizeof(float));
			// cudaMalloc(&d_count, chunk_pair_num * sizeof(size_t));


			voxel_pair_dist_kernel_for_all_streaming_within<<<chunk_pair_num, BLKDIM, 0, compute_stream>>>(
				dv1, dv2,
				gd.d_obj1, gd.d_obj2,
				gd.d_prefix_obj,
				d_out_min,
				d_out_max,
				gd.d_min_max_dist,
				gd.d_min_min_dist,
				d_count,
				pair_begin,
				within_distance
			);

			if (iter > 0)
			{
				cudaStreamWaitEvent(memcpy_stream, compute_done[prev], 0);

				cudaMemcpyAsync( valid_voxel_pairs.data() + prev_old_size,
								d_valid_pairs[prev],
								3 * prev_chunk_total_valid * sizeof(int),
								cudaMemcpyDeviceToHost,
								memcpy_stream);

				cudaMemcpyAsync( valid_voxel_pairs_dist.data() + prev_old_size_2,
								d_valid_pairs_dist[prev],
								2 * prev_chunk_total_valid * sizeof(float),
								cudaMemcpyDeviceToHost,
								memcpy_stream);

				cudaEventRecord(copy_done[prev], memcpy_stream);

				// CUDA_SAFE_CALL(cudaStreamSynchronize(memcpy_stream));

				// valid_voxel_pairs.resize(valid_voxel_pairs.size() + 3 * prev_chunk_total_valid);
				// valid_voxel_pairs_dist.resize(valid_voxel_pairs_dist.size() + 2 * prev_chunk_total_valid);

				// std::memcpy(valid_voxel_pairs.data() + prev_old_size,
				// 			h_pairs_stage,
				// 			3 * prev_chunk_total_valid * sizeof(int));

				// std::memcpy(valid_voxel_pairs_dist.data() + prev_old_size_2,
				// 			h_dist_stage,
				// 			2 * prev_chunk_total_valid * sizeof(float));
			}

			cudaStreamSynchronize(compute_stream); 

			size_t* d_chunk_prefix;
			cudaMalloc(&d_chunk_prefix, (chunk_pair_num + 1) * sizeof(size_t));
			inclusive_scan(d_count, d_chunk_prefix, chunk_pair_num);

			// cudaMemcpy(chunk_valid_prefix.data(), d_chunk_prefix,
			// 		(chunk_pair_num + 1) * sizeof(size_t),
			// 		cudaMemcpyDeviceToHost);

			// Only need the last element
			size_t chunk_total_valid;
			cudaMemcpy(&chunk_total_valid, d_chunk_prefix + chunk_pair_num, sizeof(size_t), cudaMemcpyDeviceToHost);
			

			// Let GPU do this
			fill_count_and_prefix_kernel<<<(chunk_pair_num + 255)/256, 256, 0, compute_stream>>>(d_valid_voxel_cnt,
																								d_valid_voxel_prefix,
																								d_count,
																								d_chunk_prefix,
																								pair_begin,
																								chunk_pair_num
																							);
			cudaStreamWaitEvent(compute_stream, copy_done[curr], 0);
			
			record_valid_voxel_pairs_streaming_within<<<chunk_pair_num, BLKDIM, 0, compute_stream>>>(
				dv1, dv2,
				gd.d_obj1, gd.d_obj2,
				gd.d_prefix_obj,
				d_out_min,
				d_out_max,
				gd.d_min_max_dist,
				gd.d_min_min_dist,
				d_chunk_prefix,
				d_valid_pairs[curr],
				d_valid_pairs_dist[curr],
				pair_begin,
				within_distance
			);

			fill_valid_voxel_distance_streaming<<<chunk_pair_num, BLKDIM, 0, compute_stream>>>(
				gd.d_prefix_obj,
				d_chunk_prefix,
				d_valid_pairs[curr],
				d_out_min,
				d_out_max,
				d_valid_pairs_dist[curr],
				pair_begin);

			cudaEventRecord(compute_done[curr], compute_stream);

			// update offset
			prev_old_size += 3 * prev_chunk_total_valid;
			prev_old_size_2 += 2 * prev_chunk_total_valid;
			prev_chunk_total_valid = chunk_total_valid;

			// valid_voxel_pairs.resize(prev_old_size + 3 * chunk_total_valid);
			// valid_voxel_pairs_dist.resize(prev_old_size_2 + 2 * chunk_total_valid);

			pair_begin = pair_end;
			iter++;

			// cudaFree(d_out_min);
			// cudaFree(d_out_max);
			// cudaFree(d_count);
			// cudaFree(d_chunk_prefix);
		}

		if (iter > 0) {
			int last = (iter - 1) % 2;
		
			// wait for compute 
			cudaStreamWaitEvent(memcpy_stream,
								compute_done[last],
								0);
			// memcpy last round
			cudaMemcpyAsync( valid_voxel_pairs.data() + prev_old_size,
							d_valid_pairs[last],
							3 * prev_chunk_total_valid * sizeof(int),
							cudaMemcpyDeviceToHost,
							memcpy_stream);

			cudaMemcpyAsync( valid_voxel_pairs_dist.data() + prev_old_size_2,
							d_valid_pairs_dist[last],
							2 * prev_chunk_total_valid * sizeof(float),
							cudaMemcpyDeviceToHost,
							memcpy_stream);

			// wait memcpy 
			cudaStreamSynchronize(memcpy_stream);


			// valid_voxel_pairs.resize(valid_voxel_pairs.size() + 3 * prev_chunk_total_valid);
			// valid_voxel_pairs_dist.resize(valid_voxel_pairs_dist.size() + 2 * prev_chunk_total_valid);

			// std::memcpy(valid_voxel_pairs.data() + prev_old_size,
			// 				h_pairs_stage,
			// 				3 * prev_chunk_total_valid * sizeof(int));

			// std::memcpy(valid_voxel_pairs_dist.data() + prev_old_size_2,
			// 				h_dist_stage,
			// 				2 * prev_chunk_total_valid * sizeof(float));
		}


		min_max_dist.resize(pair_num);
		min_min_dist.resize(pair_num);
		CUDA_SAFE_CALL(cudaMemcpy(min_max_dist.data(), gd.d_min_max_dist, pair_num * sizeof(float), cudaMemcpyDeviceToHost));
		CUDA_SAFE_CALL(cudaMemcpy(min_min_dist.data(), gd.d_min_min_dist, pair_num * sizeof(float), cudaMemcpyDeviceToHost));

		CUDA_SAFE_CALL(cudaMemcpy(valid_voxel_cnt.data(), d_valid_voxel_cnt, pair_num * sizeof(int), cudaMemcpyDeviceToHost));
		CUDA_SAFE_CALL(cudaMemcpy(valid_voxel_prefix.data(), d_valid_voxel_prefix, (pair_num + 1) * sizeof(size_t), cudaMemcpyDeviceToHost));

		// CUDA_SAFE_CALL(cudaFreeHost(h_pairs_stage));
		// CUDA_SAFE_CALL(cudaFreeHost(h_dist_stage));

		CUDA_SAFE_CALL(cudaHostUnregister(valid_voxel_pairs.data()));
		CUDA_SAFE_CALL(cudaHostUnregister(valid_voxel_pairs_dist.data()));


		cudaFree(d_out_min);
		cudaFree(d_out_max);
		cudaFree(d_count);
	}

	void free_cuda_memory(void *ptr)
	{
		CUDA_SAFE_CALL(cudaFree(ptr));
	}

	/*
	 * data: contains the triangles of the meshes in this join.
	 * offset_size:  contains the offset in the data for each batch, and the sizes of two data sets
	 * result: for the returned results for each batch
	 * pair_num: number of computed batches
	 *
	 * */
	void MeshDist_batch_gpu(gpu_info *gpu, const float *data, const size_t *offset_size, const float *hausdorff,
							result_container *result, const uint32_t pair_num, const uint32_t element_num)
	{
		struct timeval start = get_cur_time();
		assert(gpu);
		cudaSetDevice(gpu->device_id);
		// profile the input data
		uint32_t max_size_1 = 0;
		uint32_t max_size_2 = 0;
		for (int i = 0; i < pair_num; i++)
		{
			if (offset_size[i * 4 + 1] > max_size_1)
			{
				max_size_1 = offset_size[i * 4 + 1];
			}
			if (offset_size[i * 4 + 3] > max_size_2)
			{
				max_size_2 = offset_size[i * 4 + 3];
			}
		}
		// allocate memory in GPU
		// char *cur_d_cuda = gpu->d_data;

		char *cur_d_cuda, *d_start;
		CUDA_SAFE_CALL(cudaMalloc((void**)&cur_d_cuda, 9 * sizeof(float) * element_num 
											  +	sizeof(result_container) * pair_num
											  + 4 * sizeof(size_t) * pair_num 
											  + element_num * 2 * sizeof(float) + 32)); // 32 for padding
		d_start = cur_d_cuda;



		size_t offset = 0;
		offset = align_up(offset, alignof(float));
		float *d_data = reinterpret_cast<float*>(cur_d_cuda + offset);
		offset += 9 * sizeof(float) * element_num;

		offset = align_up(offset, alignof(result_container));
		result_container *d_dist = reinterpret_cast<result_container*>(cur_d_cuda + offset);
		offset += sizeof(result_container) * pair_num;

		offset = align_up(offset, alignof(size_t));
		size_t *d_os = reinterpret_cast<size_t*>(cur_d_cuda + offset);
		offset += 4 * sizeof(size_t) * pair_num;

		offset = align_up(offset, alignof(float));
		float *d_hausdorff = reinterpret_cast<float*>(cur_d_cuda + offset);
		offset += 2 * sizeof(float) * element_num;

		offset = align_up(offset, alignof(int));
		int *d_location = reinterpret_cast<int*>(cur_d_cuda + offset);
		offset += sizeof(int) * 3 * pair_num;


		// segment data in device
		// float *d_data = (float *)(cur_d_cuda);
		// cur_d_cuda += 9 * sizeof(float) * element_num;
		// // space for the results in GPU
		// result_container *d_dist = (result_container *)(cur_d_cuda);
		// cur_d_cuda += sizeof(result_container) * pair_num;
		// // space for the offset and size information in GPU
		// size_t *d_os = (size_t *)(cur_d_cuda);
		// cur_d_cuda += 4 * sizeof(size_t) * pair_num;
		// // space for the hausdorff distances in GPU
		// float *d_hausdorff = (float *)cur_d_cuda;

		CUDA_SAFE_CALL(cudaMemcpy(d_data, data, element_num * 9 * sizeof(float), cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(d_os, offset_size, pair_num * 4 * sizeof(size_t), cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(d_hausdorff, hausdorff, element_num * 2 * sizeof(float), cudaMemcpyHostToDevice));
		// // initialize results container array
		// CUDA_SAFE_CALL(cudaMemcpy(d_dist, result, pair_num*sizeof(result_container), cudaMemcpyHostToDevice));

		// logt("copying data to GPU", start);

		int kernel_launch_times = 0;

#define MAX_DIM 1024
		// compute the distance in parallel
		for (uint32_t tri_offset_2 = 0; tri_offset_2 < max_size_2; tri_offset_2 += MAX_DIM)
		{
			uint32_t dim2 = min(max_size_2 - tri_offset_2, (uint32_t)MAX_DIM);
			for (uint32_t tri_offset_1 = 0; tri_offset_1 < max_size_1; tri_offset_1++)
			{
				TriDist_cuda<<<pair_num, dim2>>>(d_data, d_os, d_hausdorff, d_dist, tri_offset_1, tri_offset_2);
				kernel_launch_times += 1;
				check_execution();
			}
			// cout<<pair_num<<" "<< tri_offset_2 <<" "<<dim2<<" "<< max_size_1 <<" "<< max_size_2 << endl;
		}
		cudaDeviceSynchronize();
		// logt("distances computations", start);

		CUDA_SAFE_CALL(cudaMemcpy(result, d_dist, pair_num * sizeof(result_container), cudaMemcpyDeviceToHost));
		CUDA_SAFE_CALL(cudaFree(d_start));
		// logt("copy data out", start);


		std::cout << "kernel_launch_times = " << kernel_launch_times << std::endl;
	}

	void MeshDist_batch_gpu_op_shm(gpu_info *gpu, const float *data, const uint32_t *offset_size, const float *hausdorff,
								   result_container *result, const uint32_t pair_num, const uint32_t element_num)
	{
		struct timeval start = get_cur_time();
		assert(gpu);
		cudaSetDevice(gpu->device_id);
		// profile the input data
		uint32_t max_size_1 = 0;
		uint32_t max_size_2 = 0;
		for (int i = 0; i < pair_num; i++)
		{
			if (offset_size[i * 4 + 1] > max_size_1)
			{
				max_size_1 = offset_size[i * 4 + 1];
			}
			if (offset_size[i * 4 + 3] > max_size_2)
			{
				max_size_2 = offset_size[i * 4 + 3];
			}
		}
		// allocate memory in GPU
		char *cur_d_cuda = gpu->d_data;

		// segment data in device
		float *d_data = (float *)(cur_d_cuda);
		cur_d_cuda += 9 * sizeof(float) * element_num;
		// space for the results in GPU
		result_container *d_dist = (result_container *)(cur_d_cuda);
		cur_d_cuda += sizeof(result_container) * pair_num;
		// space for the offset and size information in GPU
		uint32_t *d_os = (uint32_t *)(cur_d_cuda);
		cur_d_cuda += 4 * sizeof(uint32_t) * pair_num;
		// space for the hausdorff distances in GPU
		float *d_hausdorff = (float *)cur_d_cuda;

		CUDA_SAFE_CALL(cudaMemcpy(d_data, data, element_num * 9 * sizeof(float), cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(d_os, offset_size, pair_num * 4 * sizeof(uint32_t), cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(d_hausdorff, hausdorff, element_num * 2 * sizeof(float), cudaMemcpyHostToDevice));
		// initialize results container array
		CUDA_SAFE_CALL(cudaMemcpy(d_dist, result, pair_num * sizeof(result_container), cudaMemcpyHostToDevice));

		// logt("copying data to GPU", start);

#define MAX_DIM 1024
		// compute the distance in parallel
		for (uint32_t tri_offset_2 = 0; tri_offset_2 < max_size_2; tri_offset_2 += MAX_DIM)
		{
			uint32_t dim2 = min(max_size_2 - tri_offset_2, (uint32_t)MAX_DIM);
			TriDist_cuda_op_shm<<<pair_num, dim2>>>(d_data, d_os, d_hausdorff, d_dist, max_size_1, tri_offset_2);
			check_execution();
		}
		cudaDeviceSynchronize();
		// logt("distances computations", start);

		CUDA_SAFE_CALL(cudaMemcpy(result, d_dist, pair_num * sizeof(result_container), cudaMemcpyDeviceToHost));
		// logt("copy data out", start);
	}

	__global__ void GPU_array_initialize(
		float* d_min_min_dist,
		float* d_min_max_dist,
		int    n
	)
	{
		int tid = threadIdx.x;

		if (tid < n) {
			d_min_min_dist[tid] = FLT_MAX;
			d_min_max_dist[tid] = FLT_MAX;
		}
	}

	void MeshDist_batch_gpu_op_shm_flat(gpu_info *gpu, 
										const float *data, 
										const size_t *offset_size, 
										const float *hausdorff,
										result_container *result, 
										const uint32_t pair_num, 
										const uint32_t element_num,
										/** additional params */
										int *location,
										int obj_pair_num,
										size_t max_voxel_size,
										int N,
										int K,
										vector<int> &confirmed_after,
										vector<int> &status_after,
										vector<int> &valid_voxel_prefix,
										vector<int> &valid_voxel_pairs,
										int lod,
										size_t total_voxel_size,
										vector<float> &min_min_dist,
										vector<float> &min_max_dist
									)
	{
		struct timeval start = get_cur_time();
		assert(gpu);
		cudaSetDevice(gpu->device_id);
		// profile the input data
		uint32_t max_size_1 = 0;
		uint32_t max_size_2 = 0;
		uint32_t max_tasks = 0;

		std::vector<int> h_pair_tasks(pair_num);
		std::vector<int> h_prefix(pair_num + 1, 0);


		for (int i = 0; i < pair_num; i++)
		{
			if (offset_size[i * 4 + 1] > max_size_1)
			{
				max_size_1 = offset_size[i * 4 + 1];
			}
			if (offset_size[i * 4 + 3] > max_size_2)
			{
				max_size_2 = offset_size[i * 4 + 3];
			}
			if (offset_size[i * 4 + 1] * offset_size[i * 4 + 3] > max_tasks)
			{
				max_tasks = offset_size[i * 4 + 1] * offset_size[i * 4 + 3];
			}

			int tasks = offset_size[i * 4 + 1] * offset_size[i * 4 + 3];

			h_pair_tasks[i] = tasks;
    		h_prefix[i + 1] = h_prefix[i] + tasks;
		}

		int total_tasks = h_prefix[pair_num];

		size_t gpu_memory_occ = 9 * sizeof(float) * (size_t)element_num 
											  +	sizeof(result_container) * pair_num
											  + 4 * sizeof(size_t) * pair_num 
											  + element_num * 2 * sizeof(float);


		std::cout << "GPU memory at refinement = " << gpu_memory_occ << std::endl;


		// allocate memory in GPU
		char *cur_d_cuda, *d_start;
		CUDA_SAFE_CALL(cudaMalloc((void**)&cur_d_cuda, 9 * sizeof(float) * element_num 
											  +	sizeof(result_container) * pair_num
											  + 4 * sizeof(size_t) * pair_num 
											  + element_num * 2 * sizeof(float) + 32)); //32 for padding
		// char *cur_d_cuda = gpu->d_data;


		d_start = cur_d_cuda;

		// segment data in device
		// float *d_data = (float *)(cur_d_cuda);
		// cur_d_cuda += 9 * sizeof(float) * element_num;
		// // space for the results in GPU
		// result_container *d_dist = (result_container *)(cur_d_cuda);
		// cur_d_cuda += sizeof(result_container) * pair_num;
		// // space for the offset and size information in GPU
		// size_t *d_os = (size_t *)(cur_d_cuda);
		// cur_d_cuda += 4 * sizeof(size_t) * pair_num;
		// // space for the hausdorff distances in GPU
		// float *d_hausdorff = (float *)cur_d_cuda;

		// cur_d_cuda += element_num * 2 * sizeof(float);
		// int *d_location = (int *)cur_d_cuda;

		size_t offset = 0;
		offset = align_up(offset, alignof(float));
		float *d_data = reinterpret_cast<float*>(cur_d_cuda + offset);
		offset += 9 * sizeof(float) * element_num;

		offset = align_up(offset, alignof(result_container));
		result_container *d_dist = reinterpret_cast<result_container*>(cur_d_cuda + offset);
		offset += sizeof(result_container) * pair_num;

		offset = align_up(offset, alignof(size_t));
		size_t *d_os = reinterpret_cast<size_t*>(cur_d_cuda + offset);
		offset += 4 * sizeof(size_t) * pair_num;

		offset = align_up(offset, alignof(float));
		float *d_hausdorff = reinterpret_cast<float*>(cur_d_cuda + offset);
		offset += 2 * sizeof(float) * element_num;

		offset = align_up(offset, alignof(int));
		int *d_location = reinterpret_cast<int*>(cur_d_cuda + offset);
		offset += sizeof(int) * 3 * pair_num;
	


		CUDA_SAFE_CALL(cudaMemcpy(d_data, data, element_num * 9 * sizeof(float), cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(d_os, offset_size, pair_num * 4 * sizeof(size_t), cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(d_hausdorff, hausdorff, element_num * 2 * sizeof(float), cudaMemcpyHostToDevice));
		// initialize results container array
		CUDA_SAFE_CALL(cudaMemcpy(d_dist, result, pair_num * sizeof(result_container), cudaMemcpyHostToDevice));

		

		//@@@@@@@ I need location
		// CUDA_SAFE_CALL(cudaMemcpy(d_location, location, pair_num * 3 * sizeof(int), cudaMemcpyHostToDevice));


		// int* d_prefix;
		// CUDA_SAFE_CALL(cudaMalloc(&d_prefix, (pair_num + 1) * sizeof(int)));
		// CUDA_SAFE_CALL(cudaMemcpy(d_prefix, h_prefix.data(), (pair_num + 1) * sizeof(int), cudaMemcpyHostToDevice));



		// logt("copying data to GPU", start);

		// initialize d_min_min_dist and d_min_max_dist

		// GPU_array_initialize<<<1, obj_pair_num>>>(gd.d_min_min_dist,
		// 										gd.d_min_max_dist,
		// 										obj_pair_num);

// #define MAX_DIM 512
		// compute the distance in parallel
		std::cout << "###### LOD = " << lod << std::endl;
		// for (int start = 0; start < max_tasks; start += 512)
		// {
		// 	int dim2 = min(max_tasks - start, 512);
		// 	TriDist_cuda_op_shm_flat_gpt<<<pair_num, dim2>>>(d_data, 
		// 													d_os, 
		// 													d_hausdorff, 
		// 													d_dist, 
		// 													start,
		// 													d_location,
		// 													gd.d_prefix_obj,
		// 													gd.d_out_min,
		// 													gd.d_out_max,
		// 													gd.d_min_min_dist,
		// 													gd.d_min_max_dist,
		// 													gd.d_valid_voxel_prefix,
		// 													gd.d_valid_voxel_pairs,
		// 													lod
		// 												);
		// 	check_execution();
		// }

		TriDist_cuda_op_shm_flat_fused<<<pair_num, 512>>>(d_data, 
														d_os, 
														d_hausdorff, 
														d_dist, 
														d_location,
														gd.d_prefix_obj,
														gd.d_out_min,
														gd.d_out_max,
														gd.d_min_min_dist,
														gd.d_min_max_dist,
														gd.d_valid_voxel_prefix,
														gd.d_valid_voxel_pairs,
														lod
													);


		// float *d_g_dd        = nullptr;
		// float *d_g_min_dist  = nullptr;
		// float *d_g_max_dist  = nullptr;

		// size_t reduction_size = pair_num  * sizeof(float);


		// cudaMalloc((void**)&d_g_dd,       reduction_size);
		// cudaMalloc((void**)&d_g_min_dist, reduction_size);
		// cudaMalloc((void**)&d_g_max_dist, reduction_size);

		// TriDist_cuda_op_atomic<<<pair_num, 512>>>(
		// 	d_data,
		// 	d_os,
		// 	d_hausdorff,
		// 	d_dist,
		// 	d_g_dd,
		// 	d_g_min_dist,
		// 	d_g_max_dist
		// );


		// check_execution();
		// CUDA_SAFE_CALL(cudaDeviceSynchronize());

		// cudaFree(d_g_dd);
		// cudaFree(d_g_min_dist);
		// cudaFree(d_g_max_dist);
		

		// std::cout << "after 1 kernel" << std::endl;


		// min_min_dist.resize(obj_pair_num);
		// CUDA_SAFE_CALL(cudaMemcpy(min_min_dist.data(),
		// 						  gd.d_min_min_dist,
		// 						  obj_pair_num * sizeof(float),
		// 						  cudaMemcpyDeviceToHost));
		// min_max_dist.resize(obj_pair_num);
		// CUDA_SAFE_CALL(cudaMemcpy(min_max_dist.data(),
		// 						  gd.d_min_max_dist,
		// 						  obj_pair_num * sizeof(float),
		// 						  cudaMemcpyDeviceToHost));


		// int *d_new_count;
		// CUDA_SAFE_CALL(cudaMalloc(&d_new_count, obj_pair_num * sizeof(int)));
		// CUDA_SAFE_CALL(cudaMemset(d_new_count, 0, obj_pair_num * sizeof(int)));

		// update_partial_voxel_level_distance<<<obj_pair_num, max_voxel_size>>>(
		// 													gd.d_status_after,
		// 													gd.d_prefix_obj,
		// 													gd.d_out_min,
		// 													gd.d_min_max_dist,
		// 													gd.d_valid_voxel_prefix,
		// 													gd.d_valid_voxel_pairs,
		// 													d_new_count
		// 												);
		// CUDA_SAFE_CALL(cudaDeviceSynchronize());
		// CUDA_SAFE_CALL(cudaFree(gd.d_count));
		// gd.d_count = d_new_count; // replace old with new count

		// std::cout << "after 2 kernel" << std::endl;



		// // =================== second kernel starts here ==================

		// vector<int> new_valid_voxel_cnt(obj_pair_num);
		// CUDA_SAFE_CALL(cudaMemcpy(new_valid_voxel_cnt.data(), gd.d_count, obj_pair_num * sizeof(int), cudaMemcpyDeviceToHost));

		// vector<int> new_valid_voxel_prefix(obj_pair_num + 1, 0);
		// for (int i = 0; i < obj_pair_num; ++i)
		// {
		// 	new_valid_voxel_prefix[i + 1] = new_valid_voxel_prefix[i] + new_valid_voxel_cnt[i];
		// }
		// int new_total_valid_voxel_pairs = new_valid_voxel_prefix[obj_pair_num];

		// int *d_new_valid_voxel_prefix;
		// int *d_new_valid_voxel_pairs;
		// CUDA_SAFE_CALL(cudaMalloc(&d_new_valid_voxel_prefix, (obj_pair_num + 1) * sizeof(int)));
		// CUDA_SAFE_CALL(cudaMemcpy(d_new_valid_voxel_prefix, new_valid_voxel_prefix.data(), (obj_pair_num + 1) * sizeof(int), cudaMemcpyHostToDevice));
		// CUDA_SAFE_CALL(cudaMalloc(&d_new_valid_voxel_pairs, 3 * new_total_valid_voxel_pairs * sizeof(int)));

		// record_valid_voxel_pairs_from_old<<<obj_pair_num, max_voxel_size>>>(
		// 	gd.d_status_after,
		// 	gd.d_valid_voxel_prefix,
		// 	gd.d_valid_voxel_pairs,
		// 	gd.d_prefix_obj,
		// 	gd.d_out_min,
		// 	gd.d_min_max_dist,
		// 	d_new_valid_voxel_prefix,
		// 	d_new_valid_voxel_pairs
		// );
		// CUDA_SAFE_CALL(cudaDeviceSynchronize());

		// CUDA_SAFE_CALL(cudaFree(gd.d_valid_voxel_prefix));
		// CUDA_SAFE_CALL(cudaFree(gd.d_valid_voxel_pairs));
		// gd.d_valid_voxel_prefix = d_new_valid_voxel_prefix;
		// gd.d_valid_voxel_pairs = d_new_valid_voxel_pairs;
		
		// valid_voxel_pairs.resize(3 * new_total_valid_voxel_pairs);
		// CUDA_SAFE_CALL(cudaMemcpy(valid_voxel_pairs.data(),
		// 						  gd.d_valid_voxel_pairs,
		// 						  3 * new_total_valid_voxel_pairs * sizeof(int),
		// 						  cudaMemcpyDeviceToHost));
		// valid_voxel_prefix = new_valid_voxel_prefix;


		// out_min.resize(total_voxel_size);
		// out_max.resize(total_voxel_size);
		// CUDA_SAFE_CALL(cudaMemcpy(out_min.data(), gd.d_out_min, total_voxel_size * sizeof(float), cudaMemcpyDeviceToHost));
		// CUDA_SAFE_CALL(cudaMemcpy(out_max.data(), gd.d_out_max, total_voxel_size * sizeof(float), cudaMemcpyDeviceToHost));

		// =================== third kernel starts here ==================

		// CUDA_SAFE_CALL(cudaFree(gd.d_confirmed_before));
		// gd.d_confirmed_before = gd.d_confirmed_after;
		// CUDA_SAFE_CALL(cudaMalloc(&gd.d_confirmed_after, N * sizeof(int)));

		// CUDA_SAFE_CALL(cudaFree(gd.d_status_before));
		// gd.d_status_before = gd.d_status_after;
		// CUDA_SAFE_CALL(cudaMalloc(&gd.d_status_after, obj_pair_num * sizeof(int)));
		// CUDA_SAFE_CALL(cudaMemset(gd.d_status_after, 0, obj_pair_num * sizeof(int)));

		// int warps = N;
		// int threads = 256;
		// int blocks = (warps * 32 + threads - 1) / threads;

		// // 3. launch kernel
		// mark_candidate_status_warp<<<blocks, threads>>>(
		// 	gd.d_min_min_dist,
		// 	gd.d_min_max_dist,
		// 	gd.d_prefix_tile,
		// 	gd.d_obj2,
		// 	gd.d_confirmed_before,
		// 	gd.d_confirmed_after,
		// 	K,
		// 	N,
		// 	gd.d_status_before,
		// 	gd.d_status_after);
		// CUDA_SAFE_CALL(cudaDeviceSynchronize());

		
		// confirmed_after.resize(N);
		// CUDA_SAFE_CALL(cudaMemcpy(confirmed_after.data(), gd.d_confirmed_after,
		// 						  N * sizeof(int), cudaMemcpyDeviceToHost));

		// status_after.resize(obj_pair_num);
		// CUDA_SAFE_CALL(cudaMemcpy(status_after.data(), gd.d_status_after,
		// 						  obj_pair_num * sizeof(int), cudaMemcpyDeviceToHost));


		CUDA_SAFE_CALL(cudaMemcpy(result, d_dist, pair_num * sizeof(result_container), cudaMemcpyDeviceToHost));
		CUDA_SAFE_CALL(cudaFree(d_start));
		// logt("copy data out", start);
	}


	void MeshDist_batch_gpu_op_shm_flat_stream(gpu_info *gpu, 
										const float *data, 
										const size_t *offset_size, 
										const float *hausdorff,
										result_container *result, 
										const uint32_t pair_num, 
										const size_t element_num,
										/** additional params */
										int *location,
										int obj_pair_num,
										size_t max_voxel_size,
										int N,
										int K,
										vector<int> &confirmed_after,
										vector<int> &status_after,
										vector<int> &valid_voxel_prefix,
										vector<int> &valid_voxel_pairs,
										int lod,
										size_t total_voxel_size,
										vector<float> &min_min_dist,
										vector<float> &min_max_dist,
										int stream_id
									)
	{

		std::cout << "###### LOD = " << lod << std::endl;


		struct timeval start = get_cur_time();
		assert(gpu);
		cudaSetDevice(gpu->device_id);
	
		// allocate memory in GPU
		char *cur_d_cuda, *d_start;
		CUDA_SAFE_CALL(cudaMalloc((void**)&cur_d_cuda, 9 * sizeof(float) * element_num 
											  +	sizeof(result_container) * pair_num
											  + 4 * sizeof(size_t) * pair_num 
											  + element_num * 2 * sizeof(float) + 32)); // 32 is for padding
		// char *cur_d_cuda = gpu->d_data;


		size_t gpu_memory_occ = 9 * sizeof(float) * (size_t)element_num 
											  +	sizeof(result_container) * pair_num
											  + 4 * sizeof(size_t) * pair_num 
											  + element_num * 2 * sizeof(float);


		std::cout << "GPU memory at refinement = " << gpu_memory_occ << std::endl;

		d_start = cur_d_cuda;

		// segment data in device
		// float *d_data = (float *)(cur_d_cuda);
		// cur_d_cuda += 9 * sizeof(float) * element_num;
		// // space for the results in GPU
		// result_container *d_dist = (result_container *)(cur_d_cuda);
		// cur_d_cuda += sizeof(result_container) * pair_num;
		// // space for the offset and size information in GPU
		// size_t *d_os = (size_t *)(cur_d_cuda);
		// cur_d_cuda += 4 * sizeof(size_t) * pair_num;
		// // space for the hausdorff distances in GPU
		// float *d_hausdorff = (float *)cur_d_cuda;

		// cur_d_cuda += element_num * 2 * sizeof(float);
		// int *d_location = (int *)cur_d_cuda;



		size_t offset = 0;
		offset = align_up(offset, alignof(float));
		float *d_data = reinterpret_cast<float*>(cur_d_cuda + offset);
		offset += 9 * sizeof(float) * element_num;

		offset = align_up(offset, alignof(result_container));
		result_container *d_dist = reinterpret_cast<result_container*>(cur_d_cuda + offset);
		offset += sizeof(result_container) * pair_num;

		offset = align_up(offset, alignof(size_t));
		size_t *d_os = reinterpret_cast<size_t*>(cur_d_cuda + offset);
		offset += 4 * sizeof(size_t) * pair_num;

		offset = align_up(offset, alignof(float));
		float *d_hausdorff = reinterpret_cast<float*>(cur_d_cuda + offset);
		offset += 2 * sizeof(float) * element_num;

		offset = align_up(offset, alignof(int));
		int *d_location = reinterpret_cast<int*>(cur_d_cuda + offset);
		offset += sizeof(int) * 3 * pair_num;
	
		// pinned memory to facilitate async memcpy
		CUDA_SAFE_CALL(cudaMemcpyAsync(d_data, data, element_num * 9 * sizeof(float), cudaMemcpyHostToDevice, refine_streams[stream_id]));
		CUDA_SAFE_CALL(cudaMemcpyAsync(d_os, offset_size, pair_num * 4 * sizeof(size_t), cudaMemcpyHostToDevice, refine_streams[stream_id]));
		CUDA_SAFE_CALL(cudaMemcpyAsync(d_hausdorff, hausdorff, element_num * 2 * sizeof(float), cudaMemcpyHostToDevice, refine_streams[stream_id]));
		// initialize results container array
		CUDA_SAFE_CALL(cudaMemcpyAsync(d_dist, result, pair_num * sizeof(result_container), cudaMemcpyHostToDevice, refine_streams[stream_id]));

		// record this event
		cudaEventRecord(memcpy_done[stream_id], refine_streams[stream_id]);
	

		TriDist_cuda_op_shm_flat_fused<<<pair_num, 512, 0, refine_streams[stream_id]>>>(d_data, 
														d_os, 
														d_hausdorff, 
														d_dist, 
														d_location,
														gd.d_prefix_obj,
														gd.d_out_min,
														gd.d_out_max,
														gd.d_min_min_dist,
														gd.d_min_max_dist,
														gd.d_valid_voxel_prefix,
														gd.d_valid_voxel_pairs,
														lod
													);


		CUDA_SAFE_CALL(cudaMemcpyAsync(result, d_dist, pair_num * sizeof(result_container), cudaMemcpyDeviceToHost, refine_streams[stream_id]));
		CUDA_SAFE_CALL(cudaFreeAsync(d_start, refine_streams[stream_id]));
	}

	void update_status_and_confirm(vector<int> &new_confirmed_after,
									vector<int> &new_stauts_after,
									int N,
									int obj_pair_num
								)
	{
		CUDA_SAFE_CALL(cudaMemcpy(gd.d_confirmed_after, new_confirmed_after.data(),
								  N * sizeof(int), cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(gd.d_status_after, new_stauts_after.data(),
								  obj_pair_num * sizeof(int), cudaMemcpyHostToDevice));
	}
					

	void MeshDist_batch_gpu_op_warp_level(gpu_info *gpu, const float *data, const uint32_t *offset_size, const float *hausdorff,
										  result_container *result, const uint32_t pair_num, const uint32_t element_num)
	{
		struct timeval start = get_cur_time();
		assert(gpu);
		cudaSetDevice(gpu->device_id);
		// profile the input data
		uint32_t max_size_1 = 0;
		uint32_t max_size_2 = 0;
		for (int i = 0; i < pair_num; i++)
		{
			if (offset_size[i * 4 + 1] > max_size_1)
			{
				max_size_1 = offset_size[i * 4 + 1];
			}
			if (offset_size[i * 4 + 3] > max_size_2)
			{
				max_size_2 = offset_size[i * 4 + 3];
			}
		}
		// allocate memory in GPU
		char *cur_d_cuda = gpu->d_data;

		// segment data in device
		float *d_data = (float *)(cur_d_cuda);
		cur_d_cuda += 9 * sizeof(float) * element_num;
		// space for the results in GPU
		result_container *d_dist = (result_container *)(cur_d_cuda);
		cur_d_cuda += sizeof(result_container) * pair_num;
		// space for the offset and size information in GPU
		uint32_t *d_os = (uint32_t *)(cur_d_cuda);
		cur_d_cuda += 4 * sizeof(uint32_t) * pair_num;
		// space for the hausdorff distances in GPU
		float *d_hausdorff = (float *)cur_d_cuda;

		CUDA_SAFE_CALL(cudaMemcpy(d_data, data, element_num * 9 * sizeof(float), cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(d_os, offset_size, pair_num * 4 * sizeof(uint32_t), cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(d_hausdorff, hausdorff, element_num * 2 * sizeof(float), cudaMemcpyHostToDevice));
		// initialize results container array
		CUDA_SAFE_CALL(cudaMemcpy(d_dist, result, pair_num * sizeof(result_container), cudaMemcpyHostToDevice));

		// logt("copying data to GPU", start);

#define MAX_DIM 1024
		// compute the distance in parallel
		// for(uint32_t tri_offset_2=0;tri_offset_2<max_size_2;tri_offset_2+= MAX_DIM){
		// 	uint32_t dim2 = min(max_size_2-tri_offset_2, (uint32_t)MAX_DIM);
		// 	for(uint32_t tri_offset_1=0;tri_offset_1<max_size_1;tri_offset_1++){
		// 		TriDist_cuda<<<pair_num, dim2>>>(d_data, d_os, d_hausdorff, d_dist, tri_offset_1, tri_offset_2);
		// 		check_execution();
		// 	}
		// 	//cout<<pair_num<<" "<< tri_offset_2 <<" "<<dim2<<" "<< max_size_1 <<" "<< max_size_2 << endl;
		// }

		int threadsPerBlock = 128;
		int WARP_PER_BLOCK = threadsPerBlock / WARP_SIZE;
		int numBlocks = (pair_num + WARP_PER_BLOCK - 1) / WARP_PER_BLOCK;

		for (uint32_t tri_offset_2 = 0; tri_offset_2 < max_size_2; tri_offset_2 += 32)
		{
			TriDist_cuda_op_warp_level<<<numBlocks, threadsPerBlock>>>(d_data, d_os, d_hausdorff, d_dist, max_size_1, tri_offset_2);
			check_execution();
		}
		cudaDeviceSynchronize();
		// logt("distances computations", start);

		CUDA_SAFE_CALL(cudaMemcpy(result, d_dist, pair_num * sizeof(result_container), cudaMemcpyDeviceToHost));
		// logt("copy data out", start);
	}

	void MeshDist_batch_gpu_op_flat(gpu_info *gpu, const float *data, const uint32_t *offset_size, const float *hausdorff,
									result_container *result, const uint32_t pair_num, const uint32_t element_num)
	{
		struct timeval start = get_cur_time();
		assert(gpu);
		cudaSetDevice(gpu->device_id);
		// profile the input data
		uint32_t max_size_1 = 0;
		uint32_t max_size_2 = 0;
		for (int i = 0; i < pair_num; i++)
		{
			if (offset_size[i * 4 + 1] > max_size_1)
			{
				max_size_1 = offset_size[i * 4 + 1];
			}
			if (offset_size[i * 4 + 3] > max_size_2)
			{
				max_size_2 = offset_size[i * 4 + 3];
			}
		}
		// allocate memory in GPU
		char *cur_d_cuda = gpu->d_data;

		// segment data in device
		float *d_data = (float *)(cur_d_cuda);
		cur_d_cuda += 9 * sizeof(float) * element_num;
		// space for the results in GPU
		result_container *d_dist = (result_container *)(cur_d_cuda);
		cur_d_cuda += sizeof(result_container) * pair_num;
		// space for the offset and size information in GPU
		uint32_t *d_os = (uint32_t *)(cur_d_cuda);
		cur_d_cuda += 4 * sizeof(uint32_t) * pair_num;
		// space for the hausdorff distances in GPU
		float *d_hausdorff = (float *)cur_d_cuda;

		CUDA_SAFE_CALL(cudaMemcpy(d_data, data, element_num * 9 * sizeof(float), cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(d_os, offset_size, pair_num * 4 * sizeof(uint32_t), cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(d_hausdorff, hausdorff, element_num * 2 * sizeof(float), cudaMemcpyHostToDevice));
		// initialize results container array
		CUDA_SAFE_CALL(cudaMemcpy(d_dist, result, pair_num * sizeof(result_container), cudaMemcpyHostToDevice));

		uint32_t *prefix_num = new uint32_t[pair_num + 1];
		prefix_num[0] = 0;
		uint32_t *d_prefix_num;
		CUDA_SAFE_CALL(cudaMalloc((void **)&d_prefix_num, (pair_num + 1) * sizeof(uint32_t)));

		// logt("copying data to GPU", start);

#define MAX_DIM 1024
		for (uint32_t tri_offset_start_2 = 0; tri_offset_start_2 < max_size_2; tri_offset_start_2 += MAX_DIM)
		{
			uint32_t dim2 = min(max_size_2 - tri_offset_start_2, (uint32_t)MAX_DIM);
			// TriDist_cuda_op<<<pair_num, dim2>>>(d_data, d_os, d_hausdorff, d_dist, max_size_1, tri_offset_start_2);

			for (int i = 0; i < pair_num; ++i)
			{
				// std::cout << offset_size[i*4+1] << ", " <<  offset_size[i*4+3] << std::endl;
				prefix_num[i + 1] = prefix_num[i] + offset_size[i * 4 + 1] * offset_size[i * 4 + 3];
			}

			CUDA_SAFE_CALL(cudaMemcpy(d_prefix_num, prefix_num, (pair_num + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice));

			uint32_t total_tasks = prefix_num[pair_num];

			int threadsPerBlock = 128;
			int numBlocks = (total_tasks + threadsPerBlock - 1) / threadsPerBlock;

			// TriDist_cuda_op<<<pair_num, dim2>>>(d_data, d_os, d_hausdorff, d_dist, max_size_1, tri_offset_start_2);
			TriDist_cuda_op_balanced_wkld<<<numBlocks, threadsPerBlock>>>(d_data, d_os, d_hausdorff, d_dist, d_prefix_num, pair_num);
			check_execution();
		}
		cudaDeviceSynchronize();
		// logt("distances computations", start);

		CUDA_SAFE_CALL(cudaMemcpy(result, d_dist, pair_num * sizeof(result_container), cudaMemcpyDeviceToHost));

		delete[] prefix_num;
	}

	__global__ void clear_resultset(result_container *result, uint32_t pairnum)
	{

		int id = blockIdx.x * blockDim.x + threadIdx.x;
		if (id >= pairnum)
		{
			return;
		}
		result[id].intersected = 0;
		result[id].min_dist = DBL_MAX;
		result[id].max_dist = DBL_MAX;
	}

	void TriInt_batch_gpu(gpu_info *gpu, const float *data, const size_t *offset_size, const float *hausdorff,
						  result_container *intersection, const uint32_t pair_num, const uint32_t triangle_num)
	{

		assert(gpu);
		cudaSetDevice(gpu->device_id);
		struct timeval start = get_cur_time();
		// allocate memory in GPU
		char *cur_d_cuda = gpu->d_data;

		// profile the input data
		uint32_t max_size_1 = 0;
		uint32_t max_size_2 = 0;
		for (int i = 0; i < pair_num; i++)
		{
			if (offset_size[i * 4 + 1] > max_size_1)
			{
				max_size_1 = offset_size[i * 4 + 1];
			}
			if (offset_size[i * 4 + 3] > max_size_2)
			{
				max_size_2 = offset_size[i * 4 + 3];
			}
		}

		// log("%ld %ld", max_size_1, max_size_2);
		//  segment data in device
		float *d_data = (float *)(cur_d_cuda);
		cur_d_cuda += 9 * triangle_num * sizeof(float);
		float *d_hausdorff = NULL;
		if (hausdorff)
		{
			d_hausdorff = (float *)(cur_d_cuda);
			cur_d_cuda += 2 * triangle_num * sizeof(float);
			CUDA_SAFE_CALL(cudaMemcpy(d_hausdorff, hausdorff, triangle_num * 2 * sizeof(float), cudaMemcpyHostToDevice));
		}

		// space for the results in GPU
		result_container *d_intersect = (result_container *)(cur_d_cuda);
		cur_d_cuda += sizeof(result_container) * pair_num;
		// space for the offset and size information in GPU
		uint32_t *d_os = (uint32_t *)(cur_d_cuda);

		CUDA_SAFE_CALL(cudaMemcpy(d_data, data, triangle_num * 9 * sizeof(float), cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(d_os, offset_size, pair_num * 4 * sizeof(uint32_t), cudaMemcpyHostToDevice));
		// logt("copying data to GPU", start);

		clear_resultset<<<pair_num / 1024 + 1, 1024>>>(d_intersect, pair_num);

		// check the intersection
		for (uint32_t cur_offset_2 = 0; cur_offset_2 < max_size_2; cur_offset_2 += 1024)
		{
			uint32_t dim2 = min(max_size_2 - cur_offset_2, (uint32_t)1024);
			for (uint32_t cur_offset_1 = 0; cur_offset_1 < max_size_1; cur_offset_1++)
			{
				TriInt_cuda<<<pair_num, dim2>>>(d_data, d_os, d_hausdorff, d_intersect, cur_offset_1, cur_offset_2);
				check_execution();
			}
		}
		check_execution();

		// cout<<pair_num<<" "<<triangle_num<<" "<<sizeof(uint32_t)<<endl;
		cudaDeviceSynchronize();
		// logt("distances computations", start);

		CUDA_SAFE_CALL(cudaMemcpy(intersection, d_intersect, pair_num * sizeof(result_container), cudaMemcpyDeviceToHost));
		// logt("copy data out", start);
	}


	void evaluation_kernel(int total_pairs,
							int target_count,
							vector<float> &h_min,
							vector<float> &h_max,
							vector<int> &h_prefix_tile,
							vector<int> &h_confirmed,
							int K,
							vector<int> &h_status
						) 
	{
		float *d_min = NULL, *d_max = NULL;
		int *d_prefix, *d_status, *d_confirmed;

		CUDA_SAFE_CALL(cudaMalloc(&d_min, total_pairs * sizeof(float)));
		CUDA_SAFE_CALL(cudaMalloc(&d_max, total_pairs * sizeof(float)));
		CUDA_SAFE_CALL(cudaMalloc(&d_prefix, (target_count+1) * sizeof(int)));
		CUDA_SAFE_CALL(cudaMalloc(&d_status, total_pairs * sizeof(int)));
		CUDA_SAFE_CALL(cudaMalloc(&d_confirmed, target_count * sizeof(int)));

		CUDA_SAFE_CALL(cudaMemcpy(d_min, h_min.data(), total_pairs * sizeof(float), cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(d_max, h_max.data(), total_pairs * sizeof(float), cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(d_prefix, h_prefix_tile.data(), (target_count+1) * sizeof(int), cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(d_confirmed, h_confirmed.data(), target_count * sizeof(int), cudaMemcpyHostToDevice));

		int threads = 256;
		evaluate_knn_kernel<<<target_count, threads>>>(
			d_min, 
			d_max,
			d_prefix,
			d_confirmed,
			d_status,
			target_count,
			K
		);
		check_execution();
		CUDA_SAFE_CALL(cudaDeviceSynchronize());

		h_status.resize(total_pairs);
		cudaMemcpy(h_status.data(), d_status, total_pairs*sizeof(int), cudaMemcpyDeviceToHost);

		CUDA_SAFE_CALL(cudaFree(d_min));
		CUDA_SAFE_CALL(cudaFree(d_max));
		CUDA_SAFE_CALL(cudaFree(d_prefix));
		CUDA_SAFE_CALL(cudaFree(d_status));
		CUDA_SAFE_CALL(cudaFree(d_confirmed));
	}
}
