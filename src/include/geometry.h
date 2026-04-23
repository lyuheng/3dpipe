#ifndef HISPEED_GEOMETRY_H
#define HISPEED_GEOMETRY_H

#include <math.h>
#include <stdio.h>
#include <iostream>
#include <float.h>
#include <condition_variable>

#include "mygpu.h"
#include "util.h"
#include "pthread.h"
#include "aab.h"
using namespace std;

namespace tdbase
{

	// useful functions

	void * allocate_pinned_memory(size_t size);
	void free_pinned_memory(void *ptr);

	// copy
	inline void
	VcV(float Vr[3], const float V[3])
	{
		Vr[0] = V[0];
		Vr[1] = V[1];
		Vr[2] = V[2];
	}

	// minus
	inline void
	VmV(float Vr[3], const float V1[3], const float V2[3])
	{
		Vr[0] = V1[0] - V2[0];
		Vr[1] = V1[1] - V2[1];
		Vr[2] = V1[2] - V2[2];
	}

	// plus
	inline void
	VpV(float Vr[3], const float V1[3], const float V2[3])
	{
		Vr[0] = V1[0] + V2[0];
		Vr[1] = V1[1] + V2[1];
		Vr[2] = V1[2] + V2[2];
	}

	// plus after product
	inline void
	VpVxS(float Vr[3], const float V1[3], const float V2[3], float s)
	{
		Vr[0] = V1[0] + V2[0] * s;
		Vr[1] = V1[1] + V2[1] * s;
		Vr[2] = V1[2] + V2[2] * s;
	}

	inline void
	VcrossV(float Vr[3], const float V1[3], const float V2[3])
	{
		Vr[0] = V1[1] * V2[2] - V1[2] * V2[1];
		Vr[1] = V1[2] * V2[0] - V1[0] * V2[2];
		Vr[2] = V1[0] * V2[1] - V1[1] * V2[0];
	}

	// dot product
	inline float
	VdotV(const float V1[3], const float V2[3])
	{
		return (V1[0] * V2[0] + V1[1] * V2[1] + V1[2] * V2[2]);
	}

	// Euclid distance
	inline float
	VdistV2(const float V1[3], const float V2[3])
	{
		return ((V1[0] - V2[0]) * (V1[0] - V2[0]) +
				(V1[1] - V2[1]) * (V1[1] - V2[1]) +
				(V1[2] - V2[2]) * (V1[2] - V2[2]));
	}

	// multiple each value in V with constant s
	inline void
	VxS(float Vr[3], const float V[3], float s)
	{
		Vr[0] = V[0] * s;
		Vr[1] = V[1] * s;
		Vr[2] = V[2] * s;
	}

	inline void
	VdS(float Vr[3], const float V[3], float s)
	{
		assert(s > 0);
		Vr[0] = V[0] / s;
		Vr[1] = V[1] / s;
		Vr[2] = V[2] / s;
	}

	class result_container
	{
	public:
		uint32_t p1;
		uint32_t p2;
		bool intersected;
		float distance = FLT_MAX; // for normal distance
		float min_dist = FLT_MAX; // for distance range
		float max_dist = FLT_MAX;
		void print()
		{
			cout << "p1:\t" << p1 << endl;
			cout << "p2:\t" << p2 << endl;
			cout << "intersected:\t" << intersected << endl;
			cout << "distance:\t" << distance << endl;
			cout << "min_dist:\t" << min_dist << endl;
			cout << "max_dist:\t" << max_dist << endl;
		}
	};

	class geometry_param
	{
	public:
		int id = 0;
		uint32_t pair_num = 0;
		size_t element_num = 0;
		size_t element_pair_num = 0;
		float *data = NULL;
		float *hausdorff = NULL;
		// the offset and size of the computing pairs
		size_t *offset_size = NULL;
		result_container *results = NULL;

		//@@ Additional params
		int *location = NULL;
		int obj_pair_num;
		size_t max_voxel_size; 
		int N;
		int K;
		vector<int> confirmed_after;
		vector<int> status_after;

		vector<int> valid_voxel_prefix;
		vector<int> valid_voxel_pairs;

		int lod;
		size_t total_voxel_size;

		vector<float> min_min_dist;
		vector<float> min_max_dist;

		size_t chunk_start;
		int stream_id;

		bool allocated = false;

		void allocate_buffer()
		{
			data = new float[9 * element_num];
			hausdorff = new float[2 * element_num];
			offset_size = new size_t[4 * pair_num];

			location = new int[3 * pair_num];
		}

		void allocate_buffer_pinned()
		{
			
			data = (float *)allocate_pinned_memory(9 * element_num * 2 * sizeof(float));
			hausdorff = (float *)allocate_pinned_memory(2 * element_num * 2 * sizeof(float));
			offset_size = (size_t *)allocate_pinned_memory(4 * pair_num * 2 * sizeof(size_t));

			location = (int *)allocate_pinned_memory(3 * pair_num * 2 * sizeof(int));	
	
			std::cout <<  "element_num = " << element_num << std::endl; 
		}


		void clear_buffer()
		{
			if (data)
			{
				delete[] data;
			}
			if (hausdorff)
			{
				delete[] hausdorff;
			}
			if (offset_size)
			{
				delete[] offset_size;
			}

			if (location) 
			{
				delete[] location;
			}
		}

		void clear_pinned_memory() {
			free_pinned_memory(data);
			free_pinned_memory(hausdorff);
			free_pinned_memory(offset_size);
			free_pinned_memory(location);
		}
	};

	inline float distance(const float *p1, const float *p2)
	{
		float cur_dist = 0;
		for (int t = 0; t < 3; t++)
		{
			cur_dist += (p1[t] - p2[t]) * (p1[t] - p2[t]);
		}
		return cur_dist;
	}

	void compute_normal(float *Norm, const float *triangle);
	bool PointInTriangleCylinder(const float *point, const float *triangle);
	void project_points_to_triangle_plane(const float *point, const float *triangle, float projected_point[3]);
	float PointTriangleDist(const float *point, const float *triangle);
	float TriDist(const float *S, const float *T);
	result_container MeshDist(const float *data1, const float *data2, size_t size1, size_t size2, const float *hausdorff1 = NULL, const float *hausdorff2 = NULL);
	void MeshDist_batch_gpu(gpu_info *gpu, const float *data, const size_t *offset_size, const float *hausdorff, result_container *result, const uint32_t pair_num, const uint32_t element_num);
	void MeshDist_batch_gpu_op_shm(gpu_info *gpu, const float *data, const uint32_t *offset_size, const float *hausdorff, result_container *result, const uint32_t pair_num, const uint32_t element_num);

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
									);
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
									);

	void update_status_and_confirm(vector<int> &new_confirmed_after,
									vector<int> &new_stauts_after,
									int N,
									int obj_pair_num
								);

	void MeshDist_batch_gpu_op_flat(gpu_info *gpu, const float *data, const uint32_t *offset_size, const float *hausdorff, result_container *result, const uint32_t pair_num, const uint32_t element_num);
	void MeshDist_batch_gpu_op_warp_level(gpu_info *gpu, const float *data, const uint32_t *offset_size, const float *hausdorff, result_container *result, const uint32_t pair_num, const uint32_t element_num);
	//================================================

	void allocateVoxels(vector<Voxel *> &voxels, float* &d_wrapper_voxels);
	// vector<float *> allocateVoxelsForAll(size_t tile_size, Tile *tile);
	
	void compute_voxel_pair_distance(float *v1, int len_v1, 
									float *v2, int len_v2,
									vector<float> &out_min, vector<float> &out_max);


	// ALL GPU's Data
	struct GD {
		int *d_obj1, *d_obj2;
		size_t *d_prefix_obj;
		int *d_prefix_tile;
		float *d_out_min, *d_out_max;
		float *d_min_max_dist, *d_min_min_dist;
		int *d_count;

		size_t *d_valid_voxel_prefix;
		int *d_valid_voxel_pairs;

		int *d_confirmed_before, *d_confirmed_after;
		int *d_status_before, *d_status_after;
	};

	struct DeviceVoxels 
	{
		float *d_all_voxels;     
		int *d_voxel_offset; 
		int *d_voxel_count;
		vector<int> h_voxel_offset;
		vector<int> h_voxel_count;
		int tile_size;
	};

	enum
	{
		UNDECIDED = 0,
		CONFIRMED = 1,
		REMOVED = 2,
		REPORTED = 3
	};

	// struct used in disk writing
	struct Buffer {
		int* pairs;
		float* dist;
		size_t pair_count;
		bool ready = false;
	};



	DeviceVoxels allocateVoxelsForAll(size_t tile_size, 
									vector<int> &voxel_offset, 
									vector<int> &voxel_count, 
									vector<float> &all_voxels);
	void compute_voxel_pair_distance(
			int v1,
			vector<pair<int, range>> &candidates,
			const DeviceVoxels &dv1,
			const DeviceVoxels &dv2,
			vector<float> &h_min,
			vector<float> &h_max);

	// ONLY used for k-NN query
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
										vector<int> &status_after);

	// ONLY used for k-NN query, with streaming
	void compute_voxel_pair_distance_for_all_streaming(const DeviceVoxels &dv1,
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
										vector<int> &status_after);
	
	void compute_voxel_pair_distance_for_all_streaming_pipeline(const DeviceVoxels &dv1,
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
										vector<int> &status_after);

	void compute_voxel_pair_distance_for_all_streaming_pipeline_reduceIO(const DeviceVoxels &dv1,
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
										vector<int> &status_after);

	void compute_voxel_pair_distance_for_all_streaming_pipeline_disk(const DeviceVoxels &dv1,
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
										vector<int> &status_after);

	void freeHost(int *valid_voxel_pairs,
				float *valid_voxel_pairs_dist);
	// ONLY used for within query
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
											);

	// ONLY used for within query, with streaming
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
											);

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
											);

	void free_cuda_memory(void *ptr);

	void evaluation_kernel(int total_pairs,
							int target_count,
							vector<float> &h_min,
							vector<float> &h_max,
							vector<int> &h_prefix_tile,
							vector<int> &h_confirmed,
							int K,
							vector<int> &h_status);
	
	void initialize_refine_stream();

	void * allocate_pinned_memory(size_t size);
	void free_pinned_memory(void *ptr);
	void synchronize_gpu();
	void synchronize_stream(int stream_id);
	void synchronize_event(int stream_id);

	//================================================

	bool TriInt(const float *S, const float *T);
	result_container MeshInt(const float *data1, const float *data2, size_t size1, size_t size2, const float *hausdorff1 = NULL, const float *hausdorff2 = NULL);
	void TriInt_batch_gpu(gpu_info *gpu, const float *data, const size_t *offset_size, const float *hausdorff, result_container *result, const uint32_t batch_num, const uint32_t triangle_num);

	class geometry_computer
	{
		pthread_mutex_t gpu_lock;
		pthread_mutex_t cpu_lock;
		int max_thread_num = tdbase::get_num_threads();
		bool cpu_busy = false;
		bool gpu_busy = false;
		bool request_cpu();
		void release_cpu();
		gpu_info *request_gpu(int min_size, bool force = false);
		void release_gpu(gpu_info *info);

		char *d_cuda = NULL;
		vector<gpu_info *> gpus;

	public:
		~geometry_computer();
		geometry_computer()
		{
			pthread_mutex_init(&cpu_lock, NULL);
			pthread_mutex_init(&gpu_lock, NULL);
		}

		bool init_gpus();
		void get_distance_gpu(geometry_param &param);
		void get_distance_gpu_stream(geometry_param &param);
		void get_distance_cpu(geometry_param &param);
		void get_distance(geometry_param &param);

		void get_intersect_gpu(geometry_param &param);
		void get_intersect_cpu(geometry_param &param);
		void get_intersect(geometry_param &param);
		void set_thread_num(uint32_t num)
		{
			max_thread_num = num;
		}

		void initialize_refine_stream();
		
	};

}
#endif
