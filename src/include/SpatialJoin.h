/*
 * SpatialJoin.h
 *
 *  Created on: Nov 11, 2019
 *      Author: teng
 */

#ifndef SPATIALJOIN_H_
#define SPATIALJOIN_H_

#include <queue>
#include <tuple>
#include <condition_variable>


#include "query_context.h"
#include "aab.h"
#include "tile.h"
#include "geometry.h"
#include "candidate.h"
#include "himesh.h"
using namespace std;

namespace tdbase{

// type of the workers, GPU or CPU
// each worker took a batch of jobs (X*Y) from the job queue
// and conduct the join, the result is then stored to
// the target result addresses
enum Worker_Type{
	WT_GPU,
	WT_CPU
};

enum Join_Type{
	JT_intersect,
	JT_distance,
	JT_nearest
};

// size of the buffer is 1GB
const static long VOXEL_BUFFER_SIZE = 1<<30;

/* todo: need be updated
 * all computations will be aligned into computation units
 * of N*N. For instance, after checking the index, the
 * nearest neighbor of object a is b or c is not decided.
 * We further decode the polyhedron a, b and c if they
 * are not decoded yet. The edges and surfaces are
 * decoded and cached. Then the computation across those
 * segments and triangles are organized as many N*N
 * computing units as possible. Notice that padding may be needed to
 * align the computing units. then space in buffer is claimed
 * to store the data of those computing units. the true computation
 * is done by the GPU or CPU, and results will be copied into the result_addr
 * Corresponding to the buffer space claimed.
*/


struct Additional_Params {
	int obj_pair_num;
	size_t max_voxel_size; 
	int N;
	int K;
	size_t total_voxel_size;

	size_t chunk_start;
};

struct Chunk {
	vector<candidate_entry *> candidates;
	size_t pair_count = 0;
};

class SpatialJoin{
	geometry_computer *computer = NULL;
public:

	/**
	 *  New Data structures
	 */

	Tile *tile1;
	Tile *tile2;

	size_t tile1_size;
	size_t tile2_size;

	vector<int> prefix_tile;
	vector<size_t> prefix_obj;

	vector<int> compute_obj_1, compute_obj_2;

	vector<float> out_min;
	vector<float> out_max;

	vector<float> all_min_max_dist;
	vector<float> all_min_min_dist;
	vector<int> valid_vp_cnt;
	vector<int> valid_voxel_pairs;
	vector<float> valid_voxel_pairs_dist;
	vector<size_t> valid_voxel_prefix;
	vector<int> confirmed_after;
	vector<int> status_after;



	SpatialJoin(geometry_computer *c);
	~SpatialJoin();
	/*
	 *
	 * the main entry function to conduct next round of computation
	 * each object in tile1 need to compare with all objects in tile2.
	 * to avoid unnecessary computation, we need to build index on tile2.
	 * The unit for building the index for tile2 is the ABB for all or part
	 * of the surface (mostly triangle) of a polyhedron.
	 *
	 * */
	tuple<vector<int>, vector<int>, vector<float>> prepare_data(size_t tile_size, Tile *tile);

	vector<candidate_entry *> mbb_knn(Tile *tile1, Tile *tile2, query_context &ctx);
	vector<candidate_entry *> mbb_knn_op_gpu(Tile *tile1, Tile *tile2, query_context &ctx);
	vector<candidate_entry *> mbb_knn_op_gpu_one_kernel(Tile *tile1, Tile *tile2, query_context &ctx);
	vector<candidate_entry *> mbb_knn_op_gpu_one_kernel_disk(Tile *tile1, Tile *tile2, query_context &ctx);

	vector<candidate_entry *> mbb_within_op_gpu_one_kernel(Tile *tile1, Tile *tile2, query_context &ctx);
	vector<candidate_entry *> mbb_within(Tile *tile1, Tile *tile2, query_context &ctx);
	vector<candidate_entry *> mbb_intersect(Tile *tile1, Tile *tile2);

	range update_voxel_pair_list(vector<voxel_pair> &voxel_pairs, double minmaxdist);

	void decode_data(vector<candidate_entry *> &candidates, query_context &ctx);

	geometry_param packing_data(vector<candidate_entry *> &candidates, query_context &ctx, Additional_Params *ap = nullptr);
	geometry_param packing_data_stream(vector<candidate_entry *> &candidates, query_context &ctx, Additional_Params *ap = nullptr,
										int stream_id = 0);
	geometry_param calculate_distance(vector<candidate_entry *> &candidates, query_context &ctx, Additional_Params *ap = nullptr);
	geometry_param calculate_distance_stream(vector<candidate_entry *> &candidates, query_context &ctx, geometry_param &gp, 
												Additional_Params *ap = nullptr,
												int stream_id = 0);
	void check_intersection(vector<candidate_entry *> &candidates, query_context &ctx);

	vector<Chunk> build_chunks_by_pair(
        vector<candidate_entry *> &candidates,
        size_t target_pairs_per_chunk,
		size_t target_tris_per_chunk
	);

	void nearest_neighbor(query_context ctx);
	void nearest_neighbor_chunking(query_context ctx);
	void within(query_context ctx);
	void within_chunking(query_context ctx);

	void refine_aggregation(vector<candidate_entry *> &candidates, 
							query_context &ctx,
							int prev
						);
	void intersect(query_context ctx);

	void join(vector<pair<Tile *, Tile *>> &tile_pairs);

	/*
	 *
	 * go check the index
	 *
	 * */
	void check_index();

	// register job to gpu
	// worker can register work to gpu
	// if the GPU is idle and queue is not full
	// otherwise do it locally with CPU
	float *register_computation(char *data, int num_cu);

	/*
	 * do the geometry computation in a batch with GPU
	 *
	 * */
	void compute_gpu();

	/*
	 * do the geometry computation in a batch with CPU
	 *
	 * */
	void compute_cpu();


};

}


#endif /* SPATIALJOIN_H_ */
