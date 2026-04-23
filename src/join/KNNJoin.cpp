/*
 * KNNJoin.cpp
 *
 *  Created on: Sep 16, 2022
 *      Author: teng
 */

#include "SpatialJoin.h"

namespace tdbase{

Additional_Params addParams;

inline float get_min_max_dist(vector<voxel_pair> &voxel_pairs){
	float minmaxdist = DBL_MAX;
	for(voxel_pair &p:voxel_pairs){
		minmaxdist = min(minmaxdist, p.dist.maxdist);
	}
	return minmaxdist;
}

inline range update_voxel_pair_list(vector<voxel_pair> &voxel_pairs, double minmaxdist){

	range ret;
	ret.mindist = DBL_MAX;
	ret.maxdist = minmaxdist;
	// some voxel pair is farther than this one
	for(auto vp_iter = voxel_pairs.begin();vp_iter!=voxel_pairs.end();){
		// a closer voxel pair already exist
		if(vp_iter->dist.mindist > minmaxdist){
			// evict this unqualified voxel pairs
			voxel_pairs.erase(vp_iter);
		}else{
			ret.mindist = min(ret.mindist, vp_iter->dist.mindist);
			vp_iter++;
		}
	}
	return ret;
}

void print_candidate(candidate_entry *cand){
	if(global_ctx.verbose>=1){
		log("%ld (%d + %ld)", cand->mesh_wrapper->id, cand->candidate_confirmed, cand->candidates.size());
		int i=0;
		for(candidate_info *ci:cand->candidates){
			log("%d\t%ld:\t[%f,%f]", i++, ci->mesh_wrapper->id, ci->distance.mindist, ci->distance.maxdist);
		}
	}
}

inline void update_candidate_list_knn(candidate_entry *cand, query_context &ctx){
	HiMesh_Wrapper *target = cand->mesh_wrapper;
	int list_size = cand->candidates.size();
	for(int i=0;i<list_size && ctx.knn>cand->candidate_confirmed;){
		int sure_closer = 0;
		int maybe_closer = 0;
		for(int j=0;j<cand->candidates.size();j++){
			if(i==j){
				continue;
			}
			// count how many candidates that are surely closer than this one
			if(cand->candidates[i]->distance>=cand->candidates[j]->distance) {
				sure_closer++;
			}
			// count how many candidates that are possibly closer than this one
			if(!(cand->candidates[i]->distance<=cand->candidates[j]->distance)) {
				maybe_closer++;
			}
		}
		int cand_left = ctx.knn-cand->candidate_confirmed;
		if(global_ctx.verbose>=1){
			log("%ld\t%5ld sure closer %3d maybe closer %3d (%3d +%3d)",
					cand->mesh_wrapper->id,
					cand->candidates[i]->mesh_wrapper->id,
					sure_closer,
					maybe_closer,
					cand->candidate_confirmed,
					cand_left);
		}
		// the rank makes sure this one is confirmed
		if(maybe_closer < cand_left){
			target->report_result(cand->candidates[i]->mesh_wrapper);
			cand->candidate_confirmed++;
			//delete cand->candidates[i];
			cand->candidates.erase(cand->candidates.begin()+i);
			list_size--;
			//log("ranked %d, %d confirmed", rank, target->candidate_confirmed);
			continue;
		}

		// the rank makes sure this one should be removed as it must not be qualified
		if(sure_closer >= cand_left){
			//delete cand->candidates[i];
			cand->candidates.erase(cand->candidates.begin()+i);
			list_size--;
			continue;
		}
		i++;
	}//end for
	// the target one should be kept
}

bool result_sort(pair<int, int> a, pair<int, int> b){
	if(a.first<b.first){
		return true;
	}else if(a.first>b.first){
		return false;
	}else{
		return a.second<=b.second;
	}
}

void evaluate_candidate_lists(vector<candidate_entry *> &candidates, query_context &ctx){

#pragma omp parallel for
	for (int i = 0; i < candidates.size();i++) {
		update_candidate_list_knn(candidates[i], ctx);
	}

	int comfirm_this_round = 0;
	for(vector<candidate_entry *>::iterator it=candidates.begin();it!=candidates.end();){
		if((*it)->candidate_confirmed==ctx.knn){
			delete *it;
			it = candidates.erase(it);
			comfirm_this_round++;
		}else{
			it++;
		}
	}
	std::cout << "####comfirm_this_round = " << comfirm_this_round << std::endl;
}

void evaluate_candidate_lists_gpu(vector<candidate_entry *> &candidates, 
									query_context &ctx
								)
{
	vector<int> h_prefix_tile;  

	int target_count = candidates.size();
	h_prefix_tile.resize(target_count + 1);
	h_prefix_tile[0] = 0;

	vector<float> h_min;
	vector<float> h_max;
	vector<int> h_confirmed(target_count);

	for (int i = 0; i < target_count; ++i)
	{
		candidate_entry* ce = candidates[i];
		h_confirmed[i] = ce->candidate_confirmed;

		for (auto ci : ce->candidates)
		{
			h_min.push_back(ci->distance.mindist);
			h_max.push_back(ci->distance.maxdist);
		}

		h_prefix_tile[i+1] = h_min.size();
	}
	int total_pairs = h_min.size();


	vector<int> h_status;
	evaluation_kernel(total_pairs,
						target_count,
						h_min,
						h_max,
						h_prefix_tile,
						h_confirmed,
						ctx.knn,
						h_status
					);

	for (int t = 0; t < target_count; ++t)
	{
		candidate_entry* ce = candidates[t];
		int begin = h_prefix_tile[t];
		int end   = h_prefix_tile[t+1];

		vector<candidate_info *> new_list;

		for (int i = begin; i < end; ++i)
		{
			int local_idx = i - begin;

			if (h_status[i] == 1) // CONFIRMED
			{
				ce->mesh_wrapper->report_result(
					ce->candidates[local_idx]->mesh_wrapper);
				ce->candidate_confirmed++;

				if (ce->candidate_confirmed == ctx.knn) 
					break;
			}
			else if (h_status[i] == 0) // KEEP
			{
				new_list.push_back(ce->candidates[local_idx]);
			}
		}
		ce->candidates.swap(new_list);
	}
}

range distance_strict(Voxel *v1, Voxel* v2)
{
	range ret;
    ret.mindist = 0.0f;
    ret.maxdist = 0.0f;

    for (int i = 0; i < 3; i++)
    {
        float tmp1 = v1->low[i]  - v2->high[i];
        float tmp2 = v1->high[i] - v2->low[i];

        if (tmp2 < 0.0f)
            ret.mindist += tmp2 * tmp2;
        else if (tmp1 > 0.0f)
            ret.mindist += tmp1 * tmp1;

        // core point distance
        float dc = v1->core[i] - v2->core[i];
        ret.maxdist += dc * dc;
    }

    ret.mindist = std::sqrt(ret.mindist);
    ret.maxdist = std::sqrt(ret.maxdist);
    return ret;
}

vector<candidate_entry *> SpatialJoin::mbb_knn(Tile *tile1, Tile *tile2, query_context &ctx){
	vector<candidate_entry *> candidates;
	OctreeNode *tree = tile2->get_octree();
	size_t tile1_size = min(tile1->num_objects(), ctx.max_num_objects1);

	// double total_query_knn_time = 0.0;
	// double total_voxel_dist_time = 0.0;
	// double total_evaluate_time = 0.0;
	// double total_voxel_pair_loop_time = 0.0;

#pragma omp parallel for
	for(int i=0;i<tile1_size;i++){
		vector<pair<int, range>> candidate_ids;
		// for each object
		//1. use the distance between the mbbs of objects as a
		//	 filter to retrieve candidate objects
		HiMesh_Wrapper *wrapper1 = tile1->get_mesh_wrapper(i);
		float min_maxdistance = DBL_MAX;

		// auto t0 = std::chrono::high_resolution_clock::now();

		tree->query_knn(&(wrapper1->box), wrapper1->transform, candidate_ids, min_maxdistance, ctx.knn);
		assert(candidate_ids.size()>=ctx.knn);

		// auto t1 = std::chrono::high_resolution_clock::now();

		// total_query_knn_time += std::chrono::duration<double>(t1 - t0).count();

		if(candidate_ids.size() == ctx.knn){
			for(pair<int, range> &p:candidate_ids){
				wrapper1->report_result(tile2->get_mesh_wrapper(p.first));
			}
			candidate_ids.clear();
			continue;
		}

		candidate_entry *ce = new candidate_entry(wrapper1);

		// t0 = std::chrono::high_resolution_clock::now();

		//2. we further go through the voxels in two objects to shrink
		// 	 the candidate list in a finer grain

		/**
		 * Some improvements:
		 * 1. Use GPU to compute distance for voxel pair in one kernel at once 
		 */
		for(pair<int, range> &p:candidate_ids) {
			HiMesh_Wrapper *wrapper2 = tile2->get_mesh_wrapper(p.first);
			candidate_info *ci = new candidate_info(wrapper2);
			float min_maxdist = DBL_MAX;

			
			// auto t2 = std::chrono::high_resolution_clock::now();
			for(Voxel *v1:wrapper1->voxels){
				for(Voxel *v2:wrapper2->voxels){

					// range dist_vox = v1->distance(*v2);
					range dist_vox = distance_strict(v1, v2);

					if(dist_vox.mindist>=min_maxdist){
						continue;
					}
					// wait for later evaluation
					ci->voxel_pairs.push_back(voxel_pair(v1, v2, dist_vox));
					min_maxdist = min(min_maxdist, dist_vox.maxdist);
				}
			}
			// auto t3 = std::chrono::high_resolution_clock::now();
			// total_voxel_pair_loop_time += std::chrono::duration<double>(t3 - t2).count();
			// form the distance range of objects with the evaluations of voxel pairs
			ci->distance = update_voxel_pair_list(ci->voxel_pairs, min_maxdist);
			assert(ci->voxel_pairs.size()>0);
			assert(ci->distance.mindist<=ci->distance.maxdist);
			ce->add_candidate(ci);
		}

		// t1 = std::chrono::high_resolution_clock::now();
		// total_voxel_dist_time += std::chrono::duration<double>(t1 - t0).count();

		//log("%ld %ld", candidate_ids.size(),candidate_list.size());
		// save the candidate list
		if(ce->candidates.size()>0){
#pragma omp critical
			candidates.push_back(ce);
		}else{
			delete ce;
		}
		candidate_ids.clear();
	}

	// std::cout << "total_query_knn_time = " << total_query_knn_time << std::endl;
	// std::cout << "total_voxel_dist_time = " << total_voxel_dist_time << std::endl;
	// std::cout << "total_voxel_pair_loop_time = " << total_voxel_pair_loop_time << std::endl;
	// the candidates list need be evaluated after checking with the mbb
	// some queries might be answered with only querying the index

	auto t0 = std::chrono::high_resolution_clock::now();
	evaluate_candidate_lists(candidates, ctx);
	auto t1 = std::chrono::high_resolution_clock::now();
	double total_evaluate_time = std::chrono::duration<double>(t1 - t0).count();
	std::cout << "filtering phase evaluate time = " << total_evaluate_time << std::endl;

	return candidates;
}


tuple<vector<int>, vector<int>, vector<float>> SpatialJoin::prepare_data(size_t tile_size, Tile *tile) 
{
    vector<int> voxel_count(tile_size);
    vector<int> voxel_offset(tile_size + 1, 0);

    for (int i = 0; i < tile_size; ++i) {
        HiMesh_Wrapper *wrapper = tile->get_mesh_wrapper(i);
        voxel_count[i] = wrapper->voxels.size();
        voxel_offset[i + 1] = voxel_offset[i] + voxel_count[i];
    }

    int total_voxels = voxel_offset[tile_size];

	vector<float> h_all_voxels(total_voxels * 9);

    for (int i = 0; i < tile_size; ++i) {
        HiMesh_Wrapper *wrapper = tile->get_mesh_wrapper(i);
        int base = voxel_offset[i] * 9;
        for (int j = 0; j < wrapper->voxels.size(); ++j) {
            h_all_voxels[base + j*9 + 0] = wrapper->voxels[j]->low[0];
            h_all_voxels[base + j*9 + 1] = wrapper->voxels[j]->low[1];
            h_all_voxels[base + j*9 + 2] = wrapper->voxels[j]->low[2];
            h_all_voxels[base + j*9 + 3] = wrapper->voxels[j]->high[0];
            h_all_voxels[base + j*9 + 4] = wrapper->voxels[j]->high[1];
            h_all_voxels[base + j*9 + 5] = wrapper->voxels[j]->high[2];

            h_all_voxels[base + j*9 + 6] = wrapper->voxels[j]->core[0];
            h_all_voxels[base + j*9 + 7] = wrapper->voxels[j]->core[1];
            h_all_voxels[base + j*9 + 8] = wrapper->voxels[j]->core[2];
        }
    }
	return {voxel_offset, voxel_count, h_all_voxels};
}

vector<candidate_entry *> SpatialJoin::mbb_knn_op_gpu(Tile *tile1, Tile *tile2, query_context &ctx){
	vector<candidate_entry *> candidates;
	OctreeNode *tree = tile2->get_octree();
	tile1_size = min(tile1->num_objects(), ctx.max_num_objects1);
	tile2_size = min(tile2->num_objects(), ctx.max_num_objects1);

	// prepare data for wrapper1->voxels
	// high_x, high_y, high_z, low_x, low_y, low_z
	auto t0 = std::chrono::high_resolution_clock::now();
	
	auto r1 = prepare_data(tile1_size, tile1);
	DeviceVoxels d_voxels_1 = allocateVoxelsForAll(tile1_size, get<0>(r1), get<1>(r1), get<2>(r1));
	auto r2 = prepare_data(tile2_size, tile2);
	DeviceVoxels d_voxels_2 = allocateVoxelsForAll(tile2_size, get<0>(r2), get<1>(r2), get<2>(r2));

	auto t1 = std::chrono::high_resolution_clock::now();
	double total_prepare_time = std::chrono::duration<double>(t1 - t0).count();
	std::cout << "total_prepare_time = " << total_prepare_time << std::endl;

	double total_kernel_time = 0.0;

// #pragma omp parallel for
	for(int i=0;i<tile1_size;i++){
		vector<pair<int, range>> candidate_ids;
		// for each object
		//1. use the distance between the mbbs of objects as a
		//	 filter to retrieve candidate objects
		HiMesh_Wrapper *wrapper1 = tile1->get_mesh_wrapper(i);
		float min_maxdistance = DBL_MAX;

		tree->query_knn(&(wrapper1->box), wrapper1->transform, candidate_ids, min_maxdistance, ctx.knn);
		assert(candidate_ids.size()>=ctx.knn);


		if(candidate_ids.size() == ctx.knn){
//			for(pair<int, range> &p:candidate_ids){
//				wrapper1->report_result(tile2->get_mesh_wrapper(p.first));
//			}
			candidate_ids.clear();
			continue;
		}

		candidate_entry *ce = new candidate_entry(wrapper1);

		//2. we further go through the voxels in two objects to shrink
		// 	 the candidate list in a finer grain

		/**
		 * Some improvements:
		 * 1. Use GPU to compute distance for voxel pair in one kernel at once 
		 */
		/**
		 * @@ compute distance for wrapper1->voxels and all wrappers->voxel in candidate_ids
		 * 	one object pair per block
		*/


		vector<float> out_min;
		vector<float> out_max;

		auto t2 = std::chrono::high_resolution_clock::now();

		compute_voxel_pair_distance(i, candidate_ids, d_voxels_1, d_voxels_2, out_min, out_max);

		auto t3 = std::chrono::high_resolution_clock::now();

		total_kernel_time += std::chrono::duration<double>(t3 - t2).count();
		
		int index = 0;
		for(pair<int, range> &p:candidate_ids) {
			HiMesh_Wrapper *wrapper2 = tile2->get_mesh_wrapper(p.first);
			
			
			candidate_info *ci = new candidate_info(wrapper2);
			float min_maxdist = DBL_MAX;
			for(Voxel *v1:wrapper1->voxels){
			  	for(Voxel *v2:wrapper2->voxels){
					// range dist_vox = v1->distance(*v2);

					
					range dist_vox = range{out_min[index], out_max[index]};
					index++;

					// if (fabsf(dist_vox.mindist - dist_vox_1.mindist) < 1e-6f) {
					// 	std::cout << dist_vox.mindist << ", " << dist_vox_1.mindist << std::endl;
					// 	assert(false);
					// }
					// if (fabsf(dist_vox.maxdist - dist_vox_1.maxdist) < 1e-6f) {
					// 	std::cout << dist_vox.maxdist << ", " << dist_vox_1.maxdist << std::endl;
					// 	assert(false);
					// }

					if(dist_vox.mindist>=min_maxdist){
						continue;
					}
					// wait for later evaluation
					ci->voxel_pairs.push_back(voxel_pair(v1, v2, dist_vox));
					min_maxdist = min(min_maxdist, dist_vox.maxdist);
				}
			}
			

			// form the distance range of objects with the evaluations of voxel pairs
			ci->distance = update_voxel_pair_list(ci->voxel_pairs, min_maxdist);
			assert(ci->voxel_pairs.size()>0);
			assert(ci->distance.mindist<=ci->distance.maxdist);
			ce->add_candidate(ci);
		}

		//log("%ld %ld", candidate_ids.size(),candidate_list.size());
		// save the candidate list
		if(ce->candidates.size()>0){
// #pragma omp critical
			candidates.push_back(ce);
		}else{
			delete ce;
		}
		candidate_ids.clear();
	}
	std::cout << "total_kernel_time = " << total_kernel_time << std::endl;
	// the candidates list need be evaluated after checking with the mbb
	// some queries might be answered with only querying the index
	evaluate_candidate_lists(candidates, ctx);
	return candidates;
}

vector<candidate_entry *> SpatialJoin::mbb_knn_op_gpu_one_kernel(Tile *tile1, Tile *tile2, query_context &ctx){
	vector<candidate_entry *> candidates;
	OctreeNode *tree = tile2->get_octree();
	tile1_size = min(tile1->num_objects(), ctx.max_num_objects1);
	tile2_size = min(tile2->num_objects(), ctx.max_num_objects1);

	// prepare data for wrapper1->voxels
	// high_x, high_y, high_z, low_x, low_y, low_z

	this->tile1 = tile1;
	this->tile2 = tile2;

	auto t0 = std::chrono::high_resolution_clock::now();

	auto r1 = prepare_data(tile1_size, tile1);
	DeviceVoxels d_voxels_1 = allocateVoxelsForAll(tile1_size, get<0>(r1), get<1>(r1), get<2>(r1));
	auto r2 = prepare_data(tile2_size, tile2);
	DeviceVoxels d_voxels_2 = allocateVoxelsForAll(tile2_size, get<0>(r2), get<1>(r2), get<2>(r2));

	// 1. first gather objects ids in kNN.
	// can be used later
	prefix_tile.assign(tile1_size+1, 0);
	prefix_obj.push_back(0);

	size_t total_voxel_size = 0;
	size_t max_voxel_size = 0;
	// for(int i=0;i<tile1_size;i++) {
	// 	HiMesh_Wrapper *wrapper1 = tile1->get_mesh_wrapper(i);
	// 	vector<pair<int, range>> candidate_ids;
	// 	float min_maxdistance = DBL_MAX;

	// 	tree->query_knn(&(wrapper1->box), candidate_ids, min_maxdistance, ctx.knn);

	// 	prefix_tile[i+1] = prefix_tile[i] + candidate_ids.size();

	// 	for (int j=0; j<candidate_ids.size(); ++j) {
	// 		int id = candidate_ids[j].first;
	// 		compute_obj_2.push_back(id);
	// 		compute_obj_1.push_back(i);
	// 		prefix_obj.push_back(prefix_obj.back() + d_voxels_2.h_voxel_count[id] * d_voxels_1.h_voxel_count[i]);
	// 		total_voxel_size += d_voxels_2.h_voxel_count[id] * d_voxels_1.h_voxel_count[i];
	// 		if (d_voxels_2.h_voxel_count[id] * d_voxels_1.h_voxel_count[i] > max_voxel_size) {
	// 			max_voxel_size = d_voxels_2.h_voxel_count[id] * d_voxels_1.h_voxel_count[i];
	// 		}
	// 	}
	// }
	

	// ================== parallelized version ========================

	vector<vector<int>> all_ids(tile1_size);
	vector<size_t> candidate_count(tile1_size);

	auto ta = std::chrono::high_resolution_clock::now();

	#pragma omp parallel for schedule(dynamic)
	for (int i = 0; i < tile1_size; i++) {

		HiMesh_Wrapper *wrapper1 = tile1->get_mesh_wrapper(i);

		vector<pair<int, range>> candidate_ids;
		float min_maxdistance = DBL_MAX;

		tree->query_knn_fast(&(wrapper1->box),
						wrapper1->transform,
						candidate_ids,
						min_maxdistance,
						ctx.knn);


		candidate_count[i] = candidate_ids.size();

		all_ids[i].reserve(candidate_ids.size());

		for (auto &p : candidate_ids)
			all_ids[i].push_back(p.first);
	}

	auto tb = std::chrono::high_resolution_clock::now();
	double for_loop_time = std::chrono::duration<double>(tb - ta).count();
	std::cout << "omp for_loop_time = " << for_loop_time << std::endl;

	prefix_tile[0] = 0;
	for (int i = 0; i < tile1_size; i++)
		prefix_tile[i+1] = prefix_tile[i] + candidate_count[i];


	for (int i = 0; i < tile1_size; i++) {

		for (int id : all_ids[i]) {

			compute_obj_1.push_back(i);
			compute_obj_2.push_back(id);

			size_t vox = d_voxels_2.h_voxel_count[id] * d_voxels_1.h_voxel_count[i];

			prefix_obj.push_back(prefix_obj.back() + vox);

			total_voxel_size += vox;

			if (vox > max_voxel_size)
				max_voxel_size = vox;
		}
	}
	// ===================================================================

	std::cout << "max_voxel_size = " << max_voxel_size << std::endl;
	std::cout << "total_voxel_size = " << total_voxel_size << std::endl;

	auto t2 = std::chrono::high_resolution_clock::now();
	double total_prepare_data_time = std::chrono::duration<double>(t2 - t0).count();
	std::cout << "total_prepare_data_time = " << total_prepare_data_time << std::endl;

	/** how about compute all pairwise distance between V1 and V2 here. */
	auto t3 = std::chrono::high_resolution_clock::now();
	
	compute_voxel_pair_distance_for_all_streaming_pipeline_reduceIO(d_voxels_1, d_voxels_2, prefix_tile, prefix_obj, compute_obj_1, compute_obj_2, 
										total_voxel_size, max_voxel_size, out_min, out_max, all_min_max_dist, all_min_min_dist, 
										valid_vp_cnt, valid_voxel_prefix, valid_voxel_pairs, valid_voxel_pairs_dist,
										ctx.knn, confirmed_after, status_after);

	auto t4 = std::chrono::high_resolution_clock::now();
	double total_kernel_time = std::chrono::duration<double>(t4 - t3).count();
	std::cout << "total_kernel_time = " << total_kernel_time << std::endl;


	//@@@@ load parameters for future usage
	addParams.max_voxel_size = max_voxel_size;
	addParams.obj_pair_num = compute_obj_1.size();
	addParams.N = tile1_size;
	addParams.K = ctx.knn;
	addParams.total_voxel_size = total_voxel_size;

// #pragma omp parallel for

	t0 = std::chrono::high_resolution_clock::now();
	size_t voxel_lvl_index = 0;
	size_t object_lvl_index = 0;
	

	#pragma omp parallel for schedule(dynamic)
	for (int i=0;i<tile1_size;i++) {


		// vector<pair<int, range>> candidate_ids;
		// for each object
		//1. use the distance between the mbbs of objects as a
		//	 filter to retrieve candidate objects
		HiMesh_Wrapper *wrapper1 = tile1->get_mesh_wrapper(i);
		// float min_maxdistance = DBL_MAX;

		int candidate_size = prefix_tile[i+1] - prefix_tile[i];

		// removed
		// tree->query_knn(&(wrapper1->box), candidate_ids, min_maxdistance, ctx.knn);
		// assert(candidate_ids.size()>=ctx.knn);

		// if(candidate_ids.size() == ctx.knn){
//			for(pair<int, range> &p:candidate_ids){
//				wrapper1->report_result(tile2->get_mesh_wrapper(p.first));
//			}
			// candidate_ids.clear();
			// continue;
		// }

		if (candidate_size == ctx.knn) {
			
			int begin = prefix_tile[i];
			int end   = prefix_tile[i + 1];
			for (int c = begin; c < end; ++c) {
				int obj2_id = compute_obj_2[c];
				wrapper1->report_result(
					tile2->get_mesh_wrapper(obj2_id)
				);
			}
			continue;
		}

		candidate_entry *ce = new candidate_entry(wrapper1);

		//2. we further go through the voxels in two objects to shrink
		// 	 the candidate list in a finer grain

		if (confirmed_after[i] == ctx.knn) {

			int begin = prefix_tile[i];
			int end   = prefix_tile[i + 1];

			for (int c = begin; c < end; ++c) {
				if (status_after[c] == CONFIRMED) {
					int obj2_id = compute_obj_2[c];
					wrapper1->report_result(
						tile2->get_mesh_wrapper(obj2_id)
					);
					status_after[c] = REPORTED;
				}
			}
			// confirmed_after[i] += 1; // don't want redundant results, no harm to do this...
			continue;
		}

		for (int j = prefix_tile[i]; j < prefix_tile[i+1]; ++j) {

			if (status_after[j] == CONFIRMED) {
				int obj2_id = compute_obj_2[j];
				wrapper1->report_result(
					tile2->get_mesh_wrapper(obj2_id)
				);
				ce->candidate_confirmed++;
				status_after[j] = REPORTED;

				continue;
			}
			if (status_after[j] == UNDECIDED) 
			{
				HiMesh_Wrapper *wrapper2 = tile2->get_mesh_wrapper(compute_obj_2[j]);
				candidate_info *ci = new candidate_info(wrapper2);
				ci->dist_id = j;

				float min_maxdist = all_min_max_dist[j];
				float min_mindist = all_min_min_dist[j];
				int   valid_cnt   = valid_vp_cnt[j];
				size_t base = valid_voxel_prefix[j];

				ci->voxel_pairs.reserve(valid_cnt);

				for (int k = 0; k < valid_cnt; ++k) {
					size_t cur_idx = base + k;
					int v1_idx  = valid_voxel_pairs[3 * cur_idx];
					int v2_idx  = valid_voxel_pairs[3 * cur_idx + 1];
					int t       = valid_voxel_pairs[3 * cur_idx + 2];

					float min_dist = valid_voxel_pairs_dist[2 * cur_idx];
					float max_dist = valid_voxel_pairs_dist[2 * cur_idx + 1];

					// if (v1_idx < wrapper1->voxels.size() &&
					// 	v2_idx < wrapper2->voxels.size())

						ci->voxel_pairs.push_back(
							voxel_pair{
								wrapper1->voxels[v1_idx],
								wrapper2->voxels[v2_idx],
								range{
									min_dist,
									max_dist
								},
								i,	// which object x
								j, 	// which object y
								t   // which voxel-pair
							}
						);
				}
				ci->distance.maxdist = min_maxdist;
				ci->distance.mindist = min_mindist;
				ce->add_candidate(ci);
			}
		}
		
		/** 
		// for(pair<int, range> &p:candidate_ids) {
		for (int j=prefix_tile[i]; j<prefix_tile[i+1]; ++j) {


			// HiMesh_Wrapper *wrapper2 = tile2->get_mesh_wrapper(p.first);
			HiMesh_Wrapper *wrapper2 = tile2->get_mesh_wrapper(compute_obj_2[j]);
			
			candidate_info ci(wrapper2);
			// float min_maxdist = DBL_MAX;
			float min_maxdist = all_min_max_dist[object_lvl_index];
			float min_mindist = all_min_min_dist[object_lvl_index];
			int valid_cnt = valid_vp_cnt[object_lvl_index];

			int base = valid_voxel_prefix[object_lvl_index];

			for (int k=0; k<valid_cnt; ++k) {
				
				int cur_idx = base + k;
				int v1_idx = valid_voxel_pairs[3 * cur_idx];
				int v2_idx = valid_voxel_pairs[3 * cur_idx + 1];
				int t      = valid_voxel_pairs[3 * cur_idx + 2];

				ci.voxel_pairs.push_back(
					voxel_pair{
						wrapper1->voxels[v1_idx],
						wrapper2->voxels[v2_idx],
						range{
							out_min[prefix_obj[object_lvl_index] + t],
							out_max[prefix_obj[object_lvl_index] + t]
						}
					});
			}
			


			// for(int ii=0; ii<wrapper1->voxels.size(); ++ii){
			// 	Voxel *v1 = wrapper1->voxels[ii];
			//   	for(int jj=0; jj<wrapper2->voxels.size(); ++jj){
			// 		Voxel *v2 = wrapper2->voxels[jj];
			// 		// range dist_vox = v1->distance(*v2);
			// 		range dist_vox = range{out_min[voxel_lvl_index], out_max[voxel_lvl_index]};
			// 		voxel_lvl_index++;

			// 		if(dist_vox.mindist>=min_maxdist){
			// 			continue;
			// 		}
			// 		// wait for later evaluation
			// 		ci.voxel_pairs.push_back(voxel_pair(v1, v2, dist_vox));
			// 		// min_maxdist = min(min_maxdist, dist_vox.maxdist);
			// 	}
			// }

			// if (valid_cnt != ci.voxel_pairs.size()) {
			// 	std::cout << valid_cnt << ',' << ci.voxel_pairs.size() << std::endl;
			// 	assert(false);
			// }
			
			// form the distance range of objects with the evaluations of voxel pairs
			// auto t2 = std::chrono::high_resolution_clock::now();
			// ci.distance = update_voxel_pair_list(ci.voxel_pairs, min_maxdist);
			ci.distance.maxdist = min_maxdist;
			ci.distance.mindist = min_mindist;


			// auto t3 = std::chrono::high_resolution_clock::now();
			// total_update_time += std::chrono::duration<double>(t3 - t2).count();

			assert(ci.voxel_pairs.size()>0);
			assert(ci.distance.mindist<=ci.distance.maxdist);
			ce->add_candidate(ci);

			object_lvl_index++;
		}
		*/

		//log("%ld %ld", candidate_ids.size(),candidate_list.size());
		// save the candidate list
		if(ce->candidates.size()>0){
		#pragma omp critical
			candidates.push_back(ce);
		}else{
			delete ce;
		}
		// candidate_ids.clear();

	}

	auto t1 = std::chrono::high_resolution_clock::now();

	double total_candidate_time = std::chrono::duration<double>(t1 - t0).count();
	std::cout << "total_candidate_time = " << total_candidate_time << std::endl;

	std::cout << "total valid voxel pair = " << get_pair_num(candidates) << std::endl;

	return candidates;
}


vector<candidate_entry *> SpatialJoin::mbb_knn_op_gpu_one_kernel_disk(Tile *tile1, Tile *tile2, query_context &ctx){
	vector<candidate_entry *> candidates;
	OctreeNode *tree = tile2->get_octree();
	tile1_size = min(tile1->num_objects(), ctx.max_num_objects1);
	tile2_size = min(tile2->num_objects(), ctx.max_num_objects1);

	// prepare data for wrapper1->voxels
	// high_x, high_y, high_z, low_x, low_y, low_z

	this->tile1 = tile1;
	this->tile2 = tile2;

	auto t0 = std::chrono::high_resolution_clock::now();

	auto r1 = prepare_data(tile1_size, tile1);
	DeviceVoxels d_voxels_1 = allocateVoxelsForAll(tile1_size, get<0>(r1), get<1>(r1), get<2>(r1));
	auto r2 = prepare_data(tile2_size, tile2);
	DeviceVoxels d_voxels_2 = allocateVoxelsForAll(tile2_size, get<0>(r2), get<1>(r2), get<2>(r2));

	prefix_tile.assign(tile1_size+1, 0);
	prefix_obj.push_back(0);

	size_t total_voxel_size = 0;
	size_t max_voxel_size = 0;

	// ================== parallelized version ========================

	vector<vector<int>> all_ids(tile1_size);
	vector<size_t> candidate_count(tile1_size);

	auto ta = std::chrono::high_resolution_clock::now();

	#pragma omp parallel for schedule(dynamic)
	for (int i = 0; i < tile1_size; i++) {

		HiMesh_Wrapper *wrapper1 = tile1->get_mesh_wrapper(i);

		vector<pair<int, range>> candidate_ids;
		float min_maxdistance = DBL_MAX;

		tree->query_knn_fast(&(wrapper1->box),
						wrapper1->transform,
						candidate_ids,
						min_maxdistance,
						ctx.knn);


		candidate_count[i] = candidate_ids.size();

		all_ids[i].reserve(candidate_ids.size());

		for (auto &p : candidate_ids)
			all_ids[i].push_back(p.first);
	}

	auto tb = std::chrono::high_resolution_clock::now();
	double for_loop_time = std::chrono::duration<double>(tb - ta).count();
	std::cout << "omp for_loop_time = " << for_loop_time << std::endl;

	prefix_tile[0] = 0;
	for (int i = 0; i < tile1_size; i++)
		prefix_tile[i+1] = prefix_tile[i] + candidate_count[i];


	for (int i = 0; i < tile1_size; i++) {

		for (int id : all_ids[i]) {

			compute_obj_1.push_back(i);
			compute_obj_2.push_back(id);

			size_t vox = d_voxels_2.h_voxel_count[id] * d_voxels_1.h_voxel_count[i];

			prefix_obj.push_back(prefix_obj.back() + vox);

			total_voxel_size += vox;

			if (vox > max_voxel_size)
				max_voxel_size = vox;
		}
	}
	// ===================================================================

	std::cout << "max_voxel_size = " << max_voxel_size << std::endl;
	std::cout << "total_voxel_size = " << total_voxel_size << std::endl;

	auto t2 = std::chrono::high_resolution_clock::now();
	double total_prepare_data_time = std::chrono::duration<double>(t2 - t0).count();
	std::cout << "total_prepare_data_time = " << total_prepare_data_time << std::endl;

	/** how about compute all pairwise distance between V1 and V2 here. */
	auto t3 = std::chrono::high_resolution_clock::now();
	
	compute_voxel_pair_distance_for_all_streaming_pipeline_disk(d_voxels_1, d_voxels_2, prefix_tile, prefix_obj, compute_obj_1, compute_obj_2, 
										total_voxel_size, max_voxel_size, out_min, out_max, all_min_max_dist, all_min_min_dist, 
										valid_vp_cnt, valid_voxel_prefix, valid_voxel_pairs, valid_voxel_pairs_dist,
										ctx.knn, confirmed_after, status_after);

	auto t4 = std::chrono::high_resolution_clock::now();
	double total_kernel_time = std::chrono::duration<double>(t4 - t3).count();
	std::cout << "total_kernel_time = " << total_kernel_time << std::endl;


	//@@@@ load parameters for future usage
	addParams.max_voxel_size = max_voxel_size;
	addParams.obj_pair_num = compute_obj_1.size();
	addParams.N = tile1_size;
	addParams.K = ctx.knn;
	addParams.total_voxel_size = total_voxel_size;

// #pragma omp parallel for

	t0 = std::chrono::high_resolution_clock::now();
	size_t voxel_lvl_index = 0;
	size_t object_lvl_index = 0;

	for (int i=0;i<tile1_size;i++) {


		// vector<pair<int, range>> candidate_ids;
		// for each object
		//1. use the distance between the mbbs of objects as a
		//	 filter to retrieve candidate objects
		HiMesh_Wrapper *wrapper1 = tile1->get_mesh_wrapper(i);
		// float min_maxdistance = DBL_MAX;

		int candidate_size = prefix_tile[i+1] - prefix_tile[i];

		// removed
		// tree->query_knn(&(wrapper1->box), candidate_ids, min_maxdistance, ctx.knn);
		// assert(candidate_ids.size()>=ctx.knn);

		// if(candidate_ids.size() == ctx.knn){
//			for(pair<int, range> &p:candidate_ids){
//				wrapper1->report_result(tile2->get_mesh_wrapper(p.first));
//			}
			// candidate_ids.clear();
			// continue;
		// }

		if (candidate_size == ctx.knn) {
			
			int begin = prefix_tile[i];
			int end   = prefix_tile[i + 1];
			for (int c = begin; c < end; ++c) {
				int obj2_id = compute_obj_2[c];
				wrapper1->report_result(
					tile2->get_mesh_wrapper(obj2_id)
				);
			}
			// std::cout << "#########JUMP!!!" << std::endl;
			continue;
		}

		candidate_entry *ce = new candidate_entry(wrapper1);

		//2. we further go through the voxels in two objects to shrink
		// 	 the candidate list in a finer grain

		if (confirmed_after[i] == ctx.knn) {

			int begin = prefix_tile[i];
			int end   = prefix_tile[i + 1];

			for (int c = begin; c < end; ++c) {
				if (status_after[c] == CONFIRMED) {
					int obj2_id = compute_obj_2[c];
					wrapper1->report_result(
						tile2->get_mesh_wrapper(obj2_id)
					);
					status_after[c] = REPORTED;
				}
			}
			// confirmed_after[i] += 1; // don't want redundant results, no harm to do this...
			continue;
		}

		for (int j = prefix_tile[i]; j < prefix_tile[i+1]; ++j) {

			if (status_after[j] == CONFIRMED) {
				int obj2_id = compute_obj_2[j];
				wrapper1->report_result(
					tile2->get_mesh_wrapper(obj2_id)
				);
				ce->candidate_confirmed++;
				status_after[j] = REPORTED;

				continue;
			}
			if (status_after[j] == UNDECIDED) 
			{
				HiMesh_Wrapper *wrapper2 = tile2->get_mesh_wrapper(compute_obj_2[j]);
				candidate_info *ci = new candidate_info(wrapper2);
				ci->dist_id = j;

				float min_maxdist = all_min_max_dist[j];
				float min_mindist = all_min_min_dist[j];
				int   valid_cnt   = valid_vp_cnt[j];
				size_t base = valid_voxel_prefix[j];

				for (int k = 0; k < valid_cnt; ++k) {
					size_t cur_idx = base + k;
					int v1_idx  = valid_voxel_pairs[3 * cur_idx];
					int v2_idx  = valid_voxel_pairs[3 * cur_idx + 1];
					int t       = valid_voxel_pairs[3 * cur_idx + 2];

					float min_dist = valid_voxel_pairs_dist[2 * cur_idx];
					float max_dist = valid_voxel_pairs_dist[2 * cur_idx + 1];

					ci->voxel_pairs.push_back(
						voxel_pair{
							wrapper1->voxels[v1_idx],
							wrapper2->voxels[v2_idx],
							range{
								min_dist,
								max_dist
							},
							i,	// which object x
							j, 	// which object y
							t   // which voxel-pair
						}
					);
				}
				ci->distance.maxdist = min_maxdist;
				ci->distance.mindist = min_mindist;
				ce->add_candidate(ci);
			}
		}
		
		/** 
		// for(pair<int, range> &p:candidate_ids) {
		for (int j=prefix_tile[i]; j<prefix_tile[i+1]; ++j) {


			// HiMesh_Wrapper *wrapper2 = tile2->get_mesh_wrapper(p.first);
			HiMesh_Wrapper *wrapper2 = tile2->get_mesh_wrapper(compute_obj_2[j]);
			
			candidate_info ci(wrapper2);
			// float min_maxdist = DBL_MAX;
			float min_maxdist = all_min_max_dist[object_lvl_index];
			float min_mindist = all_min_min_dist[object_lvl_index];
			int valid_cnt = valid_vp_cnt[object_lvl_index];

			int base = valid_voxel_prefix[object_lvl_index];

			for (int k=0; k<valid_cnt; ++k) {
				
				int cur_idx = base + k;
				int v1_idx = valid_voxel_pairs[3 * cur_idx];
				int v2_idx = valid_voxel_pairs[3 * cur_idx + 1];
				int t      = valid_voxel_pairs[3 * cur_idx + 2];

				ci.voxel_pairs.push_back(
					voxel_pair{
						wrapper1->voxels[v1_idx],
						wrapper2->voxels[v2_idx],
						range{
							out_min[prefix_obj[object_lvl_index] + t],
							out_max[prefix_obj[object_lvl_index] + t]
						}
					});
			}
			


			// for(int ii=0; ii<wrapper1->voxels.size(); ++ii){
			// 	Voxel *v1 = wrapper1->voxels[ii];
			//   	for(int jj=0; jj<wrapper2->voxels.size(); ++jj){
			// 		Voxel *v2 = wrapper2->voxels[jj];
			// 		// range dist_vox = v1->distance(*v2);
			// 		range dist_vox = range{out_min[voxel_lvl_index], out_max[voxel_lvl_index]};
			// 		voxel_lvl_index++;

			// 		if(dist_vox.mindist>=min_maxdist){
			// 			continue;
			// 		}
			// 		// wait for later evaluation
			// 		ci.voxel_pairs.push_back(voxel_pair(v1, v2, dist_vox));
			// 		// min_maxdist = min(min_maxdist, dist_vox.maxdist);
			// 	}
			// }

			// if (valid_cnt != ci.voxel_pairs.size()) {
			// 	std::cout << valid_cnt << ',' << ci.voxel_pairs.size() << std::endl;
			// 	assert(false);
			// }
			
			// form the distance range of objects with the evaluations of voxel pairs
			// auto t2 = std::chrono::high_resolution_clock::now();
			// ci.distance = update_voxel_pair_list(ci.voxel_pairs, min_maxdist);
			ci.distance.maxdist = min_maxdist;
			ci.distance.mindist = min_mindist;


			// auto t3 = std::chrono::high_resolution_clock::now();
			// total_update_time += std::chrono::duration<double>(t3 - t2).count();

			assert(ci.voxel_pairs.size()>0);
			assert(ci.distance.mindist<=ci.distance.maxdist);
			ce->add_candidate(ci);

			object_lvl_index++;
		}
		*/

		//log("%ld %ld", candidate_ids.size(),candidate_list.size());
		// save the candidate list
		if(ce->candidates.size()>0){
// #pragma omp critical
			candidates.push_back(ce);
		}else{
			delete ce;
		}
		// candidate_ids.clear();

	}

	// update_status_and_confirm(confirmed_after, 
	// 							status_after,
	// 							tile1_size,
	// 							prefix_tile[tile1_size]
	// 						);

	auto t1 = std::chrono::high_resolution_clock::now();

	double total_candidate_time = std::chrono::duration<double>(t1 - t0).count();
	std::cout << "total_candidate_time = " << total_candidate_time << std::endl;

	return candidates;
}



/*
 * the main function for getting the nearest neighbor
 *
 * */
void SpatialJoin::nearest_neighbor(query_context ctx){
	struct timeval start = get_cur_time();
	struct timeval very_start = get_cur_time();

	// filtering with MBBs to get the candidate list
	std::cout << "Enter mbb_knn_op_gpu_one_kernel" << std::endl;
	auto tx = std::chrono::high_resolution_clock::now();
	
	vector<candidate_entry *> candidates = mbb_knn_op_gpu_one_kernel(ctx.tile1, ctx.tile2, ctx);
	auto ty = std::chrono::high_resolution_clock::now();
	cout << "Filtering stage takes " << std::chrono::duration<double>(ty - tx).count() << std::endl;
	ctx.index_time += logt("index retrieving", start);


	// now we start to get the distances with progressive level of details
	for(uint32_t lod:ctx.lods){
		ctx.cur_lod = lod;
		struct timeval iter_start = get_cur_time();
		start = get_cur_time();

		const int pair_num = get_pair_num(candidates);

		//TODO: not break temporarily
		if(pair_num==0){
			break;
		}

		size_t candidate_num = get_candidate_num(candidates);
		log("%ld polyhedron has %d candidates %d voxel pairs %.2f voxel pairs per candidate",
				candidates.size(), candidate_num, pair_num, (1.0*pair_num)/candidates.size());

		// truly conduct the geometric computations
		auto gp = calculate_distance(candidates, ctx, &addParams);

		// now update the distance range with the new distances
		int index = 0;
		start = get_cur_time();


		auto tx = std::chrono::high_resolution_clock::now();

		for(int i=0; i<candidates.size(); ++i){

			candidate_entry *c = candidates[i];
			HiMesh_Wrapper *wrapper1 = c->mesh_wrapper;

			for(candidate_info *ci:c->candidates){ 

				HiMesh_Wrapper *wrapper2 = ci->mesh_wrapper;


				// float min_maxdist = gp.min_max_dist[ci.dist_id];
				// float min_mindist = gp.min_min_dist[ci.dist_id];
				
				
				double vox_minmaxdist = DBL_MAX;

				// replace this for-loop
				for(voxel_pair &vp:ci->voxel_pairs){

					result_container res = ctx.results[index++];
					// update the distance
					if(vp.v1->num_triangles>0&&vp.v2->num_triangles>0){
						range dist = vp.dist;
						if(lod==ctx.highest_lod()) {
							// now we have a precise distance
							dist.mindist = res.distance;
							dist.maxdist = res.distance;

						}else if(global_ctx.hausdorf_level == 2){
							dist.mindist = std::min(dist.mindist, res.min_dist);
							dist.maxdist = std::min(dist.maxdist, res.max_dist);
//								dist.maxdist = std::min(dist.maxdist, res.distance);
						}else if(global_ctx.hausdorf_level == 1){
							dist.mindist = std::min(dist.mindist, res.distance - wrapper1->getHausdorffDistance() - wrapper2->getHausdorffDistance());
							dist.maxdist = std::min(dist.maxdist, res.distance + wrapper1->getProxyHausdorffDistance() + wrapper2->getProxyHausdorffDistance());
//								dist.maxdist = std::min(dist.maxdist, res.distance);
						}else if(global_ctx.hausdorf_level == 0){
							dist.maxdist = std::min(dist.maxdist, res.distance);
						}
						//dist.maxdist = std::min(dist.maxdist, res.distance);

						if(global_ctx.verbose>=1)
						{
							log("%ld(%d)\t%ld(%d):\t[%.2f, %.2f]->[%.2f, %.2f] res: [%.2f, %.2f, %.2f]",
									wrapper1->id,res.p1, wrapper2->id,res.p2,
									vp.dist.mindist, vp.dist.maxdist,
									dist.mindist, dist.maxdist,
									res.min_dist, res.distance, res.max_dist);
						}
						vp.dist = dist;
						vox_minmaxdist = min(vox_minmaxdist, (double)dist.maxdist);
						assert(dist.valid());
					}
				}
				// std::cout << min_maxdist << ", " << vox_minmaxdist << std::endl;
				// after each round, some voxels need to be evicted
				ci->distance = update_voxel_pair_list(ci->voxel_pairs, vox_minmaxdist);

				// ci.distance.maxdist = min_maxdist;
				// ci.distance.mindist = min_mindist;

				// std::cout << ci.distance.mindist << ", " << min_mindist << std::endl;


				// std::cout << ci.distance.mindist << ", " << ci.distance.maxdist << std::endl;

				assert(ci->voxel_pairs.size()>0);
				assert(ci->distance.mindist<=ci->distance.maxdist);

			// 	index = ci.dist_id;
			// 	float min_maxdist = gp.min_max_dist[index];
			// 	float min_mindist = gp.min_min_dist[index];
			// 	int valid_cnt = gp.valid_voxel_prefix[index+1] - gp.valid_voxel_prefix[index];
			// 	int base = gp.valid_voxel_prefix[index];

			// 	ci.voxel_pairs.clear();

			// 	for (int k = 0; k < valid_cnt; ++k) {
			// 		int cur_idx = base + k;
			// 		int v1_idx  = gp.valid_voxel_pairs[3 * cur_idx];
			// 		int v2_idx  = gp.valid_voxel_pairs[3 * cur_idx + 1];
			// 		int t       = gp.valid_voxel_pairs[3 * cur_idx + 2];

			// 		ci.voxel_pairs.push_back(
			// 			voxel_pair{
			// 				wrapper1->voxels[v1_idx],
			// 				wrapper2->voxels[v2_idx],
			// 				range{ 
			// 					0.0,  // TODO: just a random one, no use
			// 					0.0
			// 				},
			// 				i,		// which object x
			// 				index, 	// which object y
			// 				t   	// which voxel-pair
			// 			}
			// 		);
			// 	}
			// 	// ci.distance = update_voxel_pair_list(ci.voxel_pairs, min_maxdist);
			// 	ci.distance.maxdist = min_maxdist;
			// 	ci.distance.mindist = min_mindist;
			// }
			// if(global_ctx.verbose>=1){
			// 	log("");
			}
		}
	

		auto ty = std::chrono::high_resolution_clock::now();

		std::cout << "for loop time = " << std::chrono::duration<double>(ty - tx).count() << std::endl;


		auto t0 = std::chrono::high_resolution_clock::now();

		// update the list after processing each LOD
		// evaluate_candidate_lists(candidates, ctx);
		//@@ Trying to make this a kernel
		evaluate_candidate_lists_gpu(candidates, ctx);

		auto t1 = std::chrono::high_resolution_clock::now();

		std::cout << "evaluate = " << std::chrono::duration<double>(t1 - t0).count() << std::endl;

		delete []ctx.results;
		ctx.updatelist_time += logt("updating the candidate lists",start);

		logt("evaluating with lod %d", iter_start, lod);
		log(""); 
		
		// vector<candidate_entry *> new_candidates;
		// for (int i=0;i<tile1_size;i++) {
		// 	HiMesh_Wrapper *wrapper1 = tile1->get_mesh_wrapper(i);
		// 	int candidate_size = prefix_tile[i+1] - prefix_tile[i];
		// 	candidate_entry *ce = new candidate_entry(wrapper1);

		// 	if (gp.confirmed_after[i] == ctx.knn) {

		// 		int begin = prefix_tile[i];
		// 		int end   = prefix_tile[i + 1];

		// 		for (int c = begin; c < end; ++c) {
		// 			if (gp.status_after[c] == CONFIRMED) {
		// 				int obj2_id = compute_obj_2[c];
		// 				wrapper1->report_result(
		// 					tile2->get_mesh_wrapper(obj2_id)
		// 				);
		// 				gp.status_after[c] = REPORTED;
		// 			}
		// 		}
		// 		continue;
		// 	}

		// 	for (int j = prefix_tile[i]; j < prefix_tile[i+1]; ++j) {

		// 		if (gp.status_after[j] == CONFIRMED) {
		// 			int obj2_id = compute_obj_2[j];
		// 			wrapper1->report_result(
		// 				tile2->get_mesh_wrapper(obj2_id)
		// 			);
		// 			ce->candidate_confirmed++;
		// 			gp.status_after[j] = REPORTED;
		// 			continue;
		// 		}
		// 		if (gp.status_after[j] == UNDECIDED) 
		// 		{
		// 			HiMesh_Wrapper *wrapper2 = tile2->get_mesh_wrapper(compute_obj_2[j]);
		// 			candidate_info ci(wrapper2);
		// 			float min_maxdist = all_min_max_dist[j];
		// 			float min_mindist = all_min_min_dist[j];
		// 			// int   valid_cnt   = valid_vp_cnt[j];
		// 			int valid_cnt = gp.valid_voxel_prefix[j+1] - gp.valid_voxel_prefix[j];
		// 			int base = gp.valid_voxel_prefix[j];

		// 			for (int k = 0; k < valid_cnt; ++k) {
		// 				int cur_idx = base + k;
		// 				int v1_idx  = gp.valid_voxel_pairs[3 * cur_idx];
		// 				int v2_idx  = gp.valid_voxel_pairs[3 * cur_idx + 1];
		// 				int t       = gp.valid_voxel_pairs[3 * cur_idx + 2];

		// 				ci.voxel_pairs.push_back(
		// 					voxel_pair{
		// 						wrapper1->voxels[v1_idx],
		// 						wrapper2->voxels[v2_idx],
		// 						range{ 
		// 							out_min[prefix_obj[j] + t], // TODO: just a random one, no use
		// 							out_max[prefix_obj[j] + t]
		// 						},
		// 						i,	// which object x
		// 						j, 	// which object y
		// 						t   // which voxel-pair
		// 					}
		// 				);
		// 			}
		// 			// ci.distance = update_voxel_pair_list(ci.voxel_pairs, min_maxdist);
		// 			ci.distance.maxdist = min_maxdist;
		// 			ci.distance.mindist = min_mindist;
		// 			ce->add_candidate(ci);
		// 		}
		// 	}
		// 	if(ce->candidates.size()>0){
		// 		new_candidates.push_back(ce);
		// 	}else{
		// 		delete ce;
		// 	}
		// }
		// candidates = new_candidates; // replace old with new_candidates

		// // update GPU side arrays
		// update_status_and_confirm(gp.confirmed_after,
		// 						gp.status_after,
		// 						tile1_size,
		// 						prefix_tile[tile1_size]
		// 					);
	}

	ctx.overall_time = tdbase::get_time_elapsed(very_start, false);
	for(int i=0;i<ctx.tile1->num_objects();i++){
		ctx.result_count += ctx.tile1->get_mesh_wrapper(i)->results.size();
//		for(int j=0;j<ctx.tile1->get_mesh_wrapper(i)->results.size();j++){
//			cout<<ctx.tile1->get_mesh_wrapper(i)->id<<"\t"<<ctx.tile1->get_mesh_wrapper(i)->results[j]->id<<endl;
//		}
	}
	ctx.obj_count += min(ctx.tile1->num_objects(),global_ctx.max_num_objects1);
	global_ctx.merge(ctx);
}


size_t count_triangles(candidate_info* ci) {
    size_t total = 0;
    for (auto &vp : ci->voxel_pairs) {

		// std::cout << "## " <<  vp.v1 << ", " << vp.v2 << std::endl;
        total += vp.v1->num_triangles + vp.v2->num_triangles;
    }
    return total;
}


/** 
	Helper function for determining chunks
*/
vector<Chunk> SpatialJoin::build_chunks_by_pair(
        vector<candidate_entry*> &candidates,
        size_t target_pairs_per_chunk,
        size_t target_tris_per_chunk)
{
    vector<Chunk> chunks;
    Chunk cur;

    size_t cur_pairs = 0;
    size_t cur_tris  = 0;  

    for (auto c : candidates) {

        for (auto ci : c->candidates) {

            size_t ci_pairs = ci->voxel_pairs.size();
            size_t ci_tris = count_triangles(ci);

			// std::cout << ci_pairs << ", " << ci_tris << std::endl;

            bool overflow = (cur_pairs + ci_pairs > target_pairs_per_chunk) ||
                			(cur_tris  + ci_tris  > target_tris_per_chunk);

            if (overflow) {
                if (!cur.candidates.empty()) {
                    chunks.push_back(cur);
                    cur = Chunk();
                    cur_pairs = 0;
                    cur_tris  = 0;
                }
            }

            candidate_entry* sub = new candidate_entry();
            sub->mesh_wrapper = c->mesh_wrapper;
            sub->candidates.push_back(ci);


            cur.candidates.push_back(sub);

            cur_pairs += ci_pairs;
            cur_tris  += ci_tris;
        }
	}

    if (!cur.candidates.empty())
        chunks.push_back(cur);

    return chunks;
}


void SpatialJoin::nearest_neighbor_chunking(query_context ctx){
	struct timeval start = get_cur_time();
	struct timeval very_start = get_cur_time();

	// filtering with MBBs to get the candidate list
	std::cout << "Enter mbb_knn_op_gpu_one_kernel" << std::endl;
	vector<candidate_entry *> candidates = mbb_knn_op_gpu_one_kernel(ctx.tile1, ctx.tile2, ctx);
	ctx.index_time += logt("index retrieving", start);

	auto refine_start = std::chrono::high_resolution_clock::now();


	const size_t TARGET_PAIRS = 500'000;

	computer->initialize_refine_stream();

	

	auto tx = std::chrono::high_resolution_clock::now();

	ctx.results_s[0] = (result_container *)allocate_pinned_memory(TARGET_PAIRS * sizeof(result_container));
	ctx.results_s[1] = (result_container *)allocate_pinned_memory(TARGET_PAIRS * sizeof(result_container));

	ctx.data[0] = (float *)allocate_pinned_memory(9 * TARGET_PAIRS * 100 * sizeof(float));
	ctx.data[1] = (float *)allocate_pinned_memory(9 * TARGET_PAIRS * 100 * sizeof(float));

	ctx.hausdorff[0] = 	(float *)allocate_pinned_memory(2 * TARGET_PAIRS * 100 * sizeof(float));
	ctx.hausdorff[1] = 	(float *)allocate_pinned_memory(2 * TARGET_PAIRS * 100 * sizeof(float));

	ctx.offset_size[0] = (size_t *)allocate_pinned_memory(4 * TARGET_PAIRS * sizeof(size_t));
	ctx.offset_size[1] = (size_t *)allocate_pinned_memory(4 * TARGET_PAIRS * sizeof(size_t));

	ctx.location[0] = (int *)allocate_pinned_memory(3 * TARGET_PAIRS * sizeof(int));
	ctx.location[1] = (int *)allocate_pinned_memory(3 * TARGET_PAIRS * sizeof(int));

	auto ty = std::chrono::high_resolution_clock::now();
	std::cout << "allocate pinned memory = " << std::chrono::duration<double>(ty - tx).count() << std::endl;


	// now we start to get the distances with progressive level of details
	for(uint32_t lod:ctx.lods){

		ctx.cur_lod = lod;

		struct timeval iter_start = get_cur_time();
		start = get_cur_time();

		const int pair_num = get_pair_num(candidates);

		// not break temporarily
		if(pair_num==0){
			break;
		}

		size_t candidate_num = get_candidate_num(candidates);
		log("%ld polyhedron has %d candidates %d voxel pairs %.2f voxel pairs per candidate",
				candidates.size(), candidate_num, pair_num, (1.0*pair_num)/candidates.size());

		decode_data(candidates, ctx);

		// pipeline starts here
		auto chunks = build_chunks_by_pair(candidates, TARGET_PAIRS, TARGET_PAIRS * 100);
		int iteration = 0;

		// start a producer thread
		geometry_param next_gp;
        bool has_task = false;
        bool done = false;
        std::mutex mtx;
        std::condition_variable cv;


        std::thread producer([&]() {
            for(int i = 0; i < chunks.size(); i++) {
                int stream_id = i % 2;

                tdbase::synchronize_event(stream_id);
                geometry_param gp = packing_data_stream(chunks[i].candidates, ctx, &addParams, stream_id);

                std::unique_lock<std::mutex> lock(mtx);

                cv.wait(lock, [&]{ return !has_task; });

                next_gp = std::move(gp);
                has_task = true;
                lock.unlock();
                cv.notify_one();
            }
            {
                std::lock_guard<std::mutex> lock(mtx);
                done = true;
            }
            cv.notify_all();
        });

		for(int i=0; i<chunks.size(); ++i) {

			std::cout << "A NEW CHUNK STARTS." << std::endl;
			auto &chunk = chunks[i];

			int stream_id = iteration % 2;
			int prev = 1 - stream_id;

			geometry_param gp;
            {
                std::unique_lock<std::mutex> lock(mtx);
                cv.wait(lock, [&]{ return has_task || done; });
                
                gp = std::move(next_gp);
                has_task = false;

                lock.unlock();
                cv.notify_one();
            }

			
			tdbase::synchronize_stream(stream_id);
			auto gp_nouse = calculate_distance_stream(chunk.candidates, ctx, gp, &addParams, stream_id);

			// now update the distance range with the new distances
			if (iteration > 0) {

				tdbase::synchronize_stream(prev);

				auto &prev_chunk = chunks[i-1];

				int index = 0;
				start = get_cur_time();

				for(int j=0; j<prev_chunk.candidates.size(); ++j) {

					candidate_entry *c = prev_chunk.candidates[j];
					HiMesh_Wrapper *wrapper1 = c->mesh_wrapper;

					for(candidate_info *ci:c->candidates){ 

						HiMesh_Wrapper *wrapper2 = ci->mesh_wrapper;
						
						
						double vox_minmaxdist = DBL_MAX;

						// replace this for-loop
						for(voxel_pair &vp:ci->voxel_pairs){

							result_container res = ctx.results_s[prev][index++];
							// update the distance
							if(vp.v1->num_triangles>0&&vp.v2->num_triangles>0){
								range dist = vp.dist;
								if(lod==ctx.highest_lod()) {
									// now we have a precise distance
									dist.mindist = res.distance;
									dist.maxdist = res.distance;

								}else if(global_ctx.hausdorf_level == 2){
									dist.mindist = std::min(dist.mindist, res.min_dist);
									dist.maxdist = std::min(dist.maxdist, res.max_dist);
		//								dist.maxdist = std::min(dist.maxdist, res.distance);
								}else if(global_ctx.hausdorf_level == 1){
									dist.mindist = std::min(dist.mindist, res.distance - wrapper1->getHausdorffDistance() - wrapper2->getHausdorffDistance());
									dist.maxdist = std::min(dist.maxdist, res.distance + wrapper1->getProxyHausdorffDistance() + wrapper2->getProxyHausdorffDistance());
		//								dist.maxdist = std::min(dist.maxdist, res.distance);
								}else if(global_ctx.hausdorf_level == 0){
									dist.maxdist = std::min(dist.maxdist, res.distance);
								}
								//dist.maxdist = std::min(dist.maxdist, res.distance);

								if(global_ctx.verbose>=1)
								{
									log("%ld(%d)\t%ld(%d):\t[%.2f, %.2f]->[%.2f, %.2f] res: [%.2f, %.2f, %.2f]",
											wrapper1->id,res.p1, wrapper2->id,res.p2,
											vp.dist.mindist, vp.dist.maxdist,
											dist.mindist, dist.maxdist,
											res.min_dist, res.distance, res.max_dist);
								}
								vp.dist = dist;
								vox_minmaxdist = min(vox_minmaxdist, (double)dist.maxdist);
								assert(dist.valid());
							}
						}
						// std::cout << min_maxdist << ", " << vox_minmaxdist << std::endl;
						// after each round, some voxels need to be evicted
						ci->distance = update_voxel_pair_list(ci->voxel_pairs, vox_minmaxdist);

						assert(ci->voxel_pairs.size()>0);
						assert(ci->distance.mindist<=ci->distance.maxdist);
					}
				}
				// tdbase::free_pinned_memory(ctx.results_s[prev]);
			}
			iteration ++;
		}

		// for last chunk
		auto &prev_chunk = chunks.back();
		int index = 0;
		int last = (iteration - 1) % 2;
		tdbase::synchronize_stream(last);

		for(int i=0; i<prev_chunk.candidates.size(); ++i) {

			candidate_entry *c = prev_chunk.candidates[i];
			HiMesh_Wrapper *wrapper1 = c->mesh_wrapper;

			for(candidate_info *ci:c->candidates){ 

				HiMesh_Wrapper *wrapper2 = ci->mesh_wrapper;
				
				
				double vox_minmaxdist = DBL_MAX;

				// replace this for-loop
				for(voxel_pair &vp:ci->voxel_pairs){

					result_container res = ctx.results_s[last][index++];
					// update the distance
					if(vp.v1->num_triangles>0&&vp.v2->num_triangles>0){
						range dist = vp.dist;
						if(lod==ctx.highest_lod()) {
							// now we have a precise distance
							dist.mindist = res.distance;
							dist.maxdist = res.distance;

						}else if(global_ctx.hausdorf_level == 2){
							dist.mindist = std::min(dist.mindist, res.min_dist);
							dist.maxdist = std::min(dist.maxdist, res.max_dist);
//								dist.maxdist = std::min(dist.maxdist, res.distance);
						}else if(global_ctx.hausdorf_level == 1){
							dist.mindist = std::min(dist.mindist, res.distance - wrapper1->getHausdorffDistance() - wrapper2->getHausdorffDistance());
							dist.maxdist = std::min(dist.maxdist, res.distance + wrapper1->getProxyHausdorffDistance() + wrapper2->getProxyHausdorffDistance());
//								dist.maxdist = std::min(dist.maxdist, res.distance);
						}else if(global_ctx.hausdorf_level == 0){
							dist.maxdist = std::min(dist.maxdist, res.distance);
						}
						//dist.maxdist = std::min(dist.maxdist, res.distance);

						if(global_ctx.verbose>=1)
						{
							log("%ld(%d)\t%ld(%d):\t[%.2f, %.2f]->[%.2f, %.2f] res: [%.2f, %.2f, %.2f]",
									wrapper1->id,res.p1, wrapper2->id,res.p2,
									vp.dist.mindist, vp.dist.maxdist,
									dist.mindist, dist.maxdist,
									res.min_dist, res.distance, res.max_dist);
						}
						vp.dist = dist;
						vox_minmaxdist = min(vox_minmaxdist, (double)dist.maxdist);
						assert(dist.valid());
					}
				}
				// std::cout << min_maxdist << ", " << vox_minmaxdist << std::endl;
				// after each round, some voxels need to be evicted
				ci->distance = update_voxel_pair_list(ci->voxel_pairs, vox_minmaxdist);

				assert(ci->voxel_pairs.size()>0);
				assert(ci->distance.mindist<=ci->distance.maxdist);
			}
		}

		producer.join();

		// ==================================================================================================================================

		auto t0 = std::chrono::high_resolution_clock::now();

		// update the list after processing each LOD
		// evaluate_candidate_lists(candidates, ctx);
		//@@ Trying to make this a kernel
		evaluate_candidate_lists_gpu(candidates, ctx);

		auto t1 = std::chrono::high_resolution_clock::now();

		std::cout << "evaluate = " << std::chrono::duration<double>(t1 - t0).count() << std::endl;

		ctx.updatelist_time += logt("updating the candidate lists",start);

		logt("evaluating with lod %d", iter_start, lod);
		log(""); 
	}

	auto refine_end = std::chrono::high_resolution_clock::now();
	std::cout << "k-NN Refinement Stage = " << std::chrono::duration<double>(refine_end - refine_start).count() << std::endl;

	ctx.overall_time = tdbase::get_time_elapsed(very_start, false);
	for(int i=0;i<ctx.tile1->num_objects();i++){
		ctx.result_count += ctx.tile1->get_mesh_wrapper(i)->results.size();
	}
	ctx.obj_count += min(ctx.tile1->num_objects(),global_ctx.max_num_objects1);
	global_ctx.merge(ctx);
}

}
