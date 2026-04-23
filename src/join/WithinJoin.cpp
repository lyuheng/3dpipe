/*
 * WithinJoin.cpp
 *
 *  Created on: Sep 16, 2022
 *      Author: teng
 */

#include "SpatialJoin.h"

namespace tdbase{

Additional_Params addParams_2;

inline void print_candidate_within(candidate_entry *cand){
	if(global_ctx.verbose>=1){
		printf("%ld (%ld candidates)\n", cand->mesh_wrapper->id, cand->candidates.size());
		for(int i=0;i<cand->candidates.size();i++){
			printf("%d:\t%ld\t%ld\n",i,cand->candidates[i]->mesh_wrapper->id,cand->candidates[i]->voxel_pairs.size());
			for(auto &vp:cand->candidates[i]->voxel_pairs){
				printf("\t[%f,%f]\n",vp.dist.mindist,vp.dist.maxdist);
			}
		}
	}
}

vector<candidate_entry *> SpatialJoin::mbb_within_op_gpu_one_kernel(Tile *tile1, Tile *tile2, query_context &ctx) 
{
	vector<candidate_entry *> candidates;
	OctreeNode *tree = tile2->get_octree();
	tile1_size = min(tile1->num_objects(), ctx.max_num_objects1);
	tile2_size = min(tile2->num_objects(), ctx.max_num_objects1);

	this->tile1 = tile1;
	this->tile2 = tile2;

	auto t0 = std::chrono::high_resolution_clock::now();

	// TODO: Is it necessary to load all voxels ?? Fix later...
	auto r1 = prepare_data(tile1_size, tile1);
	DeviceVoxels d_voxels_1 = allocateVoxelsForAll(tile1_size, get<0>(r1), get<1>(r1), get<2>(r1));
	auto r2 = prepare_data(tile2_size, tile2);
	DeviceVoxels d_voxels_2 = allocateVoxelsForAll(tile2_size, get<0>(r2), get<1>(r2), get<2>(r2));


	prefix_tile.assign(tile1_size+1, 0);
	prefix_obj.push_back(0);

	size_t total_voxel_size = 0;
	size_t max_voxel_size = 0;

	// for(int i=0;i<tile1_size;i++) {
	// 	HiMesh_Wrapper *wrapper1 = tile1->get_mesh_wrapper(i);
	// 	vector<pair<int, range>> candidate_ids;
	// 	tree->query_within(&(wrapper1->box), candidate_ids, ctx.within_dist);

	// 	if(candidate_ids.empty()){
	// 		continue;
	// 	}

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

#pragma omp parallel for schedule(dynamic)
	for (int i = 0; i < tile1_size; i++) {

		HiMesh_Wrapper *wrapper1 = tile1->get_mesh_wrapper(i);

		vector<pair<int, range>> candidate_ids;

		tree->query_within(&(wrapper1->box), 
							wrapper1->transform,
							candidate_ids, 
							ctx.within_dist);

		candidate_count[i] = candidate_ids.size();

		all_ids[i].reserve(candidate_ids.size());

		for (auto &p : candidate_ids)
			all_ids[i].push_back(p.first);
	}

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

	auto t1 = std::chrono::high_resolution_clock::now();
	double total_prepare_data_time = std::chrono::duration<double>(t1 - t0).count();
	std::cout << "total_prepare_data_time = " << total_prepare_data_time << std::endl;

	/** how about compute all pairwise distance between V1 and V2 here. */
	t0 = std::chrono::high_resolution_clock::now();
	
	compute_voxel_pair_distance_for_all_within_streaming_pipeline(d_voxels_1, 
														d_voxels_2, 
														prefix_tile, 
														prefix_obj, 
														compute_obj_1, 
														compute_obj_2, 
														total_voxel_size, 
														max_voxel_size, 
														out_min, 
														out_max, 
														all_min_max_dist, 
														all_min_min_dist, 
														valid_vp_cnt, 
														valid_voxel_prefix, 
														valid_voxel_pairs,
														valid_voxel_pairs_dist,
														ctx.within_dist
													);

	t1 = std::chrono::high_resolution_clock::now();
	double total_kernel_time = std::chrono::duration<double>(t1 - t0).count();
	std::cout << "total_kernel_time = " << total_kernel_time << std::endl;

	addParams_2.max_voxel_size = max_voxel_size;
	addParams_2.obj_pair_num = compute_obj_1.size();
	addParams_2.N = tile1_size;
	addParams_2.K = -1;
	addParams_2.total_voxel_size = total_voxel_size;

	t0 = std::chrono::high_resolution_clock::now();

	for (int i=0;i<tile1_size;i++) {

		HiMesh_Wrapper *wrapper1 = tile1->get_mesh_wrapper(i);

		int candidate_size = prefix_tile[i+1] - prefix_tile[i];

		candidate_entry *ce = new candidate_entry(wrapper1);

		for (int j = prefix_tile[i]; j < prefix_tile[i+1]; ++j) {
			HiMesh_Wrapper *wrapper2 = tile2->get_mesh_wrapper(compute_obj_2[j]);
			candidate_info *ci = new candidate_info(wrapper2);
			ci->dist_id = j;

			float min_maxdist = all_min_max_dist[j];
			float min_mindist = all_min_min_dist[j];
			int   valid_cnt   = valid_vp_cnt[j];
			size_t base = valid_voxel_prefix[j];

			// Use object-level distances for comparision first

			if (min_mindist > ctx.within_dist) continue;

			else if (min_maxdist <= ctx.within_dist) {
				wrapper1->report_result(wrapper2);
				continue;
			}
			else {

				bool determined = false;

				for (int k = 0; k < valid_cnt; ++k) {
					size_t cur_idx = base + k;
					int v1_idx  = valid_voxel_pairs[3 * cur_idx];
					int v2_idx  = valid_voxel_pairs[3 * cur_idx + 1];
					int t       = valid_voxel_pairs[3 * cur_idx + 2];

					float min_dist = valid_voxel_pairs_dist[2 * cur_idx];
					float max_dist = valid_voxel_pairs_dist[2 * cur_idx + 1];

					if (min_dist > ctx.within_dist) {
						continue;
					}
					if (max_dist <= ctx.within_dist) {
						determined = true;
						break;
					}

					// if( wrapper2->voxels[v2_idx] == 0) {
					// 	std::cout << "v2_idx = " << v2_idx << std::endl;
					// 	std::cout << "wrapper2->voxels.size() = " << wrapper2->voxels.size() << std::endl;
					// 	exit(-1);
					// }

					if (v1_idx < wrapper1->voxels.size() &&
						v2_idx < wrapper2->voxels.size())
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
				if(determined){
					wrapper1->report_result(wrapper2);
					continue;
				}

				ci->distance.maxdist = min_maxdist;
				ci->distance.mindist = min_mindist;

				if(ci->voxel_pairs.size()>0)
				{
					ce->add_candidate(ci);
				}
			}
		}
		if(ce->candidates.size()>0){
			candidates.push_back(ce);
		}else{
			delete ce;
		}
	}
	t1 = std::chrono::high_resolution_clock::now();
	double total_candidate_time = std::chrono::duration<double>(t1 - t0).count();
	std::cout << "total_candidate_time = " << total_candidate_time << std::endl;

	return candidates;
}

vector<candidate_entry *> SpatialJoin::mbb_within(Tile *tile1, Tile *tile2, query_context &ctx){
	vector<candidate_entry *> candidates;
	OctreeNode *tree = tile2->get_octree();
	size_t tile1_size = min(tile1->num_objects(), ctx.max_num_objects1);

#pragma omp parallel for
	for(int i=0;i<tile1_size;i++){
		vector<pair<int, range>> candidate_ids;
		HiMesh_Wrapper *wrapper1 = tile1->get_mesh_wrapper(i);
		tree->query_within(&(wrapper1->box), wrapper1->transform, candidate_ids, ctx.within_dist);
		if(candidate_ids.empty()){
			continue;
		}

		candidate_entry *ce = new candidate_entry(wrapper1);
		for(pair<int, range> &p:candidate_ids){
			HiMesh_Wrapper *wrapper2 = tile2->get_mesh_wrapper(p.first);
			candidate_info *ci = new candidate_info(wrapper2);
			bool determined = false;
			float min_maxdist = DBL_MAX;
			for(Voxel *v1:wrapper1->voxels){
				for(Voxel *v2:wrapper2->voxels){
					range dist_vox = v1->distance(*v2);
					// must not within
					if(dist_vox.mindist>ctx.within_dist){
						continue;
					}
					// must be within
					if(dist_vox.maxdist<=ctx.within_dist){
						determined = true;
						break;
					}
					// the faces in those voxels need be further evaluated
					ci->voxel_pairs.push_back(voxel_pair(v1, v2, dist_vox));
					min_maxdist = min(min_maxdist, dist_vox.maxdist);
				}
				if(determined){
					break;
				}
			}

			// determined with the voxel evaluation
			if(determined){
				wrapper1->report_result(wrapper2);
				//delete ci;
				continue;
			}

			// otherwise, for further evaluation
			ci->distance = update_voxel_pair_list(ci->voxel_pairs, min_maxdist);
			// some voxel pairs need to be further evaluated
			if(ci->voxel_pairs.size()>0){
				ce->add_candidate(ci);
			}else{
				delete ci;
			}
		}
		// save the candidate list
		if(ce->candidates.size()>0){
#pragma omp critical
			candidates.push_back(ce);
		}else{
			delete ce;
		}
		candidate_ids.clear();
	}

	// int ttl_cand = 0;
	// int ttl_cand_2 = 0;
	// for (auto c: candidates) {
	// 	ttl_cand_2 += c->candidates.size();
	// 	for(auto &d: c->candidates) {
	// 		ttl_cand += d.voxel_pairs.size();
	// 	}
	// }
	// std::cout << "#### ttl_cand = " << ttl_cand << std::endl;
	// std::cout << "#### ttl_cand_2 = " << ttl_cand_2 << std::endl;

	return candidates;
}

/*
 * the main function for getting the object within a specified distance
 *
 * */
void SpatialJoin::within(query_context ctx){
	struct timeval start = get_cur_time();
	struct timeval very_start = get_cur_time();
	// filtering with MBBs to get the candidate list
	vector<candidate_entry *> candidates = mbb_within_op_gpu_one_kernel(ctx.tile1, ctx.tile2, ctx);
	ctx.index_time += get_time_elapsed(start, false);
	logt("comparing mbbs with %d candidate pairs", start, get_candidate_num(candidates));

	// now we start to get the distances with progressive level of details
	for(uint32_t lod:ctx.lods){
		ctx.cur_lod = lod;
		struct timeval iter_start = get_cur_time();
		const int pair_num = get_pair_num(candidates);
		if(pair_num==0){
			break;
		}
		size_t candidate_num = get_candidate_num(candidates);
		log("%ld polyhedron has %d candidates %d voxel pairs %.2f voxel pairs per candidate",
				candidates.size(), candidate_num, pair_num, (1.0*pair_num)/candidates.size());

		// do the computation
		calculate_distance(candidates, ctx, &addParams_2);

		// now update the candidate list with the new latest information
		int index = 0;
		start = get_cur_time();


		// ========================= original version =========================
		
// 		for(auto ce_iter=candidates.begin();ce_iter!=candidates.end();){
// 			HiMesh_Wrapper *wrapper1 = (*ce_iter)->mesh_wrapper;
// 			//print_candidate_within(*ce_iter);

// 			for(auto ci_iter=(*ce_iter)->candidates.begin();ci_iter!=(*ce_iter)->candidates.end();){

// 				bool determined = false;
// 				HiMesh_Wrapper *wrapper2 = (*ci_iter)->mesh_wrapper;
// 				if(ctx.use_aabb){
// 					range dist = (*ci_iter)->distance;
// 					result_container res = ctx.results[index++];
// 					if(lod==ctx.highest_lod()){
// 						// now we have a precise distance
// 						dist.mindist = res.distance;
// 						dist.maxdist = res.distance;
// 					}else{
// 						dist.maxdist = std::min(dist.maxdist, res.distance);
// 						dist.mindist = std::max(dist.mindist, dist.maxdist-wrapper1->getHausdorffDistance()-wrapper2->getHausdorffDistance());
// 					}
// 					if(global_ctx.verbose>=1){
// 						log("%ld\t%ld:\t[%.2f, %.2f]->[%.2f, %.2f]",wrapper1->id, wrapper2->id,
// 								(*ci_iter)->distance.mindist, (*ci_iter)->distance.maxdist,
// 								dist.mindist, dist.maxdist);
// 					}
// 					(*ci_iter)->distance = dist;
// 					if(dist.maxdist<=ctx.within_dist){
// 						// the distance is close enough
// 						wrapper1->report_result(wrapper2);
// 						determined = true;
// 					}else if(dist.mindist > ctx.within_dist){
// 						// not possible
// 						determined = true;
// 					}
// 				}else{ // end aabb

// 					for(auto vp_iter = (*ci_iter)->voxel_pairs.begin();vp_iter!=(*ci_iter)->voxel_pairs.end();){
// 						result_container res = ctx.results[index++];
// 						//cout<<vp_iter->v1->num_triangles<<"  "<<res.p1<<" "<<vp_iter->v2->num_triangles<<" "<<res.p2<<" "<<res.distance<<endl;
// 						// update the distance
// 						if(!determined && vp_iter->v1->num_triangles>0&&vp_iter->v2->num_triangles>0){
// 							range dist = vp_iter->dist;
// 							if(lod==ctx.highest_lod()){
// 								// now we have a precise distance
// 								dist.mindist = res.distance;
// 								dist.maxdist = res.distance;
// 							}else if(global_ctx.hausdorf_level == 2){
// 								dist.mindist = std::min(dist.mindist, res.min_dist);
// 								dist.maxdist = std::min(dist.maxdist, res.max_dist);
// //								dist.maxdist = std::min(dist.maxdist, res.distance);
// 							}else if(global_ctx.hausdorf_level == 1){
// 								dist.mindist = std::min(dist.mindist, res.distance - wrapper1->getHausdorffDistance() - wrapper2->getHausdorffDistance());
// 								dist.maxdist = std::min(dist.maxdist, res.distance + wrapper1->getProxyHausdorffDistance() + wrapper2->getProxyHausdorffDistance());
// //								dist.maxdist = std::min(dist.maxdist, res.distance);
// 							}else if(global_ctx.hausdorf_level == 0){
// 								dist.maxdist = std::min(dist.maxdist, res.distance);
// 							}
// 							//dist.maxdist = std::min(dist.maxdist, res.distance);

// 							if(global_ctx.verbose>=1) {
// 								log("%ld\t%ld:[%.2f, %.2f]->[%.2f, %.2f]",wrapper1->id, wrapper2->id,
// 										(*ci_iter)->distance.mindist, (*ci_iter)->distance.maxdist,
// 										dist.mindist, dist.maxdist);
// 							}
// 							vp_iter->dist = dist;

// 							// one voxel pair is close enough
// 							if(dist.maxdist<=ctx.within_dist){
// 								determined = true;
// 								wrapper1->report_result(wrapper2);
// 							}
// 						}
// 						// too far, should be removed from the voxel pair list
// 						if(vp_iter->dist.mindist>ctx.within_dist){
// 							(*ci_iter)->voxel_pairs.erase(vp_iter);
// 						}else{
// 							vp_iter++;
// 						}
// 					}
// 				}

// 				if(determined || (*ci_iter)->voxel_pairs.size()==0){
// 					// must closer than or farther than
// 					//delete *ci_iter;
// 					(*ce_iter)->candidates.erase(ci_iter);
// 				}else{
// 					ci_iter++;
// 				}
// 			}
// 			if((*ce_iter)->candidates.size()==0){
// 				delete (*ce_iter);
// 				candidates.erase(ce_iter);
// 			}else{
// 				//print_candidate_within(*ce_iter);
// 				ce_iter++;
// 			}
// 		}


		// ========================= parallelized version =========================
		std::vector<size_t> ce_begin(candidates.size());
		size_t global_index = 0;
		for (size_t i = 0; i < candidates.size(); ++i)
		{
			ce_begin[i] = global_index;
			candidate_entry* ce = candidates[i];
			size_t count = 0;
			for (auto cand : ce->candidates)
				count += cand->voxel_pairs.size();
			global_index += count;
		}

#pragma omp parallel for schedule(dynamic)
		for (int ci = 0; ci < (int)candidates.size(); ++ci)
		{
			candidate_entry* ce = candidates[ci];
			HiMesh_Wrapper* wrapper1 = ce->mesh_wrapper;

			size_t local_index = ce_begin[ci];
			size_t offset = 0;

			std::vector<candidate_info *> new_candidates;

			for (auto cand : ce->candidates)
			{
				bool determined = false;
				HiMesh_Wrapper* wrapper2 = cand->mesh_wrapper;

				std::vector<voxel_pair> new_voxels;

				for (auto& vp : cand->voxel_pairs)
				{
					result_container res = ctx.results[local_index + offset++];
					range dist = vp.dist;

					if (!determined && vp.v1->num_triangles > 0 && vp.v2->num_triangles > 0)
					{
						if (lod == ctx.highest_lod())
						{
							dist.mindist = res.distance;
							dist.maxdist = res.distance;
						}
						else if (global_ctx.hausdorf_level == 2)
						{
							dist.mindist = std::min(dist.mindist, res.min_dist);
							dist.maxdist = std::min(dist.maxdist, res.max_dist);
						}
						else if (global_ctx.hausdorf_level == 1)
						{
							dist.mindist = std::min(dist.mindist, res.distance - wrapper1->getHausdorffDistance() - wrapper2->getHausdorffDistance());
							dist.maxdist = std::min(dist.maxdist, res.distance + wrapper1->getProxyHausdorffDistance() + wrapper2->getProxyHausdorffDistance());
						}
						else
						{
							dist.maxdist = std::min(dist.maxdist, res.distance);
						}

						vp.dist = dist;

						if (dist.maxdist <= ctx.within_dist)
						{
							determined = true;
							wrapper1->report_result(wrapper2);
						}
					}

					if (dist.mindist <= ctx.within_dist)
					{
						new_voxels.push_back(vp);
					}
				}

				cand->voxel_pairs.swap(new_voxels);

				if (!determined && !cand->voxel_pairs.empty())
				{
					new_candidates.push_back(cand);
				}
			}

			ce->candidates.swap(new_candidates);
		}
		// ============= parallel version end


		
		delete []ctx.results;
		ctx.updatelist_time += logt("updating the candidate lists",start);

		logt("evaluating with lod %d", iter_start, lod);
		log("");
	}
	ctx.overall_time = tdbase::get_time_elapsed(very_start, false);
	for(int i=0;i<ctx.tile1->num_objects();i++){
		ctx.result_count += ctx.tile1->get_mesh_wrapper(i)->results.size();
		for(int j=0;j<ctx.tile1->get_mesh_wrapper(i)->results.size();j++){
			//cout<<ctx.tile1->get_mesh_wrapper(i)->id<<"\t"<<ctx.tile1->get_mesh_wrapper(i)->results[j]->id<<endl;
		}
	}
	ctx.obj_count += min(ctx.tile1->num_objects(),global_ctx.max_num_objects1);
	global_ctx.merge(ctx);
}


void SpatialJoin::refine_aggregation(vector<candidate_entry *> &candidates, 
									query_context &ctx,
									int prev
								)
{
	int index = 0;

	for(int j=0; j<candidates.size(); ++j) {

		candidate_entry *c = candidates[j];
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
					if(ctx.cur_lod==ctx.highest_lod()) {
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
}


void SpatialJoin::within_chunking(query_context ctx){
	struct timeval start = get_cur_time();
	struct timeval very_start = get_cur_time();
	// filtering with MBBs to get the candidate list
	vector<candidate_entry *> candidates = mbb_within_op_gpu_one_kernel(ctx.tile1, ctx.tile2, ctx);
	ctx.index_time += get_time_elapsed(start, false);
	logt("comparing mbbs with %d candidate pairs", start, get_candidate_num(candidates));

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
		const int pair_num = get_pair_num(candidates);
		if(pair_num==0){
			break;
		}
		size_t candidate_num = get_candidate_num(candidates);
		log("%ld polyhedron has %d candidates %d voxel pairs %.2f voxel pairs per candidate",
				candidates.size(), candidate_num, pair_num, (1.0*pair_num)/candidates.size());

		// decode data to current lod
		decode_data(candidates, ctx);

		// pipeline starts here
		auto chunks = build_chunks_by_pair(candidates, TARGET_PAIRS, TARGET_PAIRS * 100);
		int iteration = 0;

		// start a producer thread
		geometry_param next_gp;
        atomic<bool> has_task = false;
        bool done = false;
        std::mutex mtx;
        std::condition_variable cv;

        std::thread producer([&]() {
            for(int i = 0; i < chunks.size(); i++) {
                int stream_id = i % 2;

                tdbase::synchronize_event(stream_id);

                geometry_param gp = packing_data_stream(chunks[i].candidates, ctx, &addParams_2, stream_id);

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
			auto gp_nouse = calculate_distance_stream(chunk.candidates, ctx, gp, &addParams_2, stream_id);

			// now update the distance range with the new distances
			if (iteration > 0) {
				tdbase::synchronize_stream(prev);
				auto &prev_chunk = chunks[i-1];
				refine_aggregation(prev_chunk.candidates, ctx, prev);
			}
			iteration ++;
		}
		auto &prev_chunk = chunks.back();
		int last = (iteration - 1) % 2;
		tdbase::synchronize_stream(last);
		refine_aggregation(prev_chunk.candidates, ctx, last);

		producer.join();

		// obtain partial results

		auto tx = std::chrono::high_resolution_clock::now();

		for (auto *c : candidates) {
			HiMesh_Wrapper *wrapper1 = c->mesh_wrapper;

			for (auto it = c->candidates.begin(); it != c->candidates.end(); ) {
				candidate_info *ci = *it;
				HiMesh_Wrapper *wrapper2 = ci->mesh_wrapper;

				const double min_d = ci->distance.mindist;
				const double max_d = ci->distance.maxdist;

				if (max_d <= ctx.within_dist) {
					wrapper1->results.push_back(wrapper2);
					it = c->candidates.erase(it);
					delete ci;
				}
				else if (min_d > ctx.within_dist) {
					it = c->candidates.erase(it);
					delete ci;
				}
				else {
					++it;
				}
			}
		}
		auto ty = std::chrono::high_resolution_clock::now();
		std::cout << "within D evaluate = " << std::chrono::duration<double>(ty - tx).count() << std::endl;


		logt("evaluating with lod %d", iter_start, lod);
		log("");
	}
	
	auto refine_end = std::chrono::high_resolution_clock::now();
	std::cout << "Within Refinement Stage = " << std::chrono::duration<double>(refine_end - refine_start).count() << std::endl;

	ctx.overall_time = tdbase::get_time_elapsed(very_start, false);
	for(int i=0;i<ctx.tile1->num_objects();i++){
		ctx.result_count += ctx.tile1->get_mesh_wrapper(i)->results.size();
		for(int j=0;j<ctx.tile1->get_mesh_wrapper(i)->results.size();j++){
			//cout<<ctx.tile1->get_mesh_wrapper(i)->id<<"\t"<<ctx.tile1->get_mesh_wrapper(i)->results[j]->id<<endl;
		}
	}
	ctx.obj_count += min(ctx.tile1->num_objects(),global_ctx.max_num_objects1);
	global_ctx.merge(ctx);
}

}
