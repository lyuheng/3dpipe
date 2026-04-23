/*
 * data_generator.cpp
 *
 *  Created on: Nov 27, 2019
 *      Author: teng
 *
 *  generate testing data by duplicating some given examples
 *
 */

#include <fstream>
#include <tuple>
#include <filesystem>
#include <vector>
#include <string>
#include "himesh.h"
#include "tile.h"
#include "popl.h"

#include "check_inside.h"

using namespace tdbase;
using namespace std;
using namespace popl;

namespace fs = std::filesystem;

#define JOB_THRESHOLD 2000

const int buffer_size = 50*(1<<20);
HiMesh *vessel = NULL;
HiMesh *nuclei = NULL;

map<int, HiMesh *> vessels;
map<int, HiMesh *> nucleis;

aab nuclei_box;
aab vessel_box;

bool *vessel_taken;
int total_slots = 0;

int num_nuclei_per_vessel = 100;
int num_vessel = 50;
int voxel_size = 100;

vector<HiMesh_Wrapper *> generated_nucleis;
vector<HiMesh_Wrapper *> generated_vessels;

bool multi_lods = false;
bool allow_intersection = false;

// 空间范围，根据物体数量自动估算一个合理的范围
float space_range = 10.0f; // 100 = ModelNet_train.dt, 1= ModelNet_train_2.dt, 10=ModelNet_test.dt
int REPEATED = 100;


void load_prototype(const char *nuclei_path, const char *vessel_path){

	// load the vessel
	if(multi_lods){
		char path[256];
		for(int lod=100;lod>=20;lod-=20){
			sprintf(path, "%s_%d.off", vessel_path, lod);
			log("loading %s",path);
			HiMesh *m = read_mesh(path);
			assert(m);
			vessels[lod] = m;
			if(lod == 100){
				vessel = m;
			}
		}
	}else {
		vessel = read_mesh(vessel_path);
	}
	assert(vessel);
	aab mbb = vessel->get_mbb();
	vessel_box = vessel->shift(-mbb.low[0], -mbb.low[1], -mbb.low[2]);
}

HiMesh_Wrapper *organize_data(HiMesh *mesh, float shift[3]){

	HiMesh *local_mesh = mesh->clone_mesh();
	local_mesh->shift(shift[0], shift[1], shift[2]);

	HiMesh_Wrapper *wr = new HiMesh_Wrapper(local_mesh);
	
	return wr;
}

HiMesh_Wrapper *organize_data(map<int, HiMesh *> &meshes, float shift[3]){

	map<int, HiMesh *> local_meshes;

	for(auto m:meshes){
		HiMesh *nmesh = m.second->clone_mesh();
		nmesh->shift(shift[0], shift[1], shift[2]);
		local_meshes[m.first] = nmesh;
	}

	HiMesh_Wrapper *wr = new HiMesh_Wrapper(local_meshes);
	return wr;
}

bool inside_polyhedron(string path) {

	std::vector<Vec3> vertices;
    std::vector<TriangleX> triangles;
    Vec3 bbmin, bbmax;

    if (!read_off(path, vertices, triangles, bbmin, bbmax)) {
		exit(-1);
    }

    Vec3 center{
        0.5 * (bbmin.x + bbmax.x),
        0.5 * (bbmin.y + bbmax.y),
        0.5 * (bbmin.z + bbmax.z)
    };

	auto loc = pointInClosedMesh(center, triangles);

    if (loc == PointLocation::Inside || loc == PointLocation::Boundary) {
		return true;
    }
	else {
        return false;
    }
}


long global_generated = 0;

int main(int argc, char **argv){
	string nuclei_pt;
	string vessel_pt;
	string output_path;
    string train_root;

	OptionParser op("Simulator");
	auto help_option          = op.add<Switch>("h", "help",              "produce help message");
	auto hausdorff_option     = op.add<Switch>("",  "hausdorff",         "enable Hausdorff distance calculation", &HiMesh::use_hausdorff);
	auto multi_lods_option    = op.add<Switch>("m", "multi_lods",        "the input are polyhedrons in multiple files", &multi_lods);
	auto allow_intersection_option = op.add<Switch>("i", "allow_intersection", "allow the nuclei to intersect with other nuclei or vessel", &allow_intersection);
	op.add<Value<string>>("d", "datadir", "path to ModelNet rebuild root dir", "ModelNet_rebuild", &train_root);
	op.add<Value<string>>("o", "output",  "prefix of the output files",         "default",   &output_path);
	op.add<Value<int>>("",  "nv",      "number of vessels",                  50,          &num_vessel);
	op.add<Value<int>>("",  "nu",      "number of nucleis per vessel",        100,         &num_nuclei_per_vessel);
	op.add<Value<int>>("",  "vs",      "number of vertices in each voxel",    100,         &voxel_size);
	op.add<Value<int>>("",  "verbose", "verbose level",                       0,           &global_ctx.verbose);
	op.add<Value<uint32_t>>("r", "sample_rate", "sampling rate for Hausdorff distance calculation", 30, &HiMesh::sampling_rate);

	op.parse(argc, argv);

	struct timeval start = get_cur_time();

	char vessel_output[256];
	char nuclei_output[256];
	char config[100];
	// sprintf(config, "nv%d_nu%d_vs%d_r%d",
	// 		num_vessel, num_nuclei_per_vessel, voxel_size,
	// 		HiMesh::sampling_rate);
	sprintf(vessel_output, "ModelNet_test.dt");
	// sprintf(nuclei_output,  "%s_n_%s.dt", output_path.c_str(), config);

	// int dim1 = (int)(pow((float)num_vessel, (float)1.0/3) + 0.5);
	// int dim2 = dim1;
	// int dim3 = num_vessel / (dim1 * dim2);
	// int x_dim, y_dim, z_dim;
	// if (vessel_box.high[0]-vessel_box.low[0] > vessel_box.high[1]-vessel_box.low[1] &&
	//     vessel_box.high[0]-vessel_box.low[0] > vessel_box.high[2]-vessel_box.low[2]) {
	// 	x_dim = min(dim1, dim3); y_dim = max(dim1, dim3); z_dim = max(dim1, dim3);
	// } else if (vessel_box.high[1]-vessel_box.low[1] > vessel_box.high[0]-vessel_box.low[0] &&
	//            vessel_box.high[1]-vessel_box.low[1] > vessel_box.high[2]-vessel_box.low[2]) {
	// 	y_dim = min(dim1, dim3); x_dim = max(dim1, dim3); z_dim = max(dim1, dim3);
	// } else {
	// 	z_dim = min(dim1, dim3); x_dim = max(dim1, dim3); y_dim = max(dim1, dim3);
	// }

    vector<string> off_files;
    for (const auto &category_entry : fs::directory_iterator(train_root)) {
        if (!category_entry.is_directory()) continue;
        fs::path train_dir = category_entry.path() / "test";
        if (!fs::exists(train_dir) || !fs::is_directory(train_dir)) continue;
        for (const auto &file_entry : fs::directory_iterator(train_dir)) {
            if (file_entry.path().extension() == ".off") {
                off_files.push_back(file_entry.path().string());
            }
        }
    }
    log("Found %zu .off files under test/", off_files.size());
    logt("scan files", start);

     ofstream *vessel_os = new std::ofstream(vessel_output, std::ios::out | std::ios::binary);

    char type = (char)COMPRESSED;
    // nuclei_os->write(&type, 1);
    vessel_os->write(&type, 1);


    // 先收集所有mesh，计算grid尺寸
    int total_objects = (int)off_files.size() * REPEATED;
    int dim = (int)(ceil(cbrt((float)total_objects)));  // 每个维度的grid数

    // 全局计数器
    int global_idx = 0;

    for (int idx = 0; idx < (int)off_files.size(); idx++) {
        const string &vessel_pt = off_files[idx];
        std::cout << vessel_pt << std::endl;

        load_prototype(nuclei_pt.c_str(), vessel_pt.c_str());
        logt("load prototype files", start);

		bool inside = inside_polyhedron(vessel_pt);

        for (int rep = 0; rep < REPEATED; rep++) {
            HiMesh *mesh = vessel->clone_mesh();

            aab mbb = mesh->get_mbb();
            mesh->shift(-mbb.low[0], -mbb.low[1], -mbb.low[2]);

            mbb = mesh->get_mbb();
            float max_side = max({mbb.high[0] - mbb.low[0],
                                mbb.high[1] - mbb.low[1],
                                mbb.high[2] - mbb.low[2]});

            // float shrink_val = max_side * (0.5 + tdbase::get_rand_double() * 0.5);
            mesh->shrink(max_side/10); 
	

            // 根据 global_idx 计算在 grid 中的位置
            int gi = global_idx / (dim * dim);
            int gj = (global_idx % (dim * dim)) / dim;
            int gk = global_idx % dim;

            float shift[3] = {
                gi * space_range,
                gj * space_range,
                gk * space_range
            };
            mesh->shift(shift[0], shift[1], shift[2]);

			HiMesh_Wrapper *wr = new HiMesh_Wrapper(mesh);
			wr->find_closest_mbb_point(inside);

            generated_vessels.push_back(wr);
            global_idx++;
        }
    }

    logt("%zu vessels loaded", start, generated_vessels.size());
    Tile *vessel_tile = new Tile(generated_vessels);
    vessel_tile->dump_compressed(vessel_output, vessel_os);
    delete vessel_tile;
    generated_vessels.clear();

	// nuclei_os->close();
	vessel_os->close();
}