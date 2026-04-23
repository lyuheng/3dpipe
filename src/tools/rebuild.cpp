#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/convex_hull_3.h>
#include <CGAL/IO/polygon_mesh_io.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using Kernel = CGAL::Exact_predicates_inexact_constructions_kernel;
using Point  = Kernel::Point_3;
using Mesh   = CGAL::Surface_mesh<Point>;

static bool read_off_vertices_only(const std::string& filename,
                                   std::vector<Point>& points)
{
    std::ifstream in(filename);
    if (!in) {
        std::cerr << "Cannot open input file: " << filename << "\n";
        return false;
    }

    auto next_data_line = [&]() -> std::string {
        std::string line;
        while (std::getline(in, line)) {
            std::size_t first = line.find_first_not_of(" \t\r\n");
            if (first == std::string::npos) continue;
            if (line[first] == '#') continue;
            return line;
        }
        return {};
    };

    std::string line = next_data_line();

    std::string counts_line;
    if (line.size() >= 3 && line.substr(0, 3) == "OFF") {
        std::string after = line.substr(3);
        if (after.find_first_not_of(" \t\r\n") != std::string::npos) {
            counts_line = after;             // "OFF4780 6240 0" 情况
        } else {
            counts_line = next_data_line();  // "OFF\n4780 6240 0" 情况
        }
    } else {
        std::cerr << "Not a valid OFF file (missing OFF header)\n";
        return false;
    }

    if (counts_line.empty()) {
        std::cerr << "Missing OFF counts line\n";
        return false;
    }

    std::istringstream counts(counts_line);
    std::size_t n_vertices = 0, n_faces = 0, n_edges = 0;
    if (!(counts >> n_vertices >> n_faces >> n_edges)) {
        std::cerr << "Failed to parse OFF counts line\n";
        return false;
    }

    points.clear();
    points.reserve(n_vertices);

    for (std::size_t i = 0; i < n_vertices; ++i) {
        line = next_data_line();
        if (line.empty()) {
            std::cerr << "Unexpected EOF while reading vertices\n";
            return false;
        }

        std::istringstream iss(line);
        double x, y, z;
        if (!(iss >> x >> y >> z)) {
            std::cerr << "Failed to parse vertex " << i << "\n";
            return false;
        }
        points.emplace_back(x, y, z);
    }

    return true;
}

int main(int argc, char** argv)
{
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " input.off output.off\n";
        return 1;
    }

    std::vector<Point> points;
    if (!read_off_vertices_only(argv[1], points)) {
        return 1;
    }

    if (points.size() < 4) {
        std::cerr << "Need at least 4 points for a 3D convex hull\n";
        return 1;
    }

    Mesh hull;
    CGAL::convex_hull_3(points.begin(), points.end(), hull);

    if (!CGAL::IO::write_polygon_mesh(argv[2], hull,
                                  CGAL::parameters::stream_precision(17))) {
        std::cerr << "Failed to write output OFF\n";
        return 1;
    }

    std::cout << "Wrote convex hull to " << argv[2] << "\n";
    return 0;
}