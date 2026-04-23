#include <cmath>
#include <vector>
#include <limits>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <array>
#include <cctype>

struct Vec3 {
    double x, y, z;
};

static inline Vec3 operator+(const Vec3& a, const Vec3& b) { return {a.x+b.x, a.y+b.y, a.z+b.z}; }
static inline Vec3 operator-(const Vec3& a, const Vec3& b) { return {a.x-b.x, a.y-b.y, a.z-b.z}; }
static inline Vec3 operator*(const Vec3& a, double s) { return {a.x*s, a.y*s, a.z*s}; }

static inline double dot(const Vec3& a, const Vec3& b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

static inline Vec3 cross(const Vec3& a, const Vec3& b) {
    return {
        a.y*b.z - a.z*b.y,
        a.z*b.x - a.x*b.z,
        a.x*b.y - a.y*b.x
    };
}

static inline double norm2(const Vec3& a) {
    return dot(a, a);
}

struct TriangleX {
    Vec3 a, b, c;
};

enum class PointLocation {
    Outside,
    Boundary,
    Inside
};

static bool pointOnTriangle(const Vec3& p, const TriangleX& t, double eps = 1e-9) {
    Vec3 n = cross(t.b - t.a, t.c - t.a);
    double area2 = std::sqrt(norm2(n));
    if (area2 < eps) return false;

    double dist = dot(p - t.a, n) / area2;
    if (std::abs(dist) > eps) return false;

    Vec3 v0 = t.c - t.a;
    Vec3 v1 = t.b - t.a;
    Vec3 v2 = p - t.a;

    double d00 = dot(v0, v0);
    double d01 = dot(v0, v1);
    double d11 = dot(v1, v1);
    double d20 = dot(v2, v0);
    double d21 = dot(v2, v1);

    double denom = d00 * d11 - d01 * d01;
    if (std::abs(denom) < eps) return false;

    double v = (d11 * d20 - d01 * d21) / denom;
    double w = (d00 * d21 - d01 * d20) / denom;
    double u = 1.0 - v - w;

    return u >= -eps && v >= -eps && w >= -eps;
}

static bool rayIntersectsTriangle(
    const Vec3& orig,
    const Vec3& dir,
    const TriangleX& tri,
    double& tHit,
    double eps = 1e-9)
{
    Vec3 e1 = tri.b - tri.a;
    Vec3 e2 = tri.c - tri.a;

    Vec3 pvec = cross(dir, e2);
    double det = dot(e1, pvec);

    if (std::abs(det) < eps) return false;

    double invDet = 1.0 / det;
    Vec3 tvec = orig - tri.a;

    double u = dot(tvec, pvec) * invDet;
    if (u < -eps || u > 1.0 + eps) return false;

    Vec3 qvec = cross(tvec, e1);
    double v = dot(dir, qvec) * invDet;
    if (v < -eps || u + v > 1.0 + eps) return false;

    tHit = dot(e2, qvec) * invDet;
    return tHit > eps;
}

static PointLocation pointInClosedMesh(
    const Vec3& p,
    const std::vector<TriangleX>& mesh,
    double eps = 1e-9)
{
    for (const auto& tri : mesh) {
        if (pointOnTriangle(p, tri, eps)) {
            return PointLocation::Boundary;
        }
    }

    // 射线方向不要和坐标轴完全平行
    Vec3 dir{1.0, 0.37, 0.19};

    int hitCount = 0;
    for (const auto& tri : mesh) {
        double tHit = 0.0;
        if (rayIntersectsTriangle(p, dir, tri, tHit, eps)) {
            ++hitCount;
        }
    }

    return (hitCount % 2 == 1) ? PointLocation::Inside : PointLocation::Outside;
}

static bool read_off(
    const std::string& path,
    std::vector<Vec3>& vertices,
    std::vector<TriangleX>& triangles,
    Vec3& bbmin,
    Vec3& bbmax)
{
    std::ifstream in(path);
    if (!in) {
        std::cerr << "Cannot open file: " << path << "\n";
        return false;
    }

    std::string token;
    in >> token;
    if (token != "OFF") {
        std::cerr << "Not an OFF file.\n";
        return false;
    }

    size_t nv = 0, nf = 0, ne = 0;
    in >> nv >> nf >> ne;
    if (!in) {
        std::cerr << "Failed to read OFF header.\n";
        return false;
    }

    vertices.resize(nv);
    bbmin = {std::numeric_limits<double>::infinity(),
             std::numeric_limits<double>::infinity(),
             std::numeric_limits<double>::infinity()};
    bbmax = {-std::numeric_limits<double>::infinity(),
             -std::numeric_limits<double>::infinity(),
             -std::numeric_limits<double>::infinity()};

    for (size_t i = 0; i < nv; ++i) {
        double x, y, z;
        in >> x >> y >> z;
        if (!in) {
            std::cerr << "Failed to read vertex " << i << "\n";
            return false;
        }
        vertices[i] = {x, y, z};

        bbmin.x = std::min(bbmin.x, x);
        bbmin.y = std::min(bbmin.y, y);
        bbmin.z = std::min(bbmin.z, z);

        bbmax.x = std::max(bbmax.x, x);
        bbmax.y = std::max(bbmax.y, y);
        bbmax.z = std::max(bbmax.z, z);
    }

    triangles.reserve(nf * 2);

    for (size_t i = 0; i < nf; ++i) {
        std::string line;
        std::getline(in, line); // 吃掉行尾
        if (line.empty()) {
            std::getline(in, line);
        }

        while (!line.empty() && std::isspace(static_cast<unsigned char>(line[0]))) {
            line.erase(line.begin());
        }

        if (line.empty()) {
            --i;
            continue;
        }

        std::istringstream iss(line);
        int n = 0;
        iss >> n;
        if (!iss || n < 3) {
            std::cerr << "Invalid face at index " << i << "\n";
            return false;
        }

        std::vector<int> idx(n);
        for (int k = 0; k < n; ++k) {
            iss >> idx[k];
            if (!iss) {
                std::cerr << "Failed to read face indices at face " << i << "\n";
                return false;
            }
        }

        // 扇形三角化： (0,1,2), (0,2,3), ...
        for (int k = 1; k + 1 < n; ++k) {
            int i0 = idx[0];
            int i1 = idx[k];
            int i2 = idx[k + 1];

            if (i0 < 0 || i1 < 0 || i2 < 0 ||
                static_cast<size_t>(i0) >= vertices.size() ||
                static_cast<size_t>(i1) >= vertices.size() ||
                static_cast<size_t>(i2) >= vertices.size()) {
                std::cerr << "Face index out of range at face " << i << "\n";
                return false;
            }

            triangles.push_back({vertices[i0], vertices[i1], vertices[i2]});
        }
    }

    return true;
}
