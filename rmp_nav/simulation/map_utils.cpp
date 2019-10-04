#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <unordered_map>
#include <unordered_set>
#include <cstdint>
#include <vector>


namespace py = pybind11;
using std::pair;
using std::vector;
using std::unordered_map;
using std::unordered_set;


struct CellInfo {
  int parent_x, parent_y;
  double score;
  CellInfo(): parent_x(0), parent_y(0), score(0) {}
  CellInfo(int x, int y, double s): parent_x(x), parent_y(y), score(s) {}
};


struct pair_hash {
  template <class T1, class T2>
  std::size_t operator () (const std::pair<T1,T2> &p) const {
    auto h1 = std::hash<T1>{}(p.first);
    auto h2 = std::hash<T2>{}(p.second);
    return h1 * 39381 + h2 * 81731;  // Might not be a good hash, but performance is good enough.
  }
};


typedef std::unordered_map<pair<int, int>, CellInfo, pair_hash> CellMap;


static void dijkstra_choose_best_node(const CellMap& open_dict, int& best_x, int& best_y) {
  double best_g = 1e10;

  best_x = -1;
  best_y = -1;

  for (const auto &kv : open_dict) {
    int x = kv.first.first;
    int y = kv.first.second;
    const CellInfo &ci = kv.second;
    double g = ci.score;

    if (g < best_g) {
        best_g = g;
        best_x = x;
        best_y = y;
    }
  }
}


unordered_map<pair<int, int>, pair<int, int>, pair_hash> dijkstra(
    py::array_t<uint8_t> obstacle_map, int start_x, int start_y) {
  // obstacle_map a H x W uint8 array where value 0 indicates free space
  CellMap open_dict;

  unordered_set<pair<int, int>, pair_hash> close_dict;
  unordered_map<pair<int, int>, pair<int, int>, pair_hash> parents;

  open_dict[std::make_pair(start_x, start_y)] = CellInfo(start_x, start_y, 0.0);

  int height = obstacle_map.shape(0);
  int width = obstacle_map.shape(1);
  auto obstacle_map_view = obstacle_map.unchecked<2>();

  while (open_dict.size() > 0) {
    int x, y;
    dijkstra_choose_best_node(open_dict, x, y);

    auto xy = std::make_pair(x, y);

    CellInfo ci = open_dict[xy];
    open_dict.erase(xy);

    close_dict.insert(std::make_pair(x, y));

    for (int dy = -1; dy <= 1; dy++) {
      for (int dx = -1; dx <= 1; dx++) {
        if (dy == 0 && dx == 0) {
          continue;
        }

        int x2 = dx + x;
        int y2 = dy + y;

        auto xy2 = std::make_pair(x2, y2);

        if (0 <= x2 && x2 < width && 0 <= y2 && y2 < height) {
          if (close_dict.find(xy2) != close_dict.end()) {
            continue;
          }

          int cost = obstacle_map_view(y2, x2);

          if (cost == 255) {
            // 255 indicates hard obstacles
            continue;
          }

          double cost_scale = 1.0 + cost;

          auto find_result = open_dict.find(xy2);
          if (find_result == open_dict.end()) {
            double g = ci.score + std::sqrt(double(dx * dx + dy * dy)) * cost_scale;
            open_dict[xy2] = CellInfo(x, y, g);
            parents[xy2] = xy;
          } else {
            double g0 = find_result->second.score;
            double g = ci.score + std::sqrt(double(dx * dx + dy * dy) * cost_scale);
            if (g < g0) {
              open_dict[xy2] = CellInfo(x, y, g);
              parents[xy2] = xy;
            }
          }
        }
      }
    }
  }

  return parents;
}


// This is the same as the rasterize_line function in line_utils.py
void rasterize_line(int x0, int y0, int x1, int y1, vector<pair<int, int>>& points) {
  int dx = std::abs(x1 - x0);
  int dy = std::abs(y1 - y0);

  int ix = (x0 < x1) ? 1 : -1;
  int iy = (y0 < y1) ? 1 : -1;

  int e = 0;

  points.clear();
  for (int i = 0; i < dx + dy; i++) {
    points.push_back(std::make_pair(x0, y0));
    int e1 = e + dy;
    int e2 = e - dx;
    if (std::abs(e1) < std::abs(e2)) {
      x0 += ix;
      e = e1;
    } else {
      y0 += iy;
      e = e2;
    }
  }
  points.push_back(std::make_pair(x1, y1));
}


bool visible(py::array_t<float> dist_transform, int x1, int y1, int x2, int y2, float dist_thres) {
  vector<pair<int, int>> points;
  rasterize_line(x1, y1, x2, y2, points);

  int height = dist_transform.shape(0);
  int width = dist_transform.shape(1);

  auto view = dist_transform.unchecked<2>();

  for (auto& p : points) {
    int x = p.first;
    int y = p.second;
    if (0 <= x && x < width && 0 <= y && y < height && view(y, x) > dist_thres) {
      continue;
    } else {
      return false;
    }
  }

  return true;
}


PYBIND11_MODULE(map_utils_cpp, m) {
  m.def("dijkstra", &dijkstra, "");
  m.def("visible", &visible, "");
}
