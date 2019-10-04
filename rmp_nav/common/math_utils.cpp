#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <cmath>
#include <cstdint>


namespace py = pybind11;


static double vector_angle(double x1, double y1, double x2, double y2) {
  return std::atan2(x1 * y2 - y1 * x2, x1 * x2 + y1 * y2);
}


static void rotate(double x, double y, double angle, double* out_x, double* out_y) {
  double c = std::cos(angle);
  double s = std::sin(angle);
  *out_x = c * x - s * y;
  *out_y = s * x + c * y;
}


py::array_t<float> compute_normals(py::array_t<float> points) {
  int n = points.shape(0);

  auto normals = py::array_t<float>(
      py::array::ShapeContainer({n, 2}),
      py::array::StridesContainer({2 * 4, 4})
  );
  auto normals_view = normals.mutable_unchecked<2>();

  auto points_view = points.unchecked<2>();

  for (int i = 0; i < n; i++) {
    float x0 = points_view(i, 0);
    float y0 = points_view(i, 1);

    float x1, y1;
    if (i == 0) {
      x1 = points_view(n - 1, 0);
      y1 = points_view(n - 1, 1);
    } else {
      x1 = points_view((i - 1) % n, 0);
      y1 = points_view((i - 1) % n, 1);
    }

    float x2 = points_view((i + 1) % n, 0);
    float y2 = points_view((i + 1) % n, 1);

    float q1x = x1 - x0;
    float q1y = y1 - y0;
    float q2x = x2 - x0;
    float q2y = y2 - y0;

    float angle = vector_angle(q1x, q1y, q2x, q2y);
    if (angle < 0) {
      angle = -angle;
    } else {
      angle = M_PI * 2.0 - angle;
    }

    double q1_norm = std::sqrt(q1x * q1x + q1y * q1y);

    double nx, ny;
    rotate(q1x / q1_norm, q1y / q1_norm, -angle / 2, &nx, &ny);

    normals_view(i, 0) = float(nx);
    normals_view(i, 1) = float(ny);
  }

  return normals;
}


PYBIND11_MODULE(math_utils_cpp, m) {
  m.def("compute_normals", &compute_normals, "");
}
