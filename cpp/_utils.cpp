#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

using std::array, std::vector, std::pair, std::make_pair, std::cout, std::endl, std::sort;

const double PI = 3.14159265358979323846;

double polar_angle(const pair<double, double> p1, const pair<double, double> p2) {
    if (p1.second == p2.second) return PI;
    double dy = p2.second - p1.second;
    double dx = p2.first - p1.first;

    return atan2(dy, dx);
}

int orientation(const pair<double, double> p1, const pair<double, double> p2, const pair<double, double> p3) {
    double val = (p2.first - p1.first) * (p3.second - p2.second) - (p2.second - p1.second) * (p3.first - p2.first);
    if (val == 0) return 0;
    return (val > 0) ? 1 : -1;
}

double distance(const pair<double, double> p1, const pair<double, double> p2) {
    return sqrt(pow(p2.first - p1.first, 2) + pow(p2.second - p1.second, 2));
}

// grahams scan
vector<pair<double, double>>
convex_hull(vector<pair<double, double>> points) {
    int n = points.size();
    if (n < 3) return {};

    // find the point with the lowest y coordinate
    int l = 0;
    for (int i = 1; i < n; i++) {
        if (points[i].second < points[l].second) {
            l = i;
        } else if (points[i].second == points[l].second) {
            if (points[i].first < points[l].first) {
                l = i;
            }
        }
    }

    // swap the lowest y coordinate point with the first point
    pair<double, double> p0 = points[l];
    points[l] = points[0];
    points[0] = p0;


    // sort the points by polar angle
    pair<double, double> p = points[0];
    sort(points.begin() + 1, points.end(), [p](pair<double, double> p1, pair<double, double> p2) {
        double o = orientation(p, p1, p2);
        if (o == 0) {
            return distance(p, p1) < distance(p, p2);
        }
        return o == -1;
    });

    // create the convex hull
    vector<pair<double, double>> hull;
    hull.push_back(points[0]);
    hull.push_back(points[1]);
    hull.push_back(points[2]);

    for (int i = 3; i < n; i++) {
        while (orientation(hull[hull.size() - 2], hull[hull.size() - 1], points[i]) != -1) {
            hull.pop_back();
        }
        hull.push_back(points[i]);
    }

    return hull;
}

pair<double, double> diff_pair(pair<double, double> p1, pair<double, double> p2) {
    return make_pair(p1.first - p2.first, p1.second - p2.second);
}

pair<double, double> add_pair(pair<double, double> p1, pair<double, double> p2) {
    return make_pair(p1.first + p2.first, p1.second + p2.second);
}

pair<double, double> mult_pair(pair<double, double> p, double c) {
    return make_pair(p.first * c, p.second * c);
}

bool operator==(pair<double, double> p1, pair<double, double> p2) {
    return p1.first == p2.first && p1.second == p2.second;
}

bool operator!=(pair<double, double> p1, pair<double, double> p2) {
    return p1.first != p2.first || p1.second != p2.second;
}

vector<pair<double, double>> roc_curve(vector<double> labels, vector<double> predictions) {
    vector<pair<double, double>> points;
    for (int i = 0; i < labels.size(); i++) {
        points.push_back(make_pair(predictions[i], labels[i]));
    }
    sort(points.begin(), points.end(), [](pair<double, double> p1, pair<double, double> p2) {
        return p1.first > p2.first;
    });

    vector<pair<double, double>> roc;
    int tp = 0, fp = 0;
    int p = 0, n = 0;
    for (int i = 0; i < points.size(); i++) {
        if (points[i].second == 1) {
            p++;
        } else {
            n++;
        }
    }
    for (int i = 0; i < points.size(); i++) {
        if (points[i].second == 1) {
            tp++;
        } else {
            fp++;
        }
        double tpr = (double) tp / p;
        double fpr = (double) fp / n;
        roc.push_back(make_pair(fpr, tpr));
    }

    // y_pred typically has many tied values. Here we remove duplicates from the roc curve
    vector<pair<double, double>> unique;
    for (int i = 0; i < roc.size(); i++) {
        if (i == 0 || roc[i] != roc[i - 1]) {
            unique.push_back(roc[i]);
        }
    }
    // Drop thresholds corresponding to points in between and collinear with other points.
    vector<pair<double, double>> filtered;
    for (int i = 0; i < unique.size(); i++) {
        pair<double, double> derived = diff_pair(unique[i + 1], unique[i]);
        pair<double, double> second_derived = diff_pair(derived, diff_pair(unique[i], unique[i - 1]));
        if (i == 0 || i == unique.size() - 1 || (second_derived.first != 0 && second_derived.second != 0)) {
            filtered.push_back(unique[i]);
        }
    }

    // add (0, 0) and (1, 1) to the roc curve if they are not already there
    if (filtered[0] != make_pair(0, 0)) {
        filtered.insert(filtered.begin(), make_pair(0, 0));
    }
    if (filtered[filtered.size() - 1] != make_pair(1, 1)) {
        filtered.push_back(make_pair(1, 1));
    }

    return filtered;
}

namespace py = pybind11;

PYBIND11_MODULE(_empulse, m) {
    m.def("convex_hull", &convex_hull, R"pbdoc(
        A function that calculates the convex hull of a set of points.
    )pbdoc");

    m.def("roc_curve", &roc_curve, R"pbdoc(
        A function that calculates the ROC curve.
    )pbdoc");
}
