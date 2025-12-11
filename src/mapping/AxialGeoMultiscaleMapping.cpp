#include "AxialGeoMultiscaleMapping.hpp"
#include <Eigen/Core>
#include "mesh/Mesh.hpp"

namespace precice::mapping {

AxialGeoMultiscaleMapping::AxialGeoMultiscaleMapping(
    Constraint          constraint,
    int                 dimensions,
    MultiscaleDimension dimension,
    MultiscaleType      type,
    MultiscaleAxis      axis,
    double              radius,
    SpreadProfile       profile,
    double              coreRadius)
    : Mapping(constraint, dimensions, false, Mapping::InitialGuessRequirement::None),
      _dimension(dimension),
      _type(type),
      _axis(axis),
      _radius(radius),
      _profile(profile),
      _coreRadius(coreRadius)
{
  setInputRequirement(Mapping::MeshRequirement::VERTEX);
  setOutputRequirement(Mapping::MeshRequirement::VERTEX);
}

void AxialGeoMultiscaleMapping::computeMapping()
{
  PRECICE_TRACE(output()->nVertices());

  PRECICE_ASSERT(input().get() != nullptr);
  PRECICE_ASSERT(output().get() != nullptr);

  if (getConstraint() == CONSISTENT) {
    PRECICE_DEBUG("Compute consistent mapping");
    if (_type == MultiscaleType::SPREAD) {
      const int outDataDimensions   = 3;
      int       effectiveCoordinate = static_cast<std::underlying_type_t<MultiscaleType>>(_axis); // Convert enum struct to int
      PRECICE_ASSERT(effectiveCoordinate == static_cast<std::underlying_type_t<MultiscaleType>>(MultiscaleAxis::X) ||
                         effectiveCoordinate == static_cast<std::underlying_type_t<MultiscaleType>>(MultiscaleAxis::Y) ||
                         effectiveCoordinate == static_cast<std::underlying_type_t<MultiscaleType>>(MultiscaleAxis::Z),
                     "Unknown multiscale axis type.");
      size_t const inSize  = input()->nVertices();
      size_t const outSize = output()->nVertices();
      if (_dimension == MultiscaleDimension::D1D3 || _dimension == MultiscaleDimension::D1D2) {
        PRECICE_CHECK(input()->nVertices() == 1, "You can only define an axial geometric multiscale 1D-{} mapping of type spread from a mesh with exactly one vertex.");

        /* When we add support for 1D meshes (https://github.com/precice/precice/issues/1669),
          we should check for valid dimensions combination.

          const int inDataDimensions  = input()->getDimensions();
          const int outDataDimensions = output()->getDimensions();

          PRECICE_CHECK(input()->getDimensions() == 1, "The input mesh on an axial geometric multiscale mapping can only be 1D at the moment, but it was defined to be {}.", input()->getDimensions())
          PRECICE_CHECK(output()->getDimensions() == 3, "The output mesh on an axial geometric multiscale mapping can only be 3D at the moment, but it was defined to be {}.", input()->getDimensions())
        */

        // compute distances between 1D vertex and 3D vertices
        mesh::Vertex    &v0                           = input()->vertex(0);
        constexpr double distance_to_radius_threshold = 1.05;

        _vertexDistances.clear();
        _vertexDistances.reserve(output()->nVertices());

        for (size_t i = 0; i < outSize; i++) {
          Eigen::VectorXd difference(outDataDimensions);
          difference = v0.getCoords();
          difference -= output()->vertex(i).getCoords();
          double distance_to_radius = difference.norm() / _radius;
          PRECICE_CHECK(distance_to_radius <= distance_to_radius_threshold, "Output mesh has vertices that do not coincide with the geometric multiscale interface defined by the input mesh. Ratio of vertex distance to radius is {} (which is larger than the assumed threshold of distance_to_radius_threshold).", distance_to_radius);
          _vertexDistances.push_back(distance_to_radius);
        }
      } else {
        PRECICE_ASSERT(_dimension == MultiscaleDimension::D2D3);
        PRECICE_CHECK(input()->nVertices() > 1, "You can only define an axial geometric multiscale 2D-3D mapping of type spread from a mesh with more than 1 vertex.");
        _nearestVertex.clear();
        _nearestVertex.reserve(output()->nVertices());
        _vertexDistances.clear();
        _vertexDistances.reserve(output()->nVertices());
        _maxDistancePerInput.clear();
        _maxDistancePerInput.resize(input()->nVertices(), 0.0);
        for (size_t j = 0; j < outSize; j++) {
          const Eigen::VectorXd &xOut          = output()->vertex(j).getCoords();
          double                 bestDistance2 = std::numeric_limits<double>::max();
          int                    bestIdx       = -1;
          for (size_t i = 0; i < inSize; i++) {
            const Eigen::VectorXd &xIn        = input()->vertex(i).getCoords();
            Eigen::VectorXd        difference = xOut - xIn;
            double                 distance2  = difference.squaredNorm();
            if (distance2 < bestDistance2) {
              bestDistance2 = distance2;
              bestIdx       = static_cast<int>(i);
            }
          }
          PRECICE_ASSERT(bestIdx >= 0, "Could not find nearest input vertex for output vertex {}", j);
          _nearestVertex.push_back(bestIdx);
          double distance = std::sqrt(bestDistance2);
          _vertexDistances.push_back(distance);
          if (distance > _maxDistancePerInput[static_cast<size_t>(bestIdx)]) {
            _maxDistancePerInput[static_cast<size_t>(bestIdx)] = distance;
          }
        }
      }
    } else {
      PRECICE_ASSERT(_type == MultiscaleType::COLLECT);
      size_t const inSize  = input()->nVertices();
      size_t const outSize = output()->nVertices();
      if (_dimension == MultiscaleDimension::D1D3) {
        PRECICE_CHECK(outSize == 1,
                      "You can only define an axial geometric multiscale 1D-3D mappinp of type collect to a mesh with exactly one vertex.");
        _collectWeights.clear();
        _collectWeights.resize(inSize, 0.0);
        // Trivial case: only one vertex → full weight
        if (inSize == 1) {
          _collectWeights[0] = 1.0;
          return;
        }
        // ------------------------------------------------------------
        // 1) Read all coordinates and determine radial plane
        // ------------------------------------------------------------
        std::vector<Eigen::VectorXd> coords(inSize);
        for (size_t i = 0; i < inSize; ++i) {
          coords[i] = input()->vertex(i).getCoords();
        }

        int dim = static_cast<int>(coords[0].size());
        PRECICE_CHECK(dim >= 2,
                      "1D-3D axial geometric multiscale mapping requires dimension >= 2.");

        // Compute per-dimension spans to identify two "radial" directions
        Eigen::VectorXd minCoord = coords[0];
        Eigen::VectorXd maxCoord = coords[0];
        for (size_t i = 1; i < inSize; ++i) {
          minCoord = minCoord.cwiseMin(coords[i]);
          maxCoord = maxCoord.cwiseMax(coords[i]);
        }

        std::vector<std::pair<double, int>> spans;
        spans.reserve(dim);
        for (int d = 0; d < dim; ++d) {
          double span = std::abs(maxCoord[d] - minCoord[d]);
          spans.emplace_back(span, d);
        }
        std::sort(spans.begin(), spans.end(),
                  [](auto const &a, auto const &b) { return a.first > b.first; });

        // Take the two directions with the largest span as radial plane
        int dirX = spans[0].second;
        int dirY = (dim >= 2) ? spans[1].second : (spans[0].second + 1) % dim;

        // ------------------------------------------------------------
        // 2) Build r, rho, theta and inner/outer masks
        // ------------------------------------------------------------
        std::vector<double> x(inSize), y(inSize), r(inSize), rho(inSize), theta(inSize);
        double              rMax = 0.0;

        for (size_t i = 0; i < inSize; ++i) {
          double xi = coords[i][dirX];
          double yi = coords[i][dirY];
          x[i]      = xi;
          y[i]      = yi;

          double ri = std::sqrt(xi * xi + yi * yi);
          r[i]      = ri;
          rMax      = std::max(rMax, ri);

          rho[i]   = std::max(std::abs(xi), std::abs(yi)); // "square radius"
          theta[i] = std::atan2(yi, xi);
        }

        double            rhoSwitch = _coreRadius;
        std::vector<bool> isInner(inSize, false);
        std::vector<bool> isOuter(inSize, false);

        for (size_t i = 0; i < inSize; ++i) {
          if (rhoSwitch > 0.0 && rho[i] <= rhoSwitch) {
            isInner[i] = true;
          } else {
            isOuter[i] = true;
          }
        }

        // Collect indices for inner and outer sets
        std::vector<size_t> innerIdx, outerIdx;
        innerIdx.reserve(inSize);
        outerIdx.reserve(inSize);

        for (size_t i = 0; i < inSize; ++i) {
          if (isInner[i])
            innerIdx.push_back(i);
          else
            outerIdx.push_back(i);
        }

        // ------------------------------------------------------------
        // Helper: cluster 1D coordinates into levels
        // ------------------------------------------------------------
        auto cluster_1d =
            [&](const std::vector<double> &coord,
                const std::vector<size_t> &subset,
                const double              *factorOpt, // nullptr → Cartesian mode
                double                     eps,
                std::vector<int>          &levelIndex,
                int                        startLevel) -> int {
          int    currentLevel = startLevel;
          size_t nLocal       = subset.size();

          // Case: no nodes to cluster
          if (nLocal == 0)
            return startLevel;

          // Build local vector (value, globalIndex) and sort by value
          std::vector<std::pair<double, size_t>> local;
          local.reserve(nLocal);
          for (size_t k = 0; k < nLocal; ++k)
            local.emplace_back(coord[subset[k]], subset[k]);

          std::sort(local.begin(), local.end(),
                    [](auto const &a, auto const &b) { return a.first < b.first; });

          // Extract sorted coordinates
          std::vector<double> vals_sorted(nLocal);
          for (size_t k = 0; k < nLocal; ++k)
            vals_sorted[k] = local[k].first;

          // Assign first level
          levelIndex[local[0].second] = currentLevel;

          // -----------------------------------------------------------------
          // MODE 1: Cartesian-style → factorOpt == nullptr
          // -----------------------------------------------------------------
          if (factorOpt == nullptr) {
            for (size_t k = 1; k < nLocal; ++k) {
              if (std::abs(vals_sorted[k] - vals_sorted[k - 1]) > eps)
                ++currentLevel;

              levelIndex[local[k].second] = currentLevel;
            }
            return currentLevel + 1;
          }

          // -----------------------------------------------------------------
          // MODE 2: Median-gap clustering → factorOpt != nullptr
          // -----------------------------------------------------------------
          double factor = *factorOpt;

          // Compute gaps
          std::vector<double> d;
          d.reserve(nLocal > 1 ? nLocal - 1 : 0);
          for (size_t k = 1; k < nLocal; ++k)
            d.push_back(vals_sorted[k] - vals_sorted[k - 1]);

          // Collect positive gaps
          std::vector<double> pos;
          pos.reserve(d.size());
          for (double g : d)
            if (g > eps)
              pos.push_back(g);

          // No positive gaps → everything is one cluster
          if (pos.empty()) {
            for (auto &p : local)
              levelIndex[p.second] = currentLevel;

            return currentLevel + 1;
          }

          // Median of positive gaps
          std::nth_element(pos.begin(), pos.begin() + pos.size() / 2, pos.end());
          double median_d = pos[pos.size() / 2];

          double threshold = factor * median_d;

          // Ring-style cluster assignment
          for (size_t k = 1; k < nLocal; ++k) {
            if (d[k - 1] > threshold)
              ++currentLevel;

            levelIndex[local[k].second] = currentLevel;
          }

          return currentLevel + 1;
        };

        // ------------------------------------------------------------
        // 3) Inner region: Cartesian bands → area = Δx * Δy
        // ------------------------------------------------------------
        const double     EPS_INNER = 1e-1; // same as Python EPS
        std::vector<int> xLevel(inSize, -1), yLevel(inSize, -1);
        int              nXLevels = 0, nYLevels = 0;
        if (!innerIdx.empty()) {
          std::vector<double> xInner(inSize), yInner(inSize);
          for (size_t i : innerIdx) {
            xInner[i] = x[i];
            yInner[i] = y[i];
          }

          nXLevels = cluster_1d(xInner, innerIdx, /*factorOpt =*/nullptr, EPS_INNER,
                                xLevel, /*startLevel=*/0);
          nYLevels = cluster_1d(yInner, innerIdx, /*factorOpt =*/nullptr, EPS_INNER,
                                yLevel, /*startLevel=*/0);
        }

        std::vector<double> deltaX(std::max(1, nXLevels), 0.0);
        std::vector<double> deltaY(std::max(1, nYLevels), 0.0);

        // Δx per x-level (mimic np.unique + np.diff)
        for (int L = 0; L < nXLevels; ++L) {
          std::vector<double> xs;
          for (size_t i : innerIdx) {
            if (xLevel[i] == L) {
              xs.push_back(x[i]);
            }
          }

          if (xs.size() <= 1) {
            deltaX[L] = 0.0;
            continue;
          }

          // sort and unique → like np.unique(np.sort(...))
          std::sort(xs.begin(), xs.end());
          xs.erase(std::unique(xs.begin(), xs.end()), xs.end());

          if (xs.size() <= 1) {
            deltaX[L] = 0.0;
            continue;
          }

          std::vector<double> dx;
          dx.reserve(xs.size() - 1);
          for (size_t k = 1; k < xs.size(); ++k) {
            dx.push_back(xs[k] - xs[k - 1]);
          }

          if (dx.size() == 1) {
            deltaX[L] = dx[0];
          } else {
            deltaX[L] = 0.5 * (dx.front() + dx.back());
          }
        }

        // Δy per y-level
        // Δy per y-level (mimic np.unique + np.diff)
        for (int L = 0; L < nYLevels; ++L) {
          std::vector<double> ys;
          for (size_t i : innerIdx) {
            if (yLevel[i] == L) {
              ys.push_back(y[i]);
            }
          }

          if (ys.size() <= 1) {
            deltaY[L] = 0.0;
            continue;
          }

          // sort and unique → like np.unique(np.sort(...))
          std::sort(ys.begin(), ys.end());
          ys.erase(std::unique(ys.begin(), ys.end()), ys.end());

          if (ys.size() <= 1) {
            deltaY[L] = 0.0;
            continue;
          }

          std::vector<double> dy;
          dy.reserve(ys.size() - 1);
          for (size_t k = 1; k < ys.size(); ++k) {
            dy.push_back(ys[k] - ys[k - 1]);
          }

          if (dy.size() == 1) {
            deltaY[L] = dy[0];
          } else {
            deltaY[L] = 0.5 * (dy.front() + dy.back());
          }
        }

        // ------------------------------------------------------------
        // 4) Outer region: adaptive rings → area = R_j * Δr_j * Δθ_i
        // ------------------------------------------------------------
        std::vector<int> ringIndexOuter(inSize, -1);

        // Work only on the outer vertices
        std::vector<size_t> remaining = outerIdx;

        int          ringId      = 0;
        int          nRef        = -1;  // reference number of points in first (outermost) ring
        double       factorOuter = 1.0; // BASE_FACTOR_OUTER
        const double FACTOR_STEP = 0.2; // how much we increase each time
        const double MAX_FACTOR  = 5.0;
        const double EPS_OUTER   = 1e-5;

        while (!remaining.empty()) {
          // --- cluster r over the remaining nodes with current factorOuter ---
          std::vector<int> tmpLevels(inSize, -1);
          // cluster1D uses (coord, subset, factor, eps, levelIndex, startLevel)
          (void) cluster_1d(r, remaining, &factorOuter, EPS_OUTER, tmpLevels, 0);

          // --- collect which level labels exist inside "remaining" ---
          std::vector<int> levels;
          for (size_t idx : remaining) {
            int  L     = tmpLevels[idx];
            bool found = false;
            for (int v : levels) {
              if (v == L) {
                found = true;
                break;
              }
            }
            if (!found) {
              levels.push_back(L);
            }
          }

          // --- find the OUTERMOST group among the remaining ---
          double bestMeanR  = -1.0;
          int    outerLabel = -1;

          for (int L : levels) {
            double sumR = 0.0;
            int    cnt  = 0;
            for (size_t idx : remaining) {
              if (tmpLevels[idx] == L) {
                sumR += r[idx];
                ++cnt;
              }
            }
            if (cnt > 0) {
              double meanR = sumR / static_cast<double>(cnt);
              if (meanR > bestMeanR) {
                bestMeanR  = meanR;
                outerLabel = L;
              }
            }
          }

          // nodes belonging to this outermost ring
          std::vector<size_t> outerNodes;
          outerNodes.reserve(remaining.size());
          for (size_t idx : remaining) {
            if (tmpLevels[idx] == outerLabel) {
              outerNodes.push_back(idx);
            }
          }

          // --- First ring (outermost) → define reference point-count ---
          if (nRef < 0) {
            nRef = static_cast<int>(outerNodes.size());
            for (size_t idx : outerNodes) {
              ringIndexOuter[idx] = ringId;
            }
            ++ringId;

            // remove these indices from "remaining"
            std::vector<size_t> newRemaining;
            newRemaining.reserve(remaining.size());
            for (size_t idx : remaining) {
              bool isInOuter = false;
              for (size_t j : outerNodes) {
                if (idx == j) {
                  isInOuter = true;
                  break;
                }
              }
              if (!isInOuter) {
                newRemaining.push_back(idx);
              }
            }
            remaining.swap(newRemaining);
            // keep factorOuter as is
            continue;
          }

          // --- Inner rings ---
          if (static_cast<int>(outerNodes.size()) < nRef && factorOuter < MAX_FACTOR) {
            // This ring has fewer points than the reference outer ring:
            // emulate increasing factor_outer from this ring inwards
            factorOuter += FACTOR_STEP;
            continue; // re-run clustering on the same "remaining"
          } else {
            // Accept this ring (even if still smaller once we hit MAX_FACTOR)
            for (size_t idx : outerNodes) {
              ringIndexOuter[idx] = ringId;
            }
            ++ringId;

            // remove these indices from "remaining"
            std::vector<size_t> newRemaining;
            newRemaining.reserve(remaining.size());
            for (size_t idx : remaining) {
              bool isInOuter = false;
              for (size_t j : outerNodes) {
                if (idx == j) {
                  isInOuter = true;
                  break;
                }
              }
              if (!isInOuter) {
                newRemaining.push_back(idx);
              }
            }
            remaining.swap(newRemaining);
            // keep (possibly increased) factorOuter for further inner rings
            continue;
          }
        }

        // Now we have ringIndexOuter filled for all outer nodes
        int nRings = ringId;

        // Build list of vertices per ring and ring mean radius (like Python)
        std::vector<std::vector<size_t>> ringVertices(std::max(1, nRings));
        for (size_t i : outerIdx) {
          int L = ringIndexOuter[i];
          if (L >= 0 && L < nRings) {
            ringVertices[L].push_back(i);
          }
        }

        std::vector<double> ringRadius(std::max(1, nRings), 0.0);
        for (int L = 0; L < nRings; ++L) {
          double sumR = 0.0;
          int    cnt  = 0;
          for (size_t i : ringVertices[L]) {
            sumR += r[i];
            ++cnt;
          }
          ringRadius[L] = (cnt > 0) ? sumR / static_cast<double>(cnt) : 0.0;
        }

        // radial thickness per ring: deltaR, exactly like Python
        std::vector<double> deltaR(std::max(1, nRings), 0.0);
        if (nRings == 1) {
          // python: delta_r[0] = r_vals.max() - r_vals.min()
          double rMin      = std::numeric_limits<double>::max();
          double rMaxLocal = 0.0;
          for (size_t i : ringVertices[0]) {
            rMin      = std::min(rMin, r[i]);
            rMaxLocal = std::max(rMaxLocal, r[i]);
          }
          deltaR[0] = (rMaxLocal - rMin);
        } else if (nRings > 1) {
          // sort rings by radius
          std::vector<int> order(nRings);
          for (int k = 0; k < nRings; ++k) {
            order[k] = k;
          }
          std::sort(order.begin(), order.end(),
                    [&](int a, int b) { return ringRadius[a] < ringRadius[b]; });

          std::vector<double> rSorted(nRings);
          for (int k = 0; k < nRings; ++k) {
            rSorted[k] = ringRadius[order[k]];
          }
          std::vector<double> dRSorted(nRings, 0.0);

          for (int k = 0; k < nRings; ++k) {
            if (k == 0) {
              // inner boundary at rhoSwitch, same as Python:
              // dr_sorted[0] = 0.5 * (r_sorted[1] + r_sorted[0]) - rho_switch
              double innerRad = (rhoSwitch > 0.0) ? rhoSwitch : rSorted[0];
              dRSorted[k]     = 0.5 * (rSorted[1] + rSorted[0]) - innerRad;
            } else if (k == nRings - 1) {
              dRSorted[k] = 0.5 * (rSorted[nRings - 1] - rSorted[nRings - 2]);
            } else {
              dRSorted[k] = 0.5 * (rSorted[k + 1] - rSorted[k - 1]);
            }
          }

          // map back to original ring order
          for (int pos = 0; pos < nRings; ++pos) {
            int L     = order[pos];
            deltaR[L] = dRSorted[pos];
          }
        }

        // ------------------------------------------------------------
        // 5) Compute node areas and basic weights
        // ------------------------------------------------------------
        std::vector<double> area(inSize, 0.0);

        // Inner: Cartesian Δx * Δy
        for (size_t i : innerIdx) {
          int jx = xLevel[i];
          int jy = yLevel[i];
          if (jx < 0 || jy < 0)
            continue;
          double dx = (jx < nXLevels) ? deltaX[jx] : 0.0;
          double dy = (jy < nYLevels) ? deltaY[jy] : 0.0;
          area[i]   = std::max(dx, 0.0) * std::max(dy, 0.0);
        }

        // Outer: polar R_j * Δr_j * Δθ_i
        for (int L = 0; L < nRings; ++L) {
          auto const &ring = ringVertices[L];
          if (ring.empty())
            continue;

          // collect angles and sort along ring
          std::vector<std::pair<double, size_t>> angIdx;
          angIdx.reserve(ring.size());
          for (size_t i : ring) {
            angIdx.emplace_back(theta[i], i);
          }
          std::sort(angIdx.begin(), angIdx.end(),
                    [](auto const &a, auto const &b) { return a.first < b.first; });

          size_t              N = angIdx.size();
          std::vector<double> thetaU(N);
          for (size_t k = 0; k < N; ++k) {
            thetaU[k] = angIdx[k].first;
          }

          // unwrap
          for (size_t k = 1; k < N; ++k) {
            while (thetaU[k] - thetaU[k - 1] > M_PI)
              thetaU[k] -= 2.0 * M_PI;
            while (thetaU[k] - thetaU[k - 1] < -M_PI)
              thetaU[k] += 2.0 * M_PI;
          }

          std::vector<double> thetaExt(N + 2);
          thetaExt[0]     = thetaU.back() - 2.0 * M_PI;
          thetaExt[N + 1] = thetaU.front() + 2.0 * M_PI;
          for (size_t k = 0; k < N; ++k) {
            thetaExt[k + 1] = thetaU[k];
          }

          std::vector<double> dTheta(N, 0.0);
          for (size_t k = 0; k < N; ++k) {
            dTheta[k] = 0.5 * (thetaExt[k + 2] - thetaExt[k]);
          }

          double Rj = ringRadius[L];
          double dR = std::max(deltaR[L], 0.0);
          for (size_t k = 0; k < N; ++k) {
            size_t iNode = angIdx[k].second;
            area[iNode]  = Rj * dR * std::max(dTheta[k], 0.0);
          }
        }

        double totalArea = 0.0;
        for (double a : area)
          totalArea += a;

        if (totalArea <= 0.0) {
          // fallback: uniform weights
          double w = 1.0 / static_cast<double>(inSize);
          for (size_t i = 0; i < inSize; ++i) {
            _collectWeights[i] = w;
          }
          return;
        }

        std::vector<double> w0(inSize, 0.0);
        for (size_t i = 0; i < inSize; ++i) {
          w0[i] = area[i] / totalArea;
        }

        // ------------------------------------------------------------
        // 6) Optional: 2-moment correction between inner / outer
        //     to get correct <r^2> for a circular pipe
        // ------------------------------------------------------------
        double sInner = 0.0, sOuter = 0.0;
        double aInner = 0.0, bOuter = 0.0;
        for (size_t i = 0; i < inSize; ++i) {
          double wi = w0[i];
          double r2 = r[i] * r[i];
          if (isInner[i]) {
            sInner += wi;
            aInner += wi * r2;
          } else {
            sOuter += wi;
            bOuter += wi * r2;
          }
        }

        double R_eff    = rMax;
        double targetM2 = 0.5 * R_eff * R_eff;

        double alpha = 1.0;
        double beta  = 1.0;

        double A11 = sInner;
        double A12 = sOuter;
        double A21 = aInner;
        double A22 = bOuter;
        double B1  = 1.0;
        double B2  = targetM2;

        double det = A11 * A22 - A12 * A21;
        if (std::abs(det) > 1e-14) {
          alpha = (B1 * A22 - B2 * A12) / det;
          beta  = (-B1 * A21 + B2 * A11) / det;
        }

        // Build corrected weights and renormalize
        double sumW = 0.0;
        for (size_t i = 0; i < inSize; ++i) {
          double factor      = isInner[i] ? alpha : beta;
          _collectWeights[i] = factor * w0[i];
          sumW += _collectWeights[i];
        }

        if (sumW > 0.0) {
          for (size_t i = 0; i < inSize; ++i) {
            _collectWeights[i] /= sumW;
          }
        } else {
          // fallback: uniform
          double w = 1.0 / static_cast<double>(inSize);
          for (size_t i = 0; i < inSize; ++i) {
            _collectWeights[i] = w;
          }
        }

      } else if (_dimension == MultiscaleDimension::D1D2) {
        PRECICE_CHECK(output()->nVertices() == 1,
                      "You can only define an axial geometric multiscale 1D-2D mapping of type collect to a mesh with exactly one vertex.");
        _collectWeights.clear();
        _collectWeights.resize(inSize, 0.0);

        // Trivial case: only one vertex → full weight
        if (inSize == 1) {
          _collectWeights[0] = 1.0;
        } else {
          // --- 1) Read all coordinates of the 2D interface vertices ---
          std::vector<Eigen::VectorXd> coords(inSize);
          for (size_t i = 0; i < inSize; ++i) {
            coords[i] = input()->vertex(i).getCoords();
          }

          // --- 2) Detect main tangential direction (largest span) ---
          // We assume a straight line interface aligned with the axis
          // of largest extent in the global coordinates.
          int             dim      = coords[0].size();
          Eigen::VectorXd minCoord = coords[0];
          Eigen::VectorXd maxCoord = coords[0];
          for (size_t i = 1; i < inSize; ++i) {
            minCoord = minCoord.cwiseMin(coords[i]);
            maxCoord = maxCoord.cwiseMax(coords[i]);
          }

          int    mainDir = 0;
          double maxSpan = std::abs(maxCoord[0] - minCoord[0]);
          for (int d = 1; d < dim; ++d) {
            double span = std::abs(maxCoord[d] - minCoord[d]);
            if (span > maxSpan) {
              maxSpan = span;
              mainDir = d;
            }
          }

          // --- 3) Build a sorted index along this direction ---
          std::vector<size_t> indices(inSize);
          for (size_t i = 0; i < inSize; ++i) {
            indices[i] = i;
          }
          std::sort(indices.begin(), indices.end(),
                    [&](size_t a, size_t b) {
                      return coords[a][mainDir] < coords[b][mainDir];
                    });

          // --- 4) Compute length-based weights (midpoint rule) ---
          // For non-uniform spacing along the line:
          //   Δs_0     = (s_1     - s_0    ) / 2
          //   Δs_k     = (s_{k+1} - s_{k-1}) / 2,  0 < k < N-1
          //   Δs_{N-1} = (s_{N-1} - s_{N-2}) / 2
          std::vector<double> s(inSize);
          for (size_t k = 0; k < inSize; ++k) {
            s[k] = coords[indices[k]][mainDir];
          }

          std::vector<double> localWeights(inSize, 0.0);
          // first
          localWeights[0] = 0.5 * (s[1] - s[0]);
          // interior
          for (size_t k = 1; k < inSize - 1; ++k) {
            localWeights[k] = 0.5 * (s[k + 1] - s[k - 1]);
          }
          // last
          localWeights[inSize - 1] = 0.5 * (s[inSize - 1] - s[inSize - 2]);

          // --- 5) Map back to original vertex indices and normalize ---
          double totalLength = 0.0;
          for (size_t k = 0; k < inSize; ++k) {
            size_t originalIdx           = indices[k];
            double w                     = std::max(localWeights[k], 0.0); // guard against tiny negative due to round-off
            _collectWeights[originalIdx] = w;
            totalLength += w;
          }

          if (totalLength > 0.0) {
            for (size_t i = 0; i < inSize; ++i) {
              _collectWeights[i] /= totalLength;
            }
          } else {
            // Fallback: uniform weights if something went wrong
            double w = 1.0 / static_cast<double>(inSize);
            for (size_t i = 0; i < inSize; ++i) {
              _collectWeights[i] = w;
            }
          }
        }
      } else {
        PRECICE_ASSERT(_dimension == MultiscaleDimension::D2D3);
        PRECICE_CHECK(outSize > 1, "You can only define an axial geometric multiscale 2D-3D mapping of type collect to a mesh with more than 1 vertex.");
        _collectBands.clear();
        _collectBands.resize(output()->nVertices());
        for (size_t i = 0; i < inSize; i++) {
          const Eigen::VectorXd &xIn          = input()->vertex(i).getCoords();
          double                 bestDistance = std::numeric_limits<double>::max();
          int                    bestIdx      = -1;
          for (size_t j = 0; j < outSize; j++) {
            const Eigen::VectorXd &xOut       = output()->vertex(j).getCoords();
            Eigen::VectorXd        difference = xIn - xOut;
            double                 distance   = difference.squaredNorm();
            if (distance < bestDistance) {
              bestDistance = distance;
              bestIdx      = static_cast<int>(j);
            }
          }
          PRECICE_ASSERT(bestIdx >= 0, "Could not find nearest output vertex for input vertex {}", i);
          _collectBands[bestIdx].push_back(static_cast<int>(i));
        }
      }
    }

  } else {
    PRECICE_ASSERT(getConstraint() == CONSERVATIVE);
    PRECICE_UNREACHABLE("Axial conservative geometric multiscale mapping is not implemented");
    PRECICE_DEBUG("Compute conservative mapping");
  }
  _hasComputedMapping = true;
}

void AxialGeoMultiscaleMapping::clear()
{
  PRECICE_TRACE();
  _vertexDistances.clear();
  _hasComputedMapping = false;
}

void AxialGeoMultiscaleMapping::mapConservative(const time::Sample &inData, Eigen::VectorXd &outData)
{
  PRECICE_ASSERT(getConstraint() == CONSERVATIVE);
  PRECICE_UNREACHABLE("Axial conservative geometric multiscale mapping is not implemented");
  PRECICE_DEBUG("Map conservative");
}

void AxialGeoMultiscaleMapping::mapConsistent(const time::Sample &inData, Eigen::VectorXd &outData)
{
  PRECICE_TRACE();

  const int              inDataDimensions = inData.dataDims;
  const Eigen::VectorXd &inputValues      = inData.values;
  Eigen::VectorXd       &outputValues     = outData;
  // TODO: check if this needs to change when access to mesh dimension is possible
  const int outDataDimensions = outData.size() / output()->nVertices();

  // Check that the number of values for the input and output is right according to their dimensions
  PRECICE_ASSERT((inputValues.size() / static_cast<std::size_t>(inDataDimensions) == input()->nVertices()),
                 inputValues.size(), inDataDimensions, input()->nVertices());
  PRECICE_ASSERT((outputValues.size() / static_cast<std::size_t>(outDataDimensions) == output()->nVertices()),
                 outputValues.size(), outDataDimensions, output()->nVertices());

  // We currently don't support 1D data, so we need that the user specifies data of the same dimensions on both sides
  PRECICE_ASSERT(inDataDimensions == outDataDimensions);

  // Effective component (axis) to read/write: 0 for scalar fields (to avoid out-of-bounds), or 0 (x)/ 1 (y)/ 2 (z) for vectors
  int effectiveCoordinate;
  if (inDataDimensions == 1) {
    effectiveCoordinate = 0;
  } else {
    effectiveCoordinate = static_cast<std::underlying_type_t<MultiscaleType>>(_axis);
    PRECICE_ASSERT(effectiveCoordinate == static_cast<std::underlying_type_t<MultiscaleType>>(MultiscaleAxis::X) ||
                       effectiveCoordinate == static_cast<std::underlying_type_t<MultiscaleType>>(MultiscaleAxis::Y) ||
                       effectiveCoordinate == static_cast<std::underlying_type_t<MultiscaleType>>(MultiscaleAxis::Z),
                   "Unknown multiscale axis type.");
  }

  PRECICE_DEBUG("Map consistent");

  if (_type == MultiscaleType::SPREAD) {
    size_t const inSize  = input()->nVertices();
    size_t const outSize = output()->nVertices();
    if (_dimension == MultiscaleDimension::D1D3 || _dimension == MultiscaleDimension::D1D2) {
      PRECICE_ASSERT(input()->nVertices() == 1);
      for (size_t i = 0; i < outSize; i++) {
        PRECICE_ASSERT(static_cast<size_t>((i * outDataDimensions) + effectiveCoordinate) < static_cast<size_t>(outputValues.size()), ((i * outDataDimensions) + effectiveCoordinate), outputValues.size());
        if (_profile == SpreadProfile::PARABOLIC) {
          // When adding support for 2D, remember that this should be 1.5 * inputValues(effectiveCoordinate) * (1 - (_vertexDistances[i] * _vertexDistances[i]));
          const double factor                                         = (_dimension == MultiscaleDimension::D1D3) ? 2.0 : 1.5;
          outputValues((i * outDataDimensions) + effectiveCoordinate) = factor * inputValues(effectiveCoordinate) * (1 - (_vertexDistances[i] * _vertexDistances[i]));
        } else if (_profile == SpreadProfile::UNIFORM) {
          outputValues((i * outDataDimensions) + effectiveCoordinate) = inputValues(effectiveCoordinate);
        }
      }
    } else {
      PRECICE_ASSERT(_dimension == MultiscaleDimension::D2D3);
      PRECICE_ASSERT(_nearestVertex.size() == outSize);
      PRECICE_ASSERT(_vertexDistances.size() == outSize);
      PRECICE_ASSERT(_maxDistancePerInput.size() == inSize);
      for (size_t i = 0; i < outSize; i++) {
        PRECICE_ASSERT(_nearestVertex[i] >= 0 && static_cast<size_t>(_nearestVertex[i]) < inSize, _nearestVertex[i], inSize);
        PRECICE_ASSERT(static_cast<size_t>((i * outDataDimensions) + effectiveCoordinate) < static_cast<size_t>(outputValues.size()), (i * outDataDimensions) + effectiveCoordinate, outputValues.size());
        PRECICE_ASSERT(static_cast<size_t>((static_cast<size_t>(_nearestVertex[i]) * inDataDimensions) + effectiveCoordinate) < static_cast<size_t>(inputValues.size()), (static_cast<size_t>(_nearestVertex[i]) * inDataDimensions) + effectiveCoordinate, inputValues.size());
        double R = _maxDistancePerInput[static_cast<size_t>(_nearestVertex[i])];
        if (_profile == SpreadProfile::UNIFORM) {
          outputValues((i * outDataDimensions) + effectiveCoordinate) = inputValues((static_cast<size_t>(_nearestVertex[i]) * inDataDimensions) + effectiveCoordinate);
        } else if (_profile == SpreadProfile::PARABOLIC) {
          double r_hat                                                = _vertexDistances[i] / R;
          outputValues((i * outDataDimensions) + effectiveCoordinate) = (4.0 / 3.0) * inputValues((static_cast<size_t>(_nearestVertex[i]) * inDataDimensions) + effectiveCoordinate) * (1.0 - r_hat * r_hat);
        }
      }
    }
  } else {
    PRECICE_ASSERT(_type == MultiscaleType::COLLECT);
    size_t const inSize  = input()->nVertices();
    size_t const outSize = output()->nVertices();
    if (_dimension == MultiscaleDimension::D1D2 || _dimension == MultiscaleDimension::D1D3) {
      PRECICE_ASSERT(output()->nVertices() == 1);
      PRECICE_ASSERT(_collectWeights.size() == inSize);
      outputValues(effectiveCoordinate) = 0.0;
      for (size_t i = 0; i < inSize; ++i) {
        PRECICE_ASSERT(static_cast<size_t>((i * inDataDimensions) + effectiveCoordinate) < static_cast<size_t>(inputValues.size()),
                       ((i * inDataDimensions) + effectiveCoordinate), inputValues.size());
        outputValues(effectiveCoordinate) += _collectWeights[i] * inputValues((i * inDataDimensions) + effectiveCoordinate);
      }
      // _collectWeights already sum to 1, so no further normalization
    } else {
      PRECICE_ASSERT(_dimension == MultiscaleDimension::D2D3);
      PRECICE_ASSERT(_collectBands.size() == output()->nVertices());
      for (size_t j = 0; j < outSize; j++) {
        if (_collectBands[j].empty()) {
          continue;
        }
        PRECICE_ASSERT(static_cast<size_t>((j * outDataDimensions) + effectiveCoordinate) < static_cast<size_t>(outputValues.size()), (j * outDataDimensions) + effectiveCoordinate, outputValues.size());
        outputValues((j * outDataDimensions) + effectiveCoordinate) = 0.0;
        for (size_t k = 0; k < _collectBands[j].size(); k++) {
          int i = _collectBands[j][k];
          PRECICE_ASSERT(i >= 0 && static_cast<size_t>(i) < inSize, i, inSize);
          PRECICE_ASSERT(static_cast<size_t>((static_cast<size_t>(i) * inDataDimensions) + effectiveCoordinate) < static_cast<size_t>(inputValues.size()), (static_cast<size_t>(i) * inDataDimensions) + effectiveCoordinate, inputValues.size());
          outputValues((j * outDataDimensions) + effectiveCoordinate) += inputValues((static_cast<size_t>(i) * inDataDimensions) + effectiveCoordinate);
        }
        outputValues((j * outDataDimensions) + effectiveCoordinate) = outputValues((j * outDataDimensions) + effectiveCoordinate) / static_cast<double>(_collectBands[j].size());

        if (outDataDimensions > 1 && _profile == SpreadProfile::PARABOLIC) {
          outputValues((j * outDataDimensions) + effectiveCoordinate) *= 9.0 / 8.0;
        }
      }
    }
  }
}

void AxialGeoMultiscaleMapping::tagMeshFirstRound()
{
  PRECICE_TRACE();

  computeMapping();

  if (getConstraint() == CONSISTENT) {
    PRECICE_ASSERT(_type == MultiscaleType::SPREAD, "Not yet implemented");
    PRECICE_ASSERT(input()->nVertices() == 1);

    input()->tagAll();

  } else {
    PRECICE_ASSERT(getConstraint() == CONSERVATIVE);
    PRECICE_UNREACHABLE("Axial conservative geometric multiscale mapping is not implemented");
  }

  clear();
}

void AxialGeoMultiscaleMapping::tagMeshSecondRound()
{
  PRECICE_TRACE();
  // no operation needed here for the moment
}

std::string AxialGeoMultiscaleMapping::getName() const
{
  return "axial-geomultiscale";
}

} // namespace precice::mapping
