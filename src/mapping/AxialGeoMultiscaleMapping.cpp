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
    SpreadProfile       profile)
    : Mapping(constraint, dimensions, false, Mapping::InitialGuessRequirement::None),
      _dimension(dimension),
      _type(type),
      _axis(axis),
      _radius(radius),
      _profile(profile)
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
      if (_dimension == MultiscaleDimension::D1D3) {
        PRECICE_CHECK(input()->nVertices() == 1, "You can only define an axial geometric multiscale 1D-3D mapping of type spread from a mesh with exactly one vertex.");

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
        for (size_t j = 0; j < outSize; j++) {
          const Eigen::VectorXd &xOut         = output()->vertex(j).getCoords();
          double                 bestDistance = std::numeric_limits<double>::max();
          int                    bestIdx      = -1;
          for (size_t i = 0; i < inSize; i++) {
            const Eigen::VectorXd &xIn        = input()->vertex(i).getCoords();
            Eigen::VectorXd        difference = xOut - xIn;
            double                 distance   = difference.squaredNorm();
            if (distance < bestDistance) {
              bestDistance = distance;
              bestIdx      = static_cast<int>(i);
            }
          }
          PRECICE_ASSERT(bestIdx >= 0, "Could not find nearest input vertex for output vertex {}", j);
          _nearestVertex.push_back(bestIdx);
        }
      }
    } else {
      PRECICE_ASSERT(_type == MultiscaleType::COLLECT);
      size_t const inSize  = input()->nVertices();
      size_t const outSize = output()->nVertices();
      if (_dimension == MultiscaleDimension::D1D3) {
        PRECICE_CHECK(output()->nVertices() == 1, "You can only define an axial geometric multiscale 1D-3D mapping of type collect to a mesh with exactly one vertex.");
        // Nothing to do here: A consistent collect mapping only averages all the values, independently of their locations, and this is done in the mapConsistent() method.
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
    if (_dimension == MultiscaleDimension::D1D3) {
      PRECICE_ASSERT(input()->nVertices() == 1);
      for (size_t i = 0; i < outSize; i++) {
        PRECICE_ASSERT(static_cast<size_t>((i * outDataDimensions) + effectiveCoordinate) < static_cast<size_t>(outputValues.size()), ((i * outDataDimensions) + effectiveCoordinate), outputValues.size());
        if (_profile == SpreadProfile::PARABOLIC) {
          // When adding support for 2D, remember that this should be 1.5 * inputValues(effectiveCoordinate) * (1 - (_vertexDistances[i] * _vertexDistances[i]));
          outputValues((i * outDataDimensions) + effectiveCoordinate) = 2 * inputValues(effectiveCoordinate) * (1 - (_vertexDistances[i] * _vertexDistances[i]));
        } else if (_profile == SpreadProfile::UNIFORM) {
          outputValues((i * outDataDimensions) + effectiveCoordinate) = inputValues(effectiveCoordinate);
        }
      }
    } else {
      PRECICE_ASSERT(_dimension == MultiscaleDimension::D2D3);
      PRECICE_ASSERT(_nearestVertex.size() == outSize);
      for (size_t i = 0; i < outSize; i++) {
        PRECICE_ASSERT(_nearestVertex[i] >= 0 && static_cast<size_t>(_nearestVertex[i]) < inSize, _nearestVertex[i], inSize);
        PRECICE_ASSERT(static_cast<size_t>((i * outDataDimensions) + effectiveCoordinate) < static_cast<size_t>(outputValues.size()), (i * outDataDimensions) + effectiveCoordinate, outputValues.size());
        PRECICE_ASSERT(static_cast<size_t>((static_cast<size_t>(_nearestVertex[i]) * inDataDimensions) + effectiveCoordinate) < static_cast<size_t>(inputValues.size()), (static_cast<size_t>(_nearestVertex[i]) * inDataDimensions) + effectiveCoordinate, inputValues.size());

        outputValues((i * outDataDimensions) + effectiveCoordinate) = inputValues((static_cast<size_t>(_nearestVertex[i]) * inDataDimensions) + effectiveCoordinate);
      }
    }
  } else {
    PRECICE_ASSERT(_type == MultiscaleType::COLLECT);
    size_t const inSize  = input()->nVertices();
    size_t const outSize = output()->nVertices();
    if (_dimension == MultiscaleDimension::D1D3) {
      PRECICE_ASSERT(output()->nVertices() == 1);
      outputValues(effectiveCoordinate) = 0;
      for (size_t i = 0; i < inSize; i++) {
        PRECICE_ASSERT(static_cast<size_t>((i * inDataDimensions) + effectiveCoordinate) < static_cast<size_t>(inputValues.size()),
                       ((i * inDataDimensions) + effectiveCoordinate), inputValues.size());
        outputValues(effectiveCoordinate) += inputValues((i * inDataDimensions) + effectiveCoordinate);
      }
      outputValues(effectiveCoordinate) = outputValues(effectiveCoordinate) / inSize;
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
