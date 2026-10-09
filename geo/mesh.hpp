#ifndef MESH_HPP
#define MESH_HPP

#include "../core/tensoralg.hpp"
#include <string>
#include <vector>

template <size_t, dim>
class meshInterface {
	public:
		virtual ~meshInterface() = default;
		
		// We use virtual function (dynamic polymorphism) to make any solver 
		// recieve any structured/unstructured mesh type:
		
		// Number of mesh nodes:
		virtual size_t num_nodes() const = 0;
		
		// Coordinates of the mesh nodes:
		virtual Tensor<double, 1, dim> get_coord(size_t, i) const = 0;
		
		// Boolean for the existance of a boundary:
		virtual bool has_bound(const std::string& name) const = 0;
		
		// Boundary indices stored contiguously for cache-coherence:
		virtual const std::vector<size_t>& get_bound(const std::string& name) const = 0;
};

#endif
