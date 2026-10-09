#ifndef BOUNDS_HPP
#define BOUNDS_HPP

#include "../core/tensoralg.hpp"
#include "../core/tensorfield.hpp"
#include "../geo/mesh.hpp"

template <typename T, size_t rank, size_t dim>
class boundsOperator {
	public:
		virtual ~boundsOperator() = default;
		
		// Pass the SoA 'Field' by reference and use the pre-compiled boundary indices
		// to stream the target indices sequentially with minimal cache-line misses:
		virtual void apply(Field<T, rank, dim>& field, const std::vector<size_t>& bounds_ids) = 0; 
};

#endif
