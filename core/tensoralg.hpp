#ifndef TENSORALG_HPP
#define TENSORALG_HPP

#include <iostream>
#include <vector>
#include <array>
#include <cmath>
#include <type_traits>
#include <stdexcept>

// Generic tensor object, with dim and rank to be specified, it is only the algrbraic
// part of this entity since it has not yet any info on it's grid position: 

// Helper to obtain the tensor size at compile time:
constexpr size_t ipow(size_t base, size_t exp) {
	size_t result = 1;
	for (size_t i = 0; i < exp; ++i) result *= base;
	return result;
}

// This static object is stack allocated to ensure optimization in it's memory occupation:
template <typename T, size_t rank, size_t dim>
class Tensor {
	public:
		static constexpr size_t rankVal = rank;
		static constexpr size_t dimVal = dim;
		using Type = T;

		// Use the helper function to extarct the tensor size:
		static constexpr size_t size = ipow(dim, rank);
		// And define the data as a compile time array on the stack:
		std::array<T, size> data{};

		// Access operations: 
		// One dimensional case:
		T& operator[](size_t i) { return data[i]; }
		const T& operator[](size_t i) const { return data[i]; }

		// Multi-dimesional case using variadic intex numbering e.g. T(i, j) vs T(i, j, k):
		template <typename... indices, typename = std::enable_if_t<sizeof...(indices) == rank>>
		T& operator() (indices... indicesVal) {
			return data[get_flat_indices(indicesVal...)];
		}
		template <typename... indices, typename = std::enable_if_t<sizeof...(indices) == rank>>
		const T& operator() (indices... indicesVal) const {
			return data[get_flat_indices(indicesVal...)];
		}

		// The trivial scalar case:
		template <typename U = T, typename = std::enable_if_t<rank == 0, U>>
		operator T() const { return data[0]; }
		
		template <typename U = T, typename = std::enable_if_t<rank == 0, U>>
		operator T&() { return data[0]; }
	
	private:
		template <typename... indices>
		size_t get_flat_indices(indices... indicesVal) const {
			std::array<size_t, rank> 
			idx_array{static_cast<size_t>(indicesVal)...};
			size_t flat_idx = 0;
			size_t multiplier = 1;
			for (int d = static_cast<int>(rank) - 1; d >= 0; --d) {
				flat_idx += idx_array[d] * multiplier;
				multiplier *= dim;
			}
			return flat_idx;
		}	       
};

// Operations between these objects:
// We implement a generalized transpose operation, this is needed to complete the tensor 
// algebra making non-symmetric operations doable:
template <typename T, size_t rank, size_t dim> 
Tensor<T, rank, dim> transpose(const Tensor<T, rank, dim>& t) {
	Tensor<T, rank, dim> out;
	for (size_t idx_out = 0; idx_out < Tensor<T, rank, dim>::size; ++idx_out) {
		std::array<size_t, rank> multi_idx{};
		
		// unpack flat output into coordinates:
		size_t tmp = idx_out;
		for (int d = static_cast<int>(rank) - 1; d >= 0; --d) {
			multi_idx[d] = tmp % dim;
			tmp /= dim;
		}

		// Reverse coordinate for source mapping:
		size_t idx_in = 0;
		size_t multiplier = 1;
		for (size_t d = 0; d < rank; ++d) {
			idx_in += multi_idx[d] * multiplier;
			multiplier *= dim;
		}

		out[idx_out] = t[idx_in];
	}
	return out;
}

// Semantic wrappers for raw tensor data:
template <typename T, size_t rank, size_t dim>
Tensor<T, rank, dim> trp(const Tensor<T, rank, dim>& tensor) {
	return transpose(tensor);
}

template <typename T, size_t rank, size_t dim>
Tensor<T, rank, dim> t(const Tensor<T, rank, dim>& tensor) {
	return transpose(tensor);
}

// We implement a general product between two tensors, which can be seen as a contraction
// of one or more indices. It is by definition rank dependent and takes the name of inner 
// product when we have same rank tensors contracted on all indices like $\bm{a}\cdot\bm{b}$
// or $\bm{A}:\bm{B}$, it's a simple 'matrix' multiplipication when we are left with a result
// that has rank >= 1, like $\bm{A}\cdot\bm{a} = \bm{b}$ or $\bm{A}\cdot\bm{B} = \bm{C}$ but 
// is generic to work for any number of indixes > 2:
template <size_t K, typename T, size_t rankA, size_t rankB, size_t dim>
Tensor<T, rankA + rankB - 2 * K, dim> contract(const Tensor<T, rankA, dim>& a, const Tensor<T, rankB, dim>& b) {
	
	// K is the number of contracted indices
	static_assert(rankA >= K && rankB >= K, "Cannot contract more indices than the tensors max rank.");
	constexpr size_t rankRes = rankA + rankB - 2 * K;
	Tensor<T, rankRes, dim> res;
	constexpr size_t N_non_contracted_B = ipow(dim, rankB - K);
	constexpr size_t dim_to_rank = ipow(dim, K);
	constexpr size_t sizeRes = ipow(dim, rankRes);
	
	// We loop over the flat output array:
	for (size_t idx_res = 0; idx_res < sizeRes; ++idx_res) {
		size_t idx_non_contracted_A = idx_res / N_non_contracted_B;
		size_t idx_non_contracted_B = idx_res % N_non_contracted_B;
		T sum = 0;
		
		// Here we loop over the contracted indices
		for (size_t c = 0; c < dim_to_rank; ++c) {
			size_t idx_a = idx_non_contracted_A * dim_to_rank + c;
			size_t idx_b = c * N_non_contracted_B + idx_non_contracted_B;
			sum += a[idx_a] * b[idx_b];
		}
		res[idx_res] = sum;
	}
	return res;
}

// Semantic wrappers for raw tensor data: we decide to define the scalar product between two 
// vectors as dot() between two tensors as inner() and the tensor product as outer:
template <typename T, size_t rankA, size_t rankB, size_t dim>
Tensor<T, rankA + rankB, dim> outer(const Tensor<T, rankA, dim>& a, const Tensor<T, rankB, dim>& b) {
	return contract<0>(a, b);
}

template <typename T, size_t rankA, size_t rankB, size_t dim>
Tensor<T, rankA + rankB - 2, dim> dot(const Tensor<T, rankA, dim>& a, const Tensor<T, rankB, dim>& b) {
	return contract<1>(a, b);
}

template <typename T, size_t rankA, size_t rankB, size_t dim>
Tensor<T, rankA + rankB - 4, dim> inner(const Tensor<T, rankA, dim>& a, const Tensor<T, rankB, dim>& b) {
	return contract<2>(a, b);
}

// Element-wise addition:
template <typename T, size_t rank, size_t dim>
Tensor<T, rank, dim> operator+(const Tensor<T, rank, dim>& a, const Tensor<T, rank, dim>& b) {
	Tensor<T, rank, dim> res;
	for (size_t i = 0; i < Tensor<T, rank, dim>::size; ++i) { res[i] = a[i] + b[i]; }
	return res;
}

// Element-wise subtraction:
template <typename T, size_t rank, size_t dim>
Tensor<T, rank, dim> operator-(const Tensor<T, rank, dim>& a, const Tensor<T, rank, dim>& b) {
	Tensor<T, rank, dim> res;
	for (size_t i = 0; i < Tensor<T, rank, dim>::size; ++i) { res[i] = a[i] - b[i]; }
	return res;
}

// Scaling by an uniform scalar, distinguished to allocat less memory to the scalar field:
template <typename T, size_t rank, size_t dim>
Tensor<T, rank, dim> operator*(const Tensor<T, rank, dim>& a, T s) {
	Tensor<T, rank, dim> res;
	for (size_t i = 0; i < Tensor<T, rank, dim>::size; ++i) { res[i] = a[i] * s; }
	return res;
}
template <typename T, size_t rank, size_t dim>
Tensor<T, rank, dim> operator*(T s, const Tensor<T, rank, dim>& a) {
	return a * s;
}

// Scaling by a non-uniform scalar field:
template <typename T, size_t rank, size_t dim>
Tensor<T, rank, dim> operator*(const Tensor<T, rank, dim>& a, const Tensor<T, 0, dim>& s) {
	return a * s[0];
}
template <typename T, size_t rank, size_t dim>
Tensor<T, rank, dim> operator*(const Tensor<T, 0, dim>& s, const Tensor<T, rank, dim>& a) {
	return a * s[0];
}

// cross product between two 3 dimensional vectors (skew symmetric):
template <typename T>
Tensor<T, 1, 3> cross(const Tensor<T, 1, 3>& a, const Tensor<T, 1, 3>& b) {
	Tensor<T, 1, 3> res;
	res[0] = a[1] * b[2] - a[2] * b[1];
	res[1] = a[2] * b[0] - a[0] * b[2];
	res[2] = a[0] * b[1] - a[1] * b[0];
	return res;
}

#endif
