#ifndef TENSORFIELD_HPP
#define TENSORFIELD_HPP

#include "tensoralg.hpp"
#include <vector>
#include <type_traits>
#include <stdexcept>
#include <cmath>

// Now we use a technique called CRTP (Curiously Recurring Template Pattern) which 
// uses static polymorphism having inheritance from the tensor class to be at compile
// time with no lookup or overhead (which is what dynamic polymorphism implies):
template <typename Derived>
struct fieldExpr {
	const Derived& self() const {return static_cast<const Derived&>(*this); }
};

template <typename Expr>
struct traitsExpr {
	using tensorType = decltype(std::declval<Expr>()[0]);
	static constexpr size_t rank = tensorType::rankVal;
	static constexpr size_t dim = tensorType::dimVal;
	using valueType = typename tensorType::Type;
};

// Now that we need a container to store the field we use the more advanced data structure
// called the SoA (Structure of Arrays) for better memory allocation and storage:
template <typename T, size_t rank, size_t dim>
class Field : public fieldExpr<Field<T, rank, dim>> {
	private:
		size_t N_;
		static constexpr size_t C = ipow(dim, rank);
		std::vector<T> data_;
	public:
		using tensorType = Tensor<T, rank, dim>;
		Field(size_t N) : N_(N), data_(N * C, 0) {}
		size_t size() const { return N_; }
		
		tensorType operator[](size_t i) const {
			tensorType t;
			for (size_t c = 0; c < C; ++c) {
				t[c] = data_[c * N_ + i]; 
			}
			return t;
		}

		void set(size_t i, const tensorType& t) {
			for (size_t c = 0; c < C; ++c) {
				data_[c * N_ + i] = t[c];
			}
		}

		T& comp(size_t c, size_t i) { return data_[c * N_ + i]; }
		const T& comp(size_t c, size_t i) const { return data_[c * N_ + i]; }

		template <typename Derived>
		Field& operator=(const fieldExpr<Derived>& Expr) {
			const auto& e = Expr.self();
			if (N_ != e.size()) {
				throw std::runtime_error("Field size mismatch during assessment.");
			}
			for (size_t i = 0; i < N_; ++i) { set(i, e[i]); }
			return *this;
		}
};

// At compile time, the operations we make between these objects are broken down sequantially, creating
// temporary fields which cost the actual processing time significantly for large meshes with many nodes,
// for this reason we do a lazy evaluation of partial expression and fo the calculation assigning an actual
// value to the expression only through the = operator:

// Addition, with it's operator overlord:
template <typename LHS, typename RHS>
class fieldAddExpr : public fieldExpr<fieldAddExpr<LHS, RHS>> {
	private:
		const LHS& lhs_;
		const RHS& rhs_;
	public: 
		fieldAddExpr(const LHS& lhs, const RHS& rhs) : lhs_(lhs), rhs_(rhs) {}
		size_t size() const { return lhs_.size(); }
		auto operator[](size_t i) const { return lhs_[i] + rhs_[i]; }
};

template <typename LHS, typename RHS>
fieldAddExpr<LHS, RHS> operator+(const fieldExpr<LHS>& lhs, const fieldExpr<RHS>& rhs) {
	static_assert(traitsExpr<LHS>::rank == traitsExpr<RHS>::rank, "Ranks must match for field addition.");
	static_assert(traitsExpr<LHS>::dim == traitsExpr<RHS>::dim, "Dimensions must match for field addition.");
	return fieldAddExpr<LHS, RHS>(lhs.self(), rhs.self());
}

// Subtraction, with it's operator overlord:
template <typename LHS, typename RHS>
class fieldSubExpr : public fieldExpr<fieldSubExpr<LHS, RHS>> {
	private:
		const LHS& lhs_;
		const RHS& rhs_;
	public: 
		fieldSubExpr(const LHS& lhs, const RHS& rhs) : lhs_(lhs), rhs_(rhs) {}
		size_t size() const { return lhs_.size(); }
		auto operator[](size_t i) const { return lhs_[i] - rhs_[i]; }
};

template <typename LHS, typename RHS>
fieldSubExpr<LHS, RHS> operator-(const fieldExpr<LHS>& lhs, const fieldExpr<RHS>& rhs) {
	static_assert(traitsExpr<LHS>::rank == traitsExpr<RHS>::rank, "Ranks must match for field subtraction.");
	static_assert(traitsExpr<LHS>::dim == traitsExpr<RHS>::dim, "Dimensions must match for field subtraction.");
	return fieldSubExpr<LHS, RHS>(lhs.self(), rhs.self());
}

// Transpose with it's operator overlord:
template <typename LHS>
class fieldTrpExpr : public fieldExpr<fieldTrpExpr<LHS>> {
	private:
		const LHS& lhs_;
	public:
		fieldTrpExpr(const LHS& lhs) : lhs_(lhs) {}
		size_t size() const { return lhs_.size(); }
		auto operator[](size_t i) const { return transpose(lhs_[i]); }	
};

template <typename LHS>
fieldTrpExpr<LHS> transpose(const fieldExpr<LHS>& lhs) {
	return fieldTrpExpr<LHS>(lhs.self());
}

template <typename LHS>
auto trp(const fieldExpr<LHS>& lhs) {
	return transpose(lhs);
}

template <typename LHS>
auto t(const fieldExpr<LHS>& lhs) {
	return transpose(lhs);
}

// Multiplication with an uniform scalar field. with it's two different overlords 
// for the two symmetric cases:
template <typename LHS, typename T>
class fieldUnfrmScaleExpr : public fieldExpr<fieldUnfrmScaleExpr<LHS, T>> {
	private:
		const LHS& lhs_;
		T scalar_;
	public:
		fieldUnfrmScaleExpr(const LHS& lhs, T scalar) : lhs_(lhs), scalar_(scalar) {}
		size_t size() const { return lhs_.size(); }
		auto operator[](size_t i) const { return lhs_[i] * scalar_; }
};

template <typename LHS, typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
fieldUnfrmScaleExpr<LHS, T> operator*(const fieldExpr<LHS>& lhs, T scalar) {
	return fieldUnfrmScaleExpr<LHS, T>(lhs.self(), scalar);
}

template <typename RHS, typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
fieldUnfrmScaleExpr<RHS, T> operator*(T scalar, const fieldExpr<RHS>& rhs) {
	return fieldUnfrmScaleExpr<RHS, T>(rhs.self(), scalar);
}

// Multiplication with a non-uniform scalar field, again with the two symmetric overlords:
template <typename LHS, typename RHS>
class fieldNonUnfrmScaleExpr : public fieldExpr<fieldNonUnfrmScaleExpr<LHS, RHS>> {
	private:
		const LHS& lhs_;
		const RHS& rhs_;
	public:
		fieldNonUnfrmScaleExpr(const LHS& lhs, const RHS& rhs) : lhs_(lhs), rhs_(rhs) {}
		size_t size() const { return lhs_.size(); }
		auto operator[](size_t i) const { return rhs_[i] * lhs_[i]; }
};

template <typename LHS, typename RHS>
std::enable_if_t<traitsExpr<LHS>::rank == 0, fieldNonUnfrmScaleExpr<LHS, RHS>>
operator*(const fieldExpr<LHS>& lhs, const fieldExpr<RHS>& rhs) {
	return fieldNonUnfrmScaleExpr<LHS, RHS>(lhs.self(), rhs.self());
}

template <typename LHS, typename RHS>
std::enable_if_t<(traitsExpr<RHS>::rank == 0 && traitsExpr<LHS>::rank > 0), fieldNonUnfrmScaleExpr<RHS, LHS>>
operator*(const fieldExpr<LHS>& lhs, const fieldExpr<RHS>& rhs) {
	return fieldNonUnfrmScaleExpr<RHS, LHS>(rhs.self(), lhs.self());
}

// Contraction with wrapper functions for the dot and double dot products:
template <size_t K, typename LHS, typename RHS>
class fieldCntrctExpr : public fieldExpr<fieldCntrctExpr<K, LHS, RHS>> {
	private:
		const LHS& lhs_;
		const RHS& rhs_;
	public:
		fieldCntrctExpr(const LHS& lhs, const RHS& rhs) : lhs_(lhs), rhs_(rhs) {}
		size_t size() const { return lhs_.size(); }
		auto operator[](size_t i) const {return contract<K>(lhs_[i], rhs_[i]); }
};

template <size_t K, typename LHS, typename RHS>
fieldCntrctExpr<K, LHS, RHS> contract(const fieldExpr<LHS>& lhs, const fieldExpr<RHS>& rhs) {
	return fieldCntrctExpr<K, LHS, RHS>(lhs.self(), rhs.self());
}

template <typename LHS, typename RHS>
auto outer(const fieldExpr<LHS>& lhs, const fieldExpr<RHS>& rhs) {
	return contract<0>(lhs, rhs);
}

template <typename LHS, typename RHS>
auto dot(const fieldExpr<LHS>& lhs, const fieldExpr<RHS>& rhs) {
	return contract<1>(lhs, rhs);
}

template <typename LHS, typename RHS>
auto inner(const fieldExpr<LHS>& lhs, const fieldExpr<RHS>& rhs) {
	return contract<2>(lhs, rhs);
}

// Finally the cross products between two vectors that by definition accepts as entries 
// only two three dimensional, rank one tensors: 
template <typename LHS, typename RHS>
class fieldCrossExpr : public fieldExpr<fieldCrossExpr<LHS, RHS>> {
	private:
		const LHS& lhs_;
		const RHS& rhs_;
	public:
		fieldCrossExpr(const LHS& lhs, const RHS& rhs) : lhs_(lhs), rhs_(rhs) {}
		size_t size() const {return lhs_.size(); }
		auto operator[](size_t i) const { return cross(lhs_[i], rhs_[i]); }
};

template <typename LHS, typename RHS>
fieldCrossExpr<LHS, RHS> cross(const fieldExpr<LHS>& lhs, const fieldExpr<RHS>& rhs) {
	static_assert(traitsExpr<LHS>::rank == 1 && traitsExpr<RHS>::rank == 1, "Cross product is defined strictly for rank one tensors (vectors).");
	static_assert(traitsExpr<LHS>::dim == 3 && traitsExpr<RHS>::dim == 3, "Cross product is defined strictly for three dimensional tensors.");
	return fieldCrossExpr<LHS, RHS>(lhs.self(), rhs.self());
}

#endif
