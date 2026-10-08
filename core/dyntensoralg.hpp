#ifndef DYNTENSORALG_HPP
#define DYNTENSORALG_HPP

#include <vector>
#include <iostream>
#include <string>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <cctype>

class DynamicTensor {
	private:
		std::vector<double> data_;
		std::vector<size_t> shape_;
		std::vector<size_t> strides_;
		std::string name_ = "ans";

		void comp_strides() {
			strides_.resize(shape_.size());
			if (shape_.empty()) return;
			strides_.back() = 1;
			for (int i = static_cast<int>(shape_.size()) - 2; i >= 0; --i) {
				strides_[i] = strides_[i + 1] * shape_[i + 1];
			}
		}

		static size_t calc_size(const std::vector<size_t>& shape) {
			if (shape.empty()) return 1;
			size_t s = 1;
			for (size_t d : shape) s *= d;
			return s;
		}
	public:
		DynamicTensor() : data_(1, 0.0), shape_({}), strides_({}) {}

		// Constructor for albitrary dynamic shape during runtime:
		DynamicTensor(const std::vector<size_t>& shape) : shape_(shape) {
			data_.assign(calc_size(shape_), 0.0);
			comp_strides();
		}

		DynamicTensor(const std::vector<size_t>& shape, const std::vector<double>& flat_data) : data_(flat_data), shape_(shape) {
			if (data_.size() != calc_size(shape_)) {
				throw std::runtime_error("Provided flat data size does not match requested dynamic shape.");
			}
			comp_strides();
		}

		size_t rank() const { return shape_.size(); }
		size_t size() const { return data_.size(); }
		
		const std::vector<size_t>& shape() const { return shape_; }
		
		const std::vector<double>& data() const { return data_;}
		std::vector<double>& data() { return data_; }
		
		std::string name() const { return name_; }
		void set_name(const std::string& name) {name_ = name; }

		double operator[](size_t i) const {return data_[i]; }
		double& operator[](size_t i) {return data_[i]; }

		// Dynamic muit-index accessor using circle brackets:
		double operator()(const std::vector<size_t>& indices) const {
			if (indices.size() != shape_.size()) {
				throw std::runtime_error("Index count mismatch.");
			}
			size_t flat_idx = 0;
			for (size_t d = 0; d < indices.size(); ++d) {
				flat_idx += indices[d] * strides_[d];
			}
			return data_[flat_idx];
		}

		// Dynamic element-wise addition:
		DynamicTensor operator+(const DynamicTensor& other) const {
			if (shape_ != other.shape_) {
				throw std::runtime_error("Dynamic shape mismatch in addition.");
			}
			DynamicTensor res(shape_);
			for (size_t i = 0; i < data_.size(); ++i) {
				res.data_[i] = data_[i] + other.data_[i];
			}
			return res;
		}

		// Dynamic element-wise subtraction:
		DynamicTensor operator-(const DynamicTensor& other) const {
			if (shape_ != other.shape_) {
				throw std::runtime_error("Dynamic shape mismatch in subtraction.");
			}
			DynamicTensor res(shape_);
			for (size_t i = 0; i < data_.size(); ++i) {
				res.data_[i] = data_[i] - other.data_[i];
			}
			return res;
		}

		// Dynamic scaling by a scalar:
		DynamicTensor operator*(double scalar) const {
			DynamicTensor res(shape_);
			for (size_t i = 0; i < data_.size(); ++i) {
				res.data_[i] = data_[i] * scalar;
			}
			return res;
		}

		// helper function to print at terminal:
		void print() const {
			size_t r = rank();
			std::cout << name_ << " (Rank: " << r << ", Shape [";
			for (size_t d = 0; d < r; ++d) {
				std::cout << shape_[d] << (d + 1 < r ? "," : "");
			}
			std::cout << "]): ";

			if (r == 0) {
				std::cout << data_[0] << "\n";
			} else if (r == 1) {
				std::cout << "[ ";
				for (double val : data_) std::cout << val << " ";
				std::cout << "]\n";
			} else if (r == 2) {
				std::cout << "\n";
				size_t rows = shape_[0];
				size_t cols = shape_[1];
				for (size_t i = 0; i < rows; ++i) {
					std::cout << "	[ ";
					for (size_t j = 0; j < cols; ++j) {
						std::cout << data_[i * cols + j] << " ";
					}
					std::cout << "]\n";
				}
			} else {
				std::cout << " [ Flat: ";
				for (double val : data_) std::cout << val << " ";
				std::cout << "]\n";
			}
		}
};

// Dynamic tensor contract, transpose and cross product:
inline DynamicTensor contract(const DynamicTensor& a, const DynamicTensor& b, size_t K) {
	size_t rankA = a.rank();
	size_t rankB = b.rank();
	if (rankA < K || rankB < K) {
		throw std::runtime_error("Cannot contract more indices than the tensors max rank.");
	}
	for (size_t i = 0; i < K; ++i) {
		if (a.shape()[rankA - K + i] != b.shape()[i]) {
			throw std::runtime_error("Shape mismatch in dynamic index constraction.");
		}
	}

	std::vector<size_t> res_shape;
	for (size_t i = 0; i < rankA - K; ++i) res_shape.push_back(a.shape()[i]);
	for (size_t i = K; i < rankB; ++i) res_shape.push_back(b.shape()[i]);

	DynamicTensor res(res_shape);

	size_t dim_to_rank = 1;
	for (size_t i = 0; i < K; ++i) dim_to_rank *= b.shape()[i];

	size_t N_non_contracted_B = 1;
	for (size_t i = K; i < rankB; ++i) N_non_contracted_B *= b.shape()[i];	
		
	for (size_t idx_res = 0; idx_res < res.size(); ++idx_res) {
		size_t idx_non_contracted_A = idx_res / N_non_contracted_B;
		size_t idx_non_contracted_B = idx_res % N_non_contracted_B;
		double sum = 0.0;

		for (size_t c = 0; c < dim_to_rank; ++c) {
			size_t idx_a = idx_non_contracted_A * dim_to_rank + c;
			size_t idx_b = c * N_non_contracted_B + idx_non_contracted_B;
			sum += a[idx_a] * b[idx_b];
		}
		res.data()[idx_res] = sum;
	}
	return res;
}

inline DynamicTensor transpose(const DynamicTensor& a) {
	size_t r = a.rank();
	std::vector<size_t> res_shape(r);
	for (size_t i = 0; i < r; ++i) {
		res_shape[i] = a.shape()[r - 1 - i];
	}
	DynamicTensor res(res_shape);

	for (size_t idx_out = 0; idx_out < res.size(); ++idx_out) {
		size_t tmp = idx_out;
		std::vector<size_t> multi_idx(r);
		for (int d = static_cast<int>(r) - 1; d >= 0; --d) {
			multi_idx[d] = tmp % res.shape()[d];
			tmp /= res.shape()[d];
		}
		size_t idx_in = 0;
		size_t multiplier = 1;
		for (size_t d = 0; d < r; ++d) {
			idx_in += multi_idx[d] * multiplier;
			multiplier *= a.shape()[d];
		}
		res[idx_out] = a[idx_in];
	}
	return res;
}

inline DynamicTensor cross(const DynamicTensor& a, const DynamicTensor& b) {
	if (a.rank() != 1 || b.rank() != 1 || a.size() != 3 || b.size() != 3) {
		throw std::runtime_error("Cross product is well defined only between 3-Dimensional Vectors");
	}
	DynamicTensor res({3});
	res[0] = a[1] * b[2] - a[2] * b[1];
	res[1] = a[2] * b[0] - a[0] * b[2];
	res[2] = a[0] * b[1] - a[1] * b[0];
	return res;
}
// Dynamic non-uniform scalar multiplication:
inline DynamicTensor operator*(const DynamicTensor& lhs, const DynamicTensor& rhs) {
	if (lhs.rank() == 0) {
		return rhs * lhs[0];
	} else if (rhs.rank() == 0) {
		return lhs * rhs[0];
	} else {
		throw std::runtime_error("Dynamic multiplication by a non-uniform scalar requires at least one scalar.");
	}
}

// Semantic wrappers:
inline DynamicTensor t(const DynamicTensor& a) {
	return transpose(a);
}

inline DynamicTensor trp(const DynamicTensor& a) {
	return transpose(a);
}

inline DynamicTensor outer(const DynamicTensor& a, const DynamicTensor& b) {
	return contract(a, b, 0);
}

inline DynamicTensor dot(const DynamicTensor& a, const DynamicTensor& b) {
	return contract(a, b, 1);
}

inline DynamicTensor inner(const DynamicTensor& a, const DynamicTensor& b) {
	return contract(a, b, 2);
}

// Dynamic bracket interpreter engine to understand what rank and dimension of vector we are
// declaring from the terminal window soley from the syntax used (which is universal):

namespace Parser {
	
	inline std::string sanitize(const std::string& str) {
		std::string clean;
		bool last_was_space = false;
		for (char c : str) {
			if (std::isspace(c)) {
				if (!clean.empty() && !last_was_space) {
					clean += ' ';
					last_was_space = true;
				}
			} else {
				clean += c;
				last_was_space = false;
			}
		}
		if (!clean.empty() && clean.back() == ' ') clean.pop_back();
		return clean;
	}

	inline std::vector<double> extract_flat_double(const std::string& str) {
		std::vector<double> vals;
		std::string buffer;
		for (char c : str) {
			if (c == '[' || c == ']' || c == ',' || c == ';') {
				if (!buffer.empty()) {
					std::stringstream ss(buffer);
					double val;
					if (ss >> val) {
						vals.push_back(val);
						while (ss >> val) vals.push_back(val);
					}
					buffer.clear();
				}
			} else {
				buffer += c;
			}
		}
		if (!buffer.empty()) {
			std::stringstream ss(buffer);
			double val;
			if (ss >> val) {
				vals.push_back(val);
				while (ss >> val) vals.push_back(val);				
			}
		}
		return vals;
	}

	inline std::vector<std::string> split_elements(const std::string& s) {
		std::vector<std::string> elems;
		std::string current;
		size_t bracket_count = 0;

		for (char c : s) {
			if (c == '[') {
				bracket_count++;
				current += c;
			} else if (c == ']') {
				bracket_count--;
				current += c;
				if (bracket_count == 0 && !current.empty()) {
					elems.push_back(current);
					current.clear();
				}
			} else if (std::isspace(c) || c == ',' || c == ';') {
				if (bracket_count > 0) {
					current += c;
				} else {
					if (!current.empty()) {
						elems.push_back(current);
						current.clear();
					}
				}
			} else {
				current += c;
			}
		}
		if (!current.empty()) {
			elems.push_back(current);	
		}
		return elems;
	}

	inline std::vector<size_t> deduce_shape(std::string s) {
		s = sanitize(s);
		if (s.empty()) return {};
		if (s.front() == '[' && s.back() == ']') {
			s = s.substr(1, s.size() - 2);
			s = sanitize(s);
			auto elems = split_elements(s);
			if (elems.empty()) return {};
			std::vector<size_t> shape;
			shape.push_back(elems.size());

			auto sub_shape = deduce_shape(elems[0]);
			shape.insert(shape.end(), sub_shape.begin(), sub_shape.end());
			return shape;
		} else {
			return {};
		}
	}

	inline DynamicTensor parse(const std::string& expr) {
		std::string clean = sanitize(expr);
		auto shape = deduce_shape(clean);
		auto values = extract_flat_double(clean);
		if (values.empty()) throw std::runtime_error("No values found inside array expression.");
		return DynamicTensor(shape, values);
	}
}

#endif
