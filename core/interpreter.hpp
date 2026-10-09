#ifndef INTERPRETER_HPP
#define INTERPRETER_HPP

#include "tensoralg.hpp"
#include "dyntensoralg.hpp"
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <sstream>
#include <algorithm>
#include <cctype>
#include <memory>
#include <iomanip>

#include <readline/readline.h>
#include <readline/history.h>
#include <cstdlib>

// Compile time acquisiton bridge to turn dynamic tensors allocate don the heap to
// static tensors that are allocated to the stack, this induced less cash misses one
// we work with this second kind of objects:
template <typename T, size_t rank, size_t dim>
Tensor<T, rank, dim> acquire_tensor(const DynamicTensor& d_tensor) {
	if (d_tensor.rank() != rank) {
		throw std::runtime_error("Acquire Mismatch: Rank error. Expected Rank " + std::to_string(rank) + ", got " + std::to_string(d_tensor.rank()));
	}
	if constexpr (rank > 0) {
		for (size_t d = 0; d < rank; ++d) {
			if (d_tensor.shape()[d] != dim) {
				throw std::runtime_error("Acquire Mismatch: Dimension error at axis " + std::to_string(d) + ". Expected " + std::to_string(dim));
			}
		}
	}
	Tensor<T, rank, dim> s_tensor;
	std::copy(d_tensor.data().begin(), d_tensor.data().end(), s_tensor.data.begin());
	return s_tensor;
}

// And here we define the interpreter class, which manadges the storage of the symnols table,
// the evaluation of all expressions contianing all defined operations that cane be provided as
// terminal imput using the dynamic tensor class:
class TensorInterpreter {
	private:
		std::map<std::string, DynamicTensor> symbol_table_;

		std::string strip_all_spaces(const std::string& str) {
			std::string clean;
			for (char c : str) {
				if (!std::isspace(c)) clean += c;
			}
			return clean;
		}

		DynamicTensor evaluate_expr(const std::string& expr) {
			std::string s = strip_all_spaces(expr);

			auto resolve = [&](const std::string& name) -> DynamicTensor {
				auto it = symbol_table_.find(name);
				if (it != symbol_table_.end()) return it->second;
				try {
					double val = std::stod(name);
					return DynamicTensor({}, {val});
				} catch (...) {
					throw std::runtime_error("Undefined variable of invalid number: " + name);
				}
			};
			
			// Unpack semantic wrappers for transpose:
			if (s.find("t(", 0) == 0 && s.back() == ')') {
				return t(evaluate_expr(s.substr(2, s.size() - 3)));
			}
			if (s.find("trp(", 0) == 0 && s.back() == ')') {
				return trp(evaluate_expr(s.substr(4, s.size() - 5)));
			}
			if (s.find("transpose(", 0) == 0 && s.back() == ')') {
				return transpose(evaluate_expr(s.substr(10, s.size() - 11)));
			}
			
			// Unpack semantic wrappers for contract:
			if (s.find("outer(", 0) == 0 && s.back() == ')') {
				std::string inner_args = s.substr(6, s.size() - 7);
				size_t comma = inner_args.find(',');
				if (comma == std::string::npos) throw std::runtime_error("Invalid outer() arguments.");
				return outer(evaluate_expr(inner_args.substr(0, comma)), evaluate_expr(inner_args.substr(comma + 1)));
			}
			if (s.find("dot(", 0) == 0 && s.back() == ')') {
				std::string inner_args = s.substr(4, s.size() - 5);
				size_t comma = inner_args.find(',');
				if (comma == std::string::npos) throw std::runtime_error("Invalid dot() arguments.");
				return dot(evaluate_expr(inner_args.substr(0, comma)), evaluate_expr(inner_args.substr(comma + 1)));
			}
			if (s.find("inner(", 0) == 0 && s.back() == ')') {
				std::string inner_args = s.substr(6, s.size() - 7);
				size_t comma = inner_args.find(',');
				if (comma == std::string::npos) throw std::runtime_error("Invalid dotdot() arguments.");
				return inner(evaluate_expr(inner_args.substr(0, comma)), evaluate_expr(inner_args.substr(comma + 1)));
			}

			// Unpack cross product:
			if (s.find("cross(", 0) == 0 && s.back() == ')') {
				std::string inner_args = s.substr(6, s.size() - 7);
				size_t comma = inner_args.find(',');
				if (comma == std::string::npos) throw std::runtime_error("Invalid cross() arguments.");
				return cross(evaluate_expr(inner_args.substr(0, comma)), evaluate_expr(inner_args.substr(comma + 1)));
			}

			size_t op_pos = std::string::npos;
			char op = ' ';
			for (int i = static_cast<int>(s.size()) - 1; i >= 0; --i) {
				if (s[i] == '+' || s[i] == '-' || s[i] == '*') {
					size_t open_p = 0;
					for (size_t j = 0; j < static_cast<size_t>(i); ++j) {
						if (s[j] == '(') open_p++;
						if (s[j] == ')') open_p--;
					}
					if (open_p == 0) {
						op_pos = i;
						op = s[i];
						break;
					}
				}
			}

			if (op_pos != std::string::npos) {
				DynamicTensor lhs = evaluate_expr(s.substr(0, op_pos));
				DynamicTensor rhs = evaluate_expr(s.substr(op_pos + 1));
				if (op == '+') return lhs + rhs;
				if (op == '-') return lhs - rhs;
				if (op == '*') return lhs * rhs;
			}

			return resolve(s);
		}
	public:
		TensorInterpreter() = default;
		
		// Public API to execute a single command line dynamically:
		void execute_line(const std::string& sanitized) {
			size_t eq_pos = sanitized.find("=");
			if (eq_pos != std::string::npos) {
				std::string var_name = strip_all_spaces(sanitized.substr(0, eq_pos));
				std::string expr = sanitized.substr(eq_pos + 1);

				if (expr.find('[') != std::string::npos) {
					DynamicTensor dt = Parser::parse(expr);
					dt.set_name(var_name);
					symbol_table_[var_name] = dt;
					dt.print();
				} else {
					DynamicTensor dt = evaluate_expr(expr);
					dt.set_name(var_name);
					symbol_table_[var_name] = dt;
					dt.print();
				}
			} else {
				DynamicTensor dt = evaluate_expr(sanitized);
				dt.print();
			}
		}

		const DynamicTensor& get_variable(const std::string& name) const {
			auto it = symbol_table_.find(name);
			if (it == symbol_table_.end()) {
				throw std::runtime_error("Variable not found in interpreter memory: " + name);
			}
			return it->second;
		}
};

#endif
