#include "interpreter.hpp"
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <sstream>
#include <algorithm>
#include <cctype>
#include <chrono>

#include <readline/readline.h>
#include <readline/history.h>
#include <cstdlib>

void run_live_bench(const std::string& v1_name, const std::string& v2_name, const std::string& m_name, const TensorInterpreter& interpreter) {
	// Fetch dymanic variables from the interpreter	
	const auto& d_v1 = interpreter.get_variable(v1_name);
	const auto& d_v2 = interpreter.get_variable(v2_name);
	const auto& d_m = interpreter.get_variable(m_name);

	std::cout << "\n[Acquiring Tensors...] Converting live dynamic variables into stack-allocated static types...\n";

	auto s_v1 = acquire_tensor<double, 1, 3>(d_v1);
	auto s_v2 = acquire_tensor<double, 1, 3>(d_v2);
	auto s_m  = acquire_tensor<double, 2, 3>(d_m);

	constexpr size_t ITERS = 1000000;
	std::cout << "\n[Running Benckmark...] Executing " << ITERS << " operations on your live inputs...\n\n";

	// Addition
	auto start_s = std::chrono::high_resolution_clock::now();
	Tensor<double, 1, 3> s_v_res;
	for (size_t i = 0; i < ITERS; ++i) s_v_res = s_v1 + s_v2;
	auto end_s = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> t_s1 = end_s - start_s;

	auto start_d = std::chrono::high_resolution_clock::now();
	DynamicTensor d_v_res;
	for (size_t i = 0; i < ITERS; ++i) d_v_res = d_v1 + d_v2;
	auto end_d = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> t_d1 = end_d - start_d;

	std::cout << "  - Vector Addition (" << v1_name << " + " << v2_name << "):\n"
		  << "  * Static Time:    " << t_s1.count() << " ms\n"
		  << "  * Dynamic Time:   " << t_d1.count() << " ms\n"
		  << "  * Speedup         " << (t_d1.count() / t_s1.count()) << "x\n\n";

	// Contraction
	start_s = std::chrono::high_resolution_clock::now();
	for (size_t i = 0; i < ITERS; ++i) s_v_res = contract<1>(s_m, s_v1);
	end_s = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> t_s2 = end_s - start_s;

	start_d = std::chrono::high_resolution_clock::now();
	for (size_t i = 0; i < ITERS; ++i) d_v_res = contract(d_m, d_v1, 1);
	end_d = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> t_d2 = end_d - start_d;

	std::cout << "  - Contraction (" << m_name << " * " << v1_name << "):\n"
		  << "  * Static Time:    " << t_s2.count() << " ms\n"
		  << "  * Dynamic Time:   " << t_d2.count() << " ms\n"
		  << "  * Speedup         " << (t_d2.count() / t_s2.count()) << "x\n\n";

	// Double Constraction and Transpose
	start_s = std::chrono::high_resolution_clock::now();
	Tensor<double, 0, 3> s_sc_res;
	for (size_t i = 0; i < ITERS; ++i) s_sc_res = inner(s_m, t(s_m));
	end_s = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> t_s3 = end_s - start_s;

	start_d = std::chrono::high_resolution_clock::now();
	DynamicTensor d_sc_res;
	for (size_t i = 0; i < ITERS; ++i) d_sc_res = inner(d_m, t(d_m));
	end_d = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> t_d3 = end_d - start_d;

	std::cout << "  - Transpose and Double Contraction (" << m_name << " : " << m_name << "^T):\n"
		  << "  * Static Time:    " << t_s3.count() << " ms\n"
		  << "  * Dynamic Time:   " << t_d3.count() << " ms\n"
		  << "  * Speedup         " << (t_d3.count() / t_s3.count()) << "x\n\n";
}

int main() {
	TensorInterpreter interpreter;

	std::cout << "\n	=====================================\n"
		  << "	 Linear Tensor Algebra Live Terminal\n"
		  << "	=====================================\n"
		  << "	 * Type 'exit', 'quit' or 'q' to terminate\n"
		  << "	 * Supported benchmark tensors dimensions: 3\n"
		  << "	 * Execute live benchmark using: bench <v1> <v2> <M>\n\n";

	char* buf = nullptr;
	while ((buf = readline(">>> ")) != nullptr) {
		std::string line(buf);
		free(buf);

		std::string sanitized = Parser::sanitize(line);
		if (sanitized.empty()) continue;
		
		add_history(sanitized.c_str());

		if (sanitized == "exit" || sanitized == "quit" || sanitized == "q") {
			std::cout << "Closing session..." << std::endl;
			break;
		}

		if (sanitized.rfind("bench", 0) == 0) {
			std::stringstream ss(sanitized.substr(6));
			std::string v1, v2, m;
			if (ss >> v1 >> v2 >> m) {
				try {
					run_live_bench(v1, v2, m, interpreter);
				} catch (const std::exception& e) {
					std::cerr << "Benchmark error: " << e.what() << "\n";
				}
			} else {
				std::cout << "Usage: bench <v1> <v2> <M>\n";
			}
			continue;
		}
		
		try {
			interpreter.execute_line(sanitized);		
		} catch (const std::exception& e) {
			std::cerr << "Error: " << e.what() << "\n";
		}
		std::cout << "\n";
	}
	return 0;
}
