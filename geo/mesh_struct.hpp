#ifndef MESH_STRUCT_HPP
#define MESH_STRUCT_HPP

#include "../core/tensoralg.hpp"
#include "mesh.hpp"
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <map>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <iostream>

template <size_t dim>
class meshStructured : public meshInterface<dim> {
	private:
		size_t Nx_ = 0;
		size_t Ny_ = 0;
		size_t Nz_ = 1;
		double dx_ = 0.0;

		std::vector<Tensor<double, 1, dim>> nodes_;
		std::vector<uint32_t> flags_;

		std::map<std::string, uint32_t> bound_names_to_ids_;
		std::map<uint32_t, std::string> bound_ids_to_names_;
		std::map<std::string, std::vector<size_t>> bound_ids_;
		
		// .msh file defines boundary names with quotes:
		std::string strip_quotes(const std::string& str) const {
			std::string clean = str;
			if (!clean.empty() && clean.front() == '"') {
				clean.erase(0, 1);
			}
			if (!clean.empty() && clean.back() == '"') {
				clean.pop_back();
			}
			return clean;
		}
	public:
		meshStructured() = default;
		
		// 2D constructor:
		meshStructured(size_t nx, size_t ny, double dx) : Nx_(nx), Ny_(ny), Nz_(1), dx_(dx) {
			static_assert(dim == 2, "This constructor is strictly for 2D grids.");
			size_t tot_nodes = Nx_ * Ny_;
			nodes_.resize(tot_nodes);
			flags_.resize(tot_nodes, 0);

			for (size_t y = 0; y < Ny_; ++y) {
				for (size_t x = 0; x < Nx_; ++x) {
					nodes_[idx(x, y)] = get_struct_coord(x, y, 0);
				}
			}
		}
		
		// 3D constrcutor:
		meshStructured(size_t nx, size_t ny, size_t nz, double dx) : Nx_(nx), Ny_(ny), Nz_(nz), dx_(dx) {
			static_assert(dim == 3, "This constructor is strictly for 3D grids.");
			size_t = tot_nodes = Nx_ * Ny_ * Nz_;
			nodes_.resize(tot_nodes);
			flags_.resize(tot_nodes, 0);

			for (size_t z = 0; z < Nz_; ++z) {
				for (size_t y = 0; y < Ny_; ++y) {
					for (size_t x = 0; x < Nx_; ++x) {
						nodes_[idx(x, y, z)] = get_struct_coord(x, y, z);
					}
				}
			}
		}
		
		// We override the virtual functions defined in geo/mesh.hpp
		size_t num_nodes() const override { 
			return nodes_.size(); 
		}
		Tensor<double, 1, dim> get_coord(size_t i) const override { 
			return nodes_[i]; 
		}
		bool has_bound(const std::string& name) const override {
			return bound_names_to_ids_.find(name) != bound_names_to_ids_.end();
		}
		const std::vector<size_t>& get_bound(const std::string& name) const override {
			auto it = bound_ids_.find(name);
			if (it == bound_ids_.end()) {
				throw std::runtime_error("Boundary index buffer not found: " + name) 
			}
			return it->second;
		}

		// Access the private class data:
		size_t Nx() const { return Nx_; }
		size_t Ny() const { return Ny_; }
		size_t Nx() const { return Nz_; }
		double dx() const { return dx_; }

		size_t idx(size_t x, size_t y, size_t z = 0) const {
			return (z * Ny_ + y) *Nx_ + x;
		}

		uint32_t get_flag(size_t x, size_t y, size_t z = 0) const {
			return flags_[idx(x, y, z)];
		}

		void set_flag(size_t x, size_t y, size_t z, uint32_t flag) {
			flags_[idx(x, y, z)] = flag;
		}

		void set_flag(size_t x, size_t y, uint32_t flag) {
			static_assert(dim == 2, "Must specify a z coordinate for 3D grid flagging.");
			flags_[idx(x, y, 0)] = flag;
		}
};
