/*
 Crown Copyright 2012 AWE.

 This file is part of CloverLeaf.

 CloverLeaf is free software: you can redistribute it and/or modify it under
 the terms of the GNU General Public License as published by the
 Free Software Foundation, either version 3 of the License, or (at your option)
 any later version.

 CloverLeaf is distributed in the hope that it will be useful, but
 WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
 details.

 You should have received a copy of the GNU General Public License along with
 CloverLeaf. If not, see http://www.gnu.org/licenses/.
 */

#ifndef CLOVERLEAF_SYCL_SYCL_UTILS_HPP
#define CLOVERLEAF_SYCL_SYCL_UTILS_HPP

#include <synergy.hpp>
#include <iostream>
#include <utility>
#include <mpi.h>


//#define SYCL_DEBUG // enable for debugging SYCL related things, also syncs kernel calls
// #define SYNC_KERNELS // enable for fully synchronous (e.g queue.wait_and_throw()) kernel calls
//#define SYCL_FLIP_2D // enable for flipped id<2> indices from SYCL default

// this namespace houses all SYCL related abstractions
namespace clover {

	// abstracts away cl::sycl::accessor
	template<typename T,
			int N,
			cl::sycl::access::mode mode>
	struct Accessor {
		typedef cl::sycl::accessor<T, N, mode, cl::sycl::access::target::global_buffer> Type;
		typedef cl::sycl::accessor<T, N, mode, cl::sycl::access::target::host_buffer> HostType;

		inline static Type from(cl::sycl::buffer<T, N> &b, cl::sycl::handler &cgh) {
			return b.template get_access<mode, cl::sycl::access::target::global_buffer>(cgh);
		}

		inline static HostType access_host(cl::sycl::buffer<T, N> &b) {
			return b.template get_access<mode>();
		}

	};

	// abstracts away cl::sycl::buffer
	template<typename T, int N>
	struct Buffer {

		cl::sycl::buffer<T, N> buffer;

		// delegates to the corresponding buffer constructor
		explicit Buffer(cl::sycl::range<N> range) : buffer(range) {}

		// delegates to the corresponding buffer constructor
		template<typename Iterator>
		explicit Buffer(Iterator begin, Iterator end) : buffer(begin, end) {}

		// delegates to accessor.get_access<mode>(handler)
		template<cl::sycl::access::mode mode>
		inline typename Accessor<T, N, mode>::Type
		access(cl::sycl::handler &cgh) { return Accessor<T, N, mode>::from(buffer, cgh); }


		// delegates to accessor.get_access<mode>()
		// **for host buffers only**
		template<cl::sycl::access::mode mode>
		inline typename Accessor<T, N, mode>::HostType
		access() { return Accessor<T, N, mode>::access_host(buffer); }

	};

	struct Range1d {
		const size_t from, to;
		const size_t size;
		template<typename A, typename B>
		Range1d(A from, B to) : from(from), to(to), size(to - from) {
			assert(from < to);
			assert(size != 0);
		}
		friend std::ostream &operator<<(std::ostream &os, const Range1d &d) {
			os << "Range1d{"
			   << " X[" << d.from << "->" << d.to << " (" << d.size << ")]"
			   << "}";
			return os;
		}
	};

	struct Range2d {
		const size_t fromX, toX;
		const size_t fromY, toY;
		const size_t sizeX, sizeY;
		template<typename A, typename B, typename C, typename D>
		Range2d(A fromX, B fromY, C toX, D toY) :
				fromX(fromX), toX(toX), fromY(fromY), toY(toY),
				sizeX(toX - fromX), sizeY(toY - fromY) {
			assert(fromX < toX);
			assert(fromY < toY);
			assert(sizeX != 0);
			assert(sizeY != 0);
		}
		friend std::ostream &operator<<(std::ostream &os, const Range2d &d) {
			os << "Range2d{"
			   << " X[" << d.fromX << "->" << d.toX << " (" << d.sizeX << ")]"
			   << " Y[" << d.fromY << "->" << d.toY << " (" << d.sizeY << ")]"
			   << "}";
			return os;
		}
	};

	// safely offset an id<2> by j and k
	static inline cl::sycl::id<2> offset(const cl::sycl::id<2> idx, const int j, const int k) {
		int jj = static_cast<int>(idx[0]) + j;
		int kk = static_cast<int>(idx[1]) + k;
#ifdef SYCL_DEBUG
		// XXX only use on runtime that provides assertions, eg: CPU
		assert(jj >= 0);
		assert(kk >= 0);
#endif
		return cl::sycl::id<2>(jj, kk);
	}


	// delegates to parallel_for, handles flipping if enabled
	template<typename nameT, class functorT>
	static inline void par_ranged(cl::sycl::handler &cgh, const Range1d &range, functorT functor) {
		cgh.parallel_for<nameT>(
				cl::sycl::range<1>(range.size),
				cl::sycl::id<1>(range.from),
				functor);
	}

	// delegates to parallel_for, handles flipping if enabled
	template<typename nameT, class functorT>
	static inline void par_ranged(cl::sycl::handler &cgh, const Range2d &range, functorT functor) {
#ifdef SYCL_FLIP_2D
		cgh.parallel_for<nameT>(
				cl::sycl::range<2>(range.sizeY, range.sizeX),
				cl::sycl::id<2>(range.fromY, range.fromX),
				[=](cl::sycl::id<2> idx) {
					functor(cl::sycl::id<2>(idx[1], idx[0]));
				});
#else
		cgh.parallel_for<nameT>(
				cl::sycl::range<2>(range.sizeX, range.sizeY),
				cl::sycl::id<2>(range.fromX, range.fromY),
				functor);
#endif
	}

	// delegates to queue.submit(cgf), handles blocking submission if enable
	template<typename T>
	static void execute(synergy::frequency memory_freq, synergy::frequency core_freq,  synergy::queue&queue, std::string kernel_name, T cgf) {
		try {
			sycl::event e = queue.submit(memory_freq, core_freq, cgf);
			// Take the mpi process rank
			int world_rank;
    		MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
			// Fine grained energy consumption
  			// std::cout << "kernel_name: " << kernel_name << ", rank: "<< world_rank << ", energy_consumption [J]: " << queue.kernel_energy_consumption(e) << "\n";
			
#if defined(SYCL_DEBUG) || defined(SYNC_KERNELS)
			queue.wait_and_throw();
#endif
		} catch (cl::sycl::device_error &e) {
			std::cerr << "[SYCL] Device error: : `" << e.what() << "`" << std::endl;
			throw e;
		} catch (cl::sycl::exception &e) {
			std::cerr << "[SYCL] Exception : `" << e.what() << "`" << std::endl;
			throw e;
		}
	}
}


#endif //CLOVERLEAF_SYCL_SYCL_UTILS_HPP
