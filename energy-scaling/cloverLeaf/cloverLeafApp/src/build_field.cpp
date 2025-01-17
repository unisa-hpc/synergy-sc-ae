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


// @brief  Allocates the data for each mesh chunk
// @author Wayne Gaudin
// @details The data fields for the mesh chunk are allocated based on the mesh
// size.

#include "build_field.h"
#include "sycl_utils.hpp"

// Allocate Kokkos Views for the data arrays
void build_field(global_variables &globals) {

	for (int tile = 0; tile < globals.config.tiles_per_chunk; ++tile) {

		tile_type &t = globals.chunk.tiles[tile];

		const size_t xrange = (t.info.t_xmax + 2) - (t.info.t_xmin - 2) + 1;
		const size_t yrange = (t.info.t_ymax + 2) - (t.info.t_ymin - 2) + 1;


		field_type &field = t.field;


		clover::execute(1107, 1155,globals.queue, "build_field_1",[&](cl::sycl::handler &h) {
			auto work_array1 = field.work_array1.access<W>(h);
			auto work_array2 = field.work_array2.access<W>(h);
			auto work_array3 = field.work_array3.access<W>(h);
			auto work_array4 = field.work_array4.access<W>(h);
			auto work_array5 = field.work_array5.access<W>(h);
			auto work_array6 = field.work_array6.access<W>(h);
			auto work_array7 = field.work_array7.access<W>(h);
			auto xvel0 = field.xvel0.access<W>(h);
			auto xvel1 = field.xvel1.access<W>(h);
			auto yvel0 = field.yvel0.access<W>(h);
			auto yvel1 = field.yvel1.access<W>(h);
			// Nested loop over (t_ymin-2:t_ymax+3) and (t_xmin-2:t_xmax+3) inclusive
			clover::par_ranged<class APPEND_LN(build_field)>(h, {0u, 0u, xrange + 1, yrange + 1}, [=](
					id<2> id) {
				work_array1[id] = 0.0;
				work_array2[id] = 0.0;
				work_array3[id] = 0.0;
				work_array4[id] = 0.0;
				work_array5[id] = 0.0;
				work_array6[id] = 0.0;
				work_array7[id] = 0.0;

				xvel0[id] = 0.0;
				xvel1[id] = 0.0;
				yvel0[id] = 0.0;
				yvel1[id] = 0.0;
			});
		});

		clover::execute(1107, 1162,globals.queue,"build_field_1", [&](cl::sycl::handler &h) {
			auto density0 = field.density0.access<W>(h);
			auto density1 = field.density1.access<W>(h);
			auto energy0 = field.energy0.access<W>(h);
			auto energy1 = field.energy1.access<W>(h);
			auto pressure = field.pressure.access<W>(h);
			auto viscosity = field.viscosity.access<W>(h);
			auto soundspeed = field.soundspeed.access<W>(h);
			auto volume = field.volume.access<W>(h);
			// Nested loop over (t_ymin-2:t_ymax+2) and (t_xmin-2:t_xmax+2) inclusive
			clover::par_ranged<class APPEND_LN(build_field)>(h, {0u, 0u, xrange, yrange}, [=](id<2> idx) {
				density0[idx] = 0.0;
				density1[idx] = 0.0;
				energy0[idx] = 0.0;
				energy1[idx] = 0.0;
				pressure[idx] = 0.0;
				viscosity[idx] = 0.0;
				soundspeed[idx] = 0.0;
				volume[idx] = 0.0;
			});
		});

		clover::execute(1107, 1162,globals.queue, "build_field_3", [&](cl::sycl::handler &h) {
			auto vol_flux_x = field.vol_flux_x.access<W>(h);
			auto mass_flux_x = field.mass_flux_x.access<W>(h);
			auto xarea = field.xarea.access<W>(h);
			// Nested loop over (t_ymin-2:t_ymax+2) and (t_xmin-2:t_xmax+3) inclusive
			clover::par_ranged<class APPEND_LN(build_field)>(h, {0u, 0u, xrange, yrange}, [=](id<2> idx) {
				vol_flux_x[idx] = 0.0;
				mass_flux_x[idx] = 0.0;
				xarea[idx] = 0.0;
			});
		});

		clover::execute(1107, 1162,globals.queue, "build_field_4", [&](cl::sycl::handler &h) {
			auto vol_flux_y = field.vol_flux_y.access<W>(h);
			auto mass_flux_y = field.mass_flux_y.access<W>(h);
			auto yarea = field.yarea.access<W>(h);
			// Nested loop over (t_ymin-2:t_ymax+3) and (t_xmin-2:t_xmax+2) inclusive
			clover::par_ranged<class APPEND_LN(build_field)>(h, {0u, 0u, xrange, yrange + 1}, [=](id<2> idx) {
				vol_flux_y[idx] = 0.0;
				mass_flux_y[idx] = 0.0;
				yarea[idx] = 0.0;
			});
		});

		clover::execute(1107, 1117,globals.queue, "build_field_5", [&](cl::sycl::handler &h) {
			auto cellx = field.cellx.access<W>(h);
			auto celldx = field.celldx.access<W>(h);
			// (t_xmin-2:t_xmax+2) inclusive
			clover::par_ranged<class APPEND_LN(build_field)>(h, {0u, xrange}, [=](id<1> id) {
				cellx[id] = 0.0;
				celldx[id] = 0.0;
			});
		});

		clover::execute(1107, 1162,globals.queue, "build_field_6", [&](cl::sycl::handler &h) {
			auto celly = field.celly.access<W>(h);
			auto celldy = field.celldy.access<W>(h);
			// (t_ymin-2:t_ymax+2) inclusive
			clover::par_ranged<class APPEND_LN(build_field)>(h, {0u, yrange}, [=](id<1> id) {
				celly[id] = 0.0;
				celldy[id] = 0.0;
			});
		});

		clover::execute(1107, 1162,globals.queue,"build_field_7", [&](cl::sycl::handler &h) {
			auto vertexx = field.vertexx.access<W>(h);
			auto vertexdx = field.vertexdx.access<W>(h);
			// (t_xmin-2:t_xmax+3) inclusive
			clover::par_ranged<class APPEND_LN(build_field)>(h, {0u, xrange + 1}, [=](id<1> id) {
				vertexx[id] = 0.0;
				vertexdx[id] = 0.0;
			});
		});

		clover::execute(1107, 1162,globals.queue,"build_field_8", [&](cl::sycl::handler &h) {
			auto vertexy = field.vertexy.access<W>(h);
			auto vertexdy = field.vertexdy.access<W>(h);
			// (t_ymin-2:t_ymax+3) inclusive
			clover::par_ranged<class APPEND_LN(build_field)>(h, {0u, yrange + 1}, [=](id<1> id) {
				vertexy[id] = 0.0;
				vertexdy[id] = 0.0;
			});
		});


	}

}

