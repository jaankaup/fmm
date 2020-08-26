use cgmath::{prelude::*,Vector3, Vector4};
use std::collections::HashMap;
use crate::bvh::{BBox, Triangle}; 
//use crate::misc::{Vertex_vvvc};
use crate::misc::*;

const SQ3: f32 = 1.73205080757;

#[derive(Clone, Copy)]
pub enum Cell {
    Known(f32, Vector3<f32>), // (distance, normal_vector)
    Band(f32, Vector3<f32>), // normal_vector
    Far(),
}

//   Initial state
//  
//   I :: Surface ( b(x) == 0 )
//   + :: Ghost point (boundary condition b(x))
//   F :: Far
//   K :: Known
//   base_position :: (0.0, 0.0, 0.0)
//  
//                                    w(x) < 0                                           w(x) > 0
//  
//    5.5  +---------+---------+---------+---------+---------+---------I---------+---------+---------+---------+--
//         |         |         |         |         |         |       II|         |         |         |         |       
//         |         |         |         |         |         |      I  |         |         |         |         |       
//    5.0  |    F    |    F    |    F    |    F    |    F    |   F I   |    K    |    F    |    F    |    F    |       
//         |         |         |         |         |         |   II    |         |         |         |         |       
//         |         |         |         |         |         | II      |         |         |         |         |       
//    4.5  +---------+---------+---------+---------+---------+I--------+---------+---------+---------+---------+--
//         |         |         |         |         |         I         |         |         |         |         |       
//         |         |         |         |         |        I|         |         |         |         |         |       
//    4.0  |    F    |    F    |    F    |    F    |    F  I |    K    |    F    |    F    |    F    |    F    |       
//         |         |         |         |         |       I |         |         |         |         |         |       
//         |         |         |         |         |        I|         |         |         |         |         |       
//    3.5  +---------+---------+---------+---------+---------I---------+---------+---------+---------+---------+--
//         |         |         |         |         |         I         |         |         |         |         |       
//         |         |         |         |         |         |I        |         |         |         |         |       
//    3.0  |    F    |    F    |    F    |    F    |    F    | I  K    |    F    |    F    |    F    |    F    |       
//         |         |         |         |         |         |  I      |         |         |         |         |       
//         |         |         |         |         |         |  I      |         |         |         |         |       
//    2.5  +---------+---------+---------+---------+---------+--I------+---------+---------+---------+---------+--
//         |         |         |         |         |         | I       |         |         |         |         |       
//         |         |         |         |         |         |I        |         |         |         |         |       
//    2.0  |    F    |    F    |    F    |    F    |    F    I    K    |    F    |    F    |    F    |    F    |       
//         |         |         |         |         |         I         |         |         |         |         |       
//         |         |         |         |         |        I|         |         |         |         |         |       
//    1.5  +---------+---------+---------+---------+-------I-+---------+---------+---------+---------+---------+--
//         |         |         |         |         |      I  |         |         |         |         |         |       
//         |         |         |         |         |     I   |         |         |         |         |         |       
//    1.0  |    F    |    F    |    F    |    F    |  F I    |    K    |    F    |    F    |    F    |    F    |       
//         |         |         |         |         |    I    |         |         |         |         |         |       
//         |         |         |         |         |   I     |         |         |         |         |         |       
//    0.5  +---------+---------+---------+---------+---I-----+---------+---------+---------+---------+---------+--
//  
//        0.5  1.0  1.5  2.0  2.5  3.0  3.5  4.0  4.5  5.0  5.5  6.0  6.5  7.0  7.5  8.0  8.5  9.0  9.5  10.0  10.5
//  

/// Data structure for holding the information about the fmm state.
pub struct FMM_Domain {
    dimension: (std::ops::Range<u32>, std::ops::Range<u32>, std::ops::Range<u32>),
    base_position: Vector3<f32>,
    grid_length: f32,
    cells: Vec<Cell>,
    boundary_points: Option<Vec<f32>>,
}

impl FMM_Domain {

    pub fn new(dimension: (u32, u32, u32), base_position: Vector3<f32>, grid_length: f32, boundary: bool) -> Self {
        assert!(grid_length > 0.0, "FMM_Domain.new: grid_length {} > 0.0", grid_length);
        assert!(grid_length > 0.0, "FMM_Domain.new: grid_length {} > 0.0", grid_length);

        // Initialize cell points with far values.
        let cells: Vec<Cell> = vec![Cell::Far() ; (dimension.0 * dimension.1 * dimension.2) as usize];

        // Initialize boundary points.
        let mut boundary_points = None;
        if boundary {
            boundary_points = Some(vec![0.0 ; ((dimension.0 + 1) * (dimension.1 + 1) * (dimension.2 + 1)) as usize]); 
        }

        Self {
            dimension: (0..dimension.0, 0..dimension.1, 0..dimension.2),
            base_position: base_position ,
            grid_length: grid_length,
            cells: cells,
            boundary_points: boundary_points,
        }
    }

    /// Set cell value at (i, j, k) position.
    pub fn set_cell(&mut self, i: u32, j: u32, k: u32, cell: Cell) {
        let index = self.get_array_index(i, j, k).unwrap() as usize; 
        self.cells[index] = cell;
    }

    /// Get cell value at (i, j, k) position.
    pub fn get_cell(&self, i: u32, j: u32, k: u32) -> Cell {
        let index = self.get_array_index(i, j, k).unwrap() as usize;      
        self.cells[index]
    }

    // /// Get the cell position in "world" space.
    // pub fn get_cell_w_pos(&self, i: u32, j: u32, k: u32) -> Vector3<f32> {

    //     // For debugging purpose.
    //     self.check_cell_index(i, j, k).unwrap();

    //     let result = self.base_position + (Vector3::<f32>::new(i as f32, j as f32, k as f32)) * self.grid_length;
    //     result
    // }

    pub fn cell_to_aabb(&self, i:u32, j:u32, k:u32) -> Option<BBox> {
        None   
    }

    /// Map given world position to nearest cell array index coordinates. Returns None if given point is outside
    /// computational domain. SHOULD WORK NOW
    pub fn xyz_to_ijk(&self, world_pos: &Vector3<f32>, snap_to_farest: bool) -> Option<(u32, u32, u32)> {

        let mut i: u32 = 0;
        let mut j: u32 = 0;
        let mut k: u32 = 0;
        // println!("nuapurit!");
        // let neighbors = self.xyz_to_all_nearest_ijk(&world_pos);
        // let mut closest: Option<(u32, u32, u32)> = None; 

        // match neighbors {
        //     None => { return None; }
        //     Some((v_ijk)) => {
        //         let mut distance = 1000000000.0;
        //         for (r, q, s) in &v_ijk {
        //             let t = self.ijk_to_xyz(*r, *q, *s).unwrap(); 
        //             let d = world_pos.distance(t);
        //             if d < distance { closest = Some((*r, *q, *s)); } 
        //             distance = d;
        //         }
        //     }
        // }

        //closest


        if snap_to_farest {
            i = ((world_pos.x + self.grid_length * 0.5 - self.base_position.x) / self.grid_length).ceil() as u32;
            j = ((world_pos.y + self.grid_length * 0.5 - self.base_position.y) / self.grid_length).ceil() as u32;
            k = ((world_pos.z + self.grid_length * 0.5 - self.base_position.z) / self.grid_length).ceil() as u32;
        }
        else {
            i = ((world_pos.x - self.base_position.x + 0.5) / self.grid_length).floor() as u32;
            j = ((world_pos.y - self.base_position.y + 0.5) / self.grid_length).floor() as u32;
            k = ((world_pos.z - self.base_position.z + 0.5) / self.grid_length).floor() as u32;
        }
        //println!("i :: {}, j :: {}, k :: {}", i, j, k);

        match self.check_cell_index(i, j, k) {
            Ok(()) => { 
                return Some((i, j, k));
            }
            Err(_) => {
                //println!("{}", s);
                return None;
            }
        }
    }

    /// Map cell array index coordinates to world position. Returns None if i, j, k is outside
    /// index space. Should work now. 
    pub fn ijk_to_xyz(&self, i: u32, j: u32, k: u32) -> Option<Vector3<f32>> {

        match self.check_cell_index(i, j, k) {
            Ok(()) => { 
                let x = i as f32 * self.grid_length + self.base_position.x;
                let y = j as f32 * self.grid_length + self.base_position.y;
                let z = k as f32 * self.grid_length + self.base_position.z;
                return Some(Vector3::<f32>::new(x, y, z));
            }
            _ => {
                println!("({}, {}, {}) are outside" , i, j, k);
                return None;
            }
        }
    }

    /// Get the nearest cell point from given world coordinates (xyz). Returns None if given world
    /// position is outside the computational domain. Should work now.
    pub fn xyz_to_nearest_cell_xyz(&self, world_pos: &Vector3<f32>) -> Option<Vector3<f32>> {
         
        if let Some((i,j,k)) = self.xyz_to_ijk(&world_pos, false) {
            if let Some(v) = self.ijk_to_xyz(i, j, k) {
                return Some(v);
            }
            else { // is this necessery? TODO: remove if this is never reached.
                return None;
            }
        }
        None
    }

    // :noremap <F12> :wall \| !./native_compile.sh && ./runNative.sh<CR>

    /// Get all nearest cell array indices which satisfies the condition: dist(world_pos,
    /// cell_point) <= sqrt(3) * cube_length / 2. Return None if the given world position is too far
    /// away from any cell point.
    pub fn xyz_to_all_nearest_ijk(&self, world_pos: &Vector3<f32>) -> Option<Vec<(u32, u32, u32)>> {

        // If the given position is inside the computational domain.
        if let Some((i, j, k)) = self.xyz_to_ijk(&world_pos, false) {

            // Seach for all 26 neighbors. Only test for those indices that lies inside
            // computational domain. 
           
            let mut cell_indices: Vec<(u32, u32, u32)> = Vec::new();
            println!("nearest:");

            let a_range = std::ops::Range::<i32> { start: i as i32 - 1, end: i as i32 + 2 };
            let b_range = std::ops::Range::<i32> { start: j as i32 - 1, end: j as i32 + 2 };
            let c_range = std::ops::Range::<i32> { start: k as i32 - 1, end: k as i32 + 2 };

            for a in a_range {
            for b in b_range.clone() {
            for c in c_range.clone() {

                // Skip negative indices.
                if a < 0 || b < 0 || c < 0 { continue; }

                if let Ok(()) = self.check_cell_index(a as u32, b as u32, c as u32) {
                    let cell_point = self.ijk_to_xyz(a as u32, b as u32, c as u32).unwrap();
                    if world_pos.distance(cell_point) <= SQ3 * self.grid_length * 0.5 { 
                        println!("cell_point == ({}, {}, {})", cell_point.x, cell_point.y, cell_point.z);
                        cell_indices.push((a as u32, b as u32, c as u32));
                        println!("({}, {}, {}) added", a, b, c);
                    }
                }
            }}};

            Some(cell_indices)
        }
        else {
            None
        }
    }

    /// Return all world coordinates which are inside a cell centered ball r less than the length
    /// from cell center to the nearest cell cube corner. Returns None if given world coordinate
    /// is too far away from any cell center. Seems to work.
    pub fn xyz_to_all_nearest_cells_xyz(&self, world_pos: &Vector3<f32>) -> Option<Vec<Vector3<f32>>> {

        let mut result: Vec<Vector3<f32>> = Vec::new();

        // Find all nearest cell i, j, k with distance condition. 
        if let Some(v_ijk) = self.xyz_to_all_nearest_ijk(&world_pos) {
            for i in v_ijk.iter() {
                result.push(self.ijk_to_xyz(i.0, i.1, i.2).unwrap());
                let temp = self.ijk_to_xyz(i.0, i.1, i.2).unwrap();
            }
        }

        // At least one found.
        if result.len() > 0 { Some(result) }

        // The point is too far from computational context. No cell found.
        else { None }
    }

    /// Get the nearest cell point based on given w_pos coordinate.
    pub fn get_neighbor_grid_points(&self, w_pos: &Vector3<f32>) -> (Vec<Vertex_vvvc_nnnn>, Vec<Vertex_vvvc>) {

        // Cubes.
        let mut result: Vec<Vertex_vvvc_nnnn> = Vec::new();

        // Lines.
        let mut result2: Vec<Vertex_vvvc> = Vec::new();

        // let coords = self.xyz_to_all_nearest_cells_xyz(&w_pos);
        // match coords {
        //     Some(v_cell_positions) => { 
        //         let cell_color = encode_rgba_u32(255, 50, 50, 255); 
        //         for cell_pos in v_cell_positions.iter() {
        //             let cube = create_cube_triangles(cell_pos - Vector3::<f32>::new(0.004, 0.004, 0.004), 0.008, cell_color);
        //             result.extend(cube);
        //             result2.push(Vertex_vvvc {
        //                 position: [cell_pos.x, cell_pos.y, cell_pos.z],
        //                 color: encode_rgba_u32(255, 50, 50, 255),
        //             });
        //             result2.push(Vertex_vvvc {
        //                 position: [w_pos.x, w_pos.y, w_pos.z], 
        //                 color: encode_rgba_u32(100, 50, 255, 255),
        //             });
        //         }
        //     },
        //    _ => { println!("nuapira ei löövy"); }
        // }

        // Only one cell center.
        let coord = self.xyz_to_nearest_cell_xyz(&w_pos);
        match coord {
            Some(cell_pos) => { 

                let cell_color = encode_rgba_u32(255, 50, 50, 255); 
                let cube = create_cube_triangles(cell_pos, 0.006, cell_color);
                result.extend(cube);
                result2.push(Vertex_vvvc {
                    position: [cell_pos.x, cell_pos.y, cell_pos.z],
                    color: encode_rgba_u32(255, 50, 50, 255),
                });
                result2.push(Vertex_vvvc {
                    position: [w_pos.x, w_pos.y, w_pos.z], 
                    color: encode_rgba_u32(100, 50, 255, 255),
                });
            },
           _ => { println!("nuapira ei löövy"); }
        }
        (result, result2)
    }

    /// Create vvvv data from cell data. Cell color is encoded to in vvvc form, where vvv part is
    /// the coordinates xyz and c is encoded rgba information. 
    pub fn cells_to_vvvc_nnnn(&self) -> Vec<Vertex_vvvc_nnnn> {

        let mut result: Vec<Vertex_vvvc_nnnn> = Vec::new();
        for i in 0..self.dimension.0.end { 
        for j in 0..self.dimension.1.end { 
        for k in 0..self.dimension.2.end { 
            let cell = self.cells[self.get_array_index(i, j, k).unwrap() as usize]; 
            let cell_color = match cell {
                Cell::Known(_,_) => { encode_rgba_u32(0, 255, 0, 255) } 
                Cell::Band(_,_)  => { encode_rgba_u32(255, 255, 0, 255) }  
                Cell::Far()    => { encode_rgba_u32(255, 0, 255, 0) }  
            };
            let mut cell_ws_pos = self.ijk_to_xyz(i, j, k).unwrap();
            let cube = create_cube_triangles(cell_ws_pos, 0.004, cell_color);

            result.extend(cube);
        }}};
        result
    }

    /// Create vvvc cell cube data. 
    pub fn boundary_lines(&self) -> Vec<Vertex_vvvc> {

        let mut result: Vec<Vertex_vvvc> = Vec::new();

        let offset = self.grid_length * 0.5;
        let grid_offset = self.grid_length;
        for i in 0..self.dimension.0.end { 
        for j in 0..self.dimension.1.end { 
        for k in 0..self.dimension.2.end { 

            // Create cube.
            let cell_ws_pos = self.ijk_to_xyz(i, j, k).unwrap();

            let p0 = Vector3::<f32>::new(cell_ws_pos.x - offset, cell_ws_pos.y - offset, cell_ws_pos.z - offset);
            let p1 = p0 + Vector3::<f32>::new(0.0,         grid_offset, 0.0);
            let p2 = p0 + Vector3::<f32>::new(grid_offset, grid_offset, 0.0);
            let p3 = p0 + Vector3::<f32>::new(grid_offset, 0.0        , 0.0);

            let p4 = p0 + Vector3::<f32>::new(0.0,         0.0,         grid_offset);
            let p5 = p0 + Vector3::<f32>::new(0.0,         grid_offset, grid_offset);
            let p6 = p0 + Vector3::<f32>::new(grid_offset, grid_offset, grid_offset);
            let p7 = p0 + Vector3::<f32>::new(grid_offset, 0.0,         grid_offset);

            let r = 10;
            let g = 10;
            let b = 10;
            let a = 255;

            result.push(Vertex_vvvc { position: [p0.x, p0.y, p0.z], color: encode_rgba_u32(r, g, b, a), });
            result.push(Vertex_vvvc { position: [p1.x, p1.y, p1.z], color: encode_rgba_u32(r, g, b, a), });
            result.push(Vertex_vvvc { position: [p1.x, p1.y, p1.z], color: encode_rgba_u32(r, g, b, a), });
            result.push(Vertex_vvvc { position: [p2.x, p2.y, p2.z], color: encode_rgba_u32(r, g, b, a), });
            result.push(Vertex_vvvc { position: [p2.x, p2.y, p2.z], color: encode_rgba_u32(r, g, b, a), });
            result.push(Vertex_vvvc { position: [p3.x, p3.y, p3.z], color: encode_rgba_u32(r, g, b, a), });
            result.push(Vertex_vvvc { position: [p3.x, p3.y, p3.z], color: encode_rgba_u32(r, g, b, a), });
            result.push(Vertex_vvvc { position: [p0.x, p0.y, p0.z], color: encode_rgba_u32(r, g, b, a), });
            result.push(Vertex_vvvc { position: [p0.x, p0.y, p0.z], color: encode_rgba_u32(r, g, b, a), });
            result.push(Vertex_vvvc { position: [p4.x, p4.y, p4.z], color: encode_rgba_u32(r, g, b, a), });
            result.push(Vertex_vvvc { position: [p4.x, p4.y, p4.z], color: encode_rgba_u32(r, g, b, a), });
            result.push(Vertex_vvvc { position: [p5.x, p5.y, p5.z], color: encode_rgba_u32(r, g, b, a), });
            result.push(Vertex_vvvc { position: [p5.x, p5.y, p5.z], color: encode_rgba_u32(r, g, b, a), });
            result.push(Vertex_vvvc { position: [p6.x, p6.y, p6.z], color: encode_rgba_u32(r, g, b, a), });
            result.push(Vertex_vvvc { position: [p6.x, p6.y, p6.z], color: encode_rgba_u32(r, g, b, a), });
            result.push(Vertex_vvvc { position: [p7.x, p7.y, p7.z], color: encode_rgba_u32(r, g, b, a), });
            result.push(Vertex_vvvc { position: [p7.x, p7.y, p7.z], color: encode_rgba_u32(r, g, b, a), });
            result.push(Vertex_vvvc { position: [p4.x, p4.y, p4.z], color: encode_rgba_u32(r, g, b, a), });
            result.push(Vertex_vvvc { position: [p1.x, p1.y, p1.z], color: encode_rgba_u32(r, g, b, a), });
            result.push(Vertex_vvvc { position: [p5.x, p5.y, p5.z], color: encode_rgba_u32(r, g, b, a), });
            result.push(Vertex_vvvc { position: [p2.x, p2.y, p2.z], color: encode_rgba_u32(r, g, b, a), });
            result.push(Vertex_vvvc { position: [p6.x, p6.y, p6.z], color: encode_rgba_u32(r, g, b, a), });
            result.push(Vertex_vvvc { position: [p3.x, p3.y, p3.z], color: encode_rgba_u32(r, g, b, a), });
            result.push(Vertex_vvvc { position: [p7.x, p7.y, p7.z], color: encode_rgba_u32(r, g, b, a), });

        }}};
        result
    }

    /// A helper function for checking the bounds of given cell index.
    /// If assert is true, program will panic if conditions are not met.
    /// returns true is index is valid, false otherwise.
    pub fn check_cell_index(&self, i: u32, j: u32, k: u32) -> Result<(), String> {

        let x_ok = self.dimension.0.contains(&i);
        let y_ok = self.dimension.1.contains(&j);
        let z_ok = self.dimension.2.contains(&k);

        if !x_ok {
            return Err(format!("FMM_Domain::check_cell_index(i: {}, j: {}, k: {}) :: i not in range 0..{}", i, j ,k, self.dimension.0.end).to_string());
        }
        if !y_ok {
            return Err(format!("FMM_Domain::check_cell_index(i: {}, j: {}, k: {}) :: j not in range 0..{}", i, j ,k, self.dimension.0.end).to_string());
        }
        if !z_ok {
            return Err(format!("FMM_Domain::check_cell_index(i: {}, j: {}, k: {}) :: k not in range 0..{}", i, j ,k, self.dimension.0.end).to_string());
        }
        
        Ok(())
    }

    /// Get array index for cell coordinates i,j,k. 
    pub fn get_array_index(&self, i: u32, j: u32, k: u32) -> Result<u32, String> {

        // Check if index is inside domain bounds.
        match self.check_cell_index(i, j, k) {
            Err(s) => {
                let mut error = format!("FMM_Domain::get_array_index({}, {}, {})\n", i, j, k).to_string();
                error.push_str(&s);
                return Err(error);
                }
            _ => { }
        }

        let x_range = self.dimension.0.end; 
        let y_range = self.dimension.1.end; 
        let index = i + j * x_range + k * x_range * y_range;

        Ok(index)
    }

    // Unused.
    pub fn get_domain_bounding_box(&self) -> BBox {
        let bbox = BBox::create_from_line(
            &self.base_position,
            &(self.base_position
                + Vector3::<f32>::new(
                    self.dimension.0.end as f32 * self.grid_length,
                    self.dimension.1.end as f32 * self.grid_length,
                    self.dimension.2.end as f32 * self.grid_length)
            ),
        );
        // println!("get_cells_bounding_box");
        // println!("BBox({}, {}, {}, {}, {}, {}", bbox.min.x, bbox.min.y, bbox.min.z, bbox.max.x, bbox.max.y, bbox.max.z);

        bbox
    }

// Update initial state with given triangle (a,b,c). If some of triangle vertices are outside
    // the computational domain, the triangle is rejected.
    pub fn add_triangles(&mut self, triangles: &Vec<Triangle>) -> (Vec<Vertex_vvvc_nnnn>, Vec<Vertex_vvvc>) {

        let domain_aabb = self.get_domain_bounding_box();

        // Cubes.
        let mut result: Vec<Vertex_vvvc_nnnn> = Vec::new();

        // Lines.
        let mut result2: Vec<Vertex_vvvc> = Vec::new();


        for tr in triangles {

            // Triangle is inside computational domain.
            if domain_aabb.includes_point(&tr.a, true) && domain_aabb.includes_point(&tr.b, true) && domain_aabb.includes_point(&tr.c, true) {

                // println!("Triangle [({}, {}, {}), ({}, {}, {}), ({}, {}, {})] is inside domain.", 
                //     tr.a.x, tr.a.y, tr.a.z, tr.b.x, tr.b.y, tr.b.z, tr.c.x, tr.c.y, tr.c.z); 

                // TODO: implement from triangle type.
                let aabb = BBox::create_from_triangle(&tr.a, &tr.b, &tr.c);

                // expand to nearest cell grid points.
                //aabb.expand(

                // Get the cell index bounds for aabb.
                //let min_indices = self.xyz_to_ijk(&aabb.min, true);
                let min_indices = self.xyz_to_ijk(&aabb.min, true);

                let mut i_min = 0;
                let mut j_min = 0;
                let mut k_min = 0;
                let mut i_max = 0;
                let mut j_max = 0;
                let mut k_max = 0;

                // if there is something wrong with indeces, ignore this triangle.
                match min_indices {
                    Some((some_i_min, some_j_min, some_k_min)) => {
                        i_min = some_i_min - 1; // TODO: check if some_i_min is zero. 
                        j_min = some_j_min - 1; // TODO: check if some_i_min is zero.
                        k_min = some_k_min - 1; // TODO: check if some_i_min is zero.
                    },
                    None => { println!("TROUBLES!!!"); continue; },
                }

                //let max_indices = self.xyz_to_ijk(&aabb.max, true);
                let max_indices = self.xyz_to_ijk(&aabb.max, true);
                match max_indices {
                    Some((some_i_max, some_j_max, some_k_max)) => {
                        i_max = some_i_max;
                        j_max = some_j_max;
                        k_max = some_k_max;
                        
                    },
                    None => { println!("TROUBLES!!!");continue; },
                }

                ////let mut temp_cells: HashMap<u32, (Vec<Vertex_vvvc_nnnn>, Vec<Vertex_vvvc>)> = HashMap::new();

                // Update the closest cell points to the triangle.
                for q in i_min..i_max {
                for r in j_min..j_max {
                for s in k_min..k_max {
                    let cell_pos = self.ijk_to_xyz(q, r, s).unwrap();

                    let (distance, sign) = tr.distance_to_triangle(&cell_pos);
                    let signed_distance = match sign { true => distance, false => -distance }; 
                    let triangle_point = tr.closest_point_to_triangle(&cell_pos);
                    if distance > SQ3 * 0.5 * self.grid_length {
                        continue;
                    }
                    else {
                        //if distance.abs() > 2.0 { println!("distance == {}", distance); }
                        let mut cell = self.get_cell(q, r, s);
                        let mut new_found = false;
                        let normal = (cell_pos - triangle_point).normalize(); //ac.cross(ab).normalize();
                        match cell {
                            Cell::Known(val,_) => {
                                if distance < val.abs() { cell = Cell::Known(signed_distance, normal); self.set_cell(q, r, s, cell); new_found = true; } 
                            }
                            Cell::Band(val,_) => {
                                if distance < val.abs() { cell = Cell::Known(signed_distance, normal); self.set_cell(q, r, s, cell); new_found = true; } 
                            }
                            Cell::Far() => {
                                cell = Cell::Known(signed_distance, normal); self.set_cell(q, r, s, cell); new_found = true;
                            }
                        }
                    }
                }}};
            }
        }

        for i in 0..self.dimension.0.end {
        for j in 0..self.dimension.1.end {
        for k in 0..self.dimension.2.end {
            let item = self.get_cell(i, j, k);
            let grid_pos = self.ijk_to_xyz(i, j, k).unwrap(); 
            match item {
                Cell::Known(dist, dir) => {
                    let sign = if dist < 0.0 { false } else { true }; 
                    let grid_pos = self.ijk_to_xyz(i, j, k).unwrap(); 
                    let triangle_point = grid_pos - dir * dist.abs(); 
                    let cell_color = match sign { true => encode_rgba_u32(255, 50, 50, 255), false => encode_rgba_u32(50, 50, 200, 255) } ; 
                    let cube = create_cube_triangles(grid_pos , 0.006, cell_color);
                    result.extend(cube);
                    result2.push(Vertex_vvvc {
                        position: [grid_pos.x, grid_pos.y, grid_pos.z],
                        color: cell_color,// encode_rgba_u32(255, 50, 50, 255),
                    });
                    result2.push(Vertex_vvvc {
                        position: [triangle_point.x, triangle_point.y, triangle_point.z], 
                        color: cell_color, //encode_rgba_u32(100, 50, 255, 255),
                    });
                    
                }
                Cell::Band(dist, dir) => {
                    
                }
                Cell::Far() => {
                    // let cell_color = encode_rgba_u32(50, 155, 50, 255); 
                    // let cube = create_cube_triangles(grid_pos , 0.001, cell_color);
                    // result.extend(cube);
                }
            }
        }}};
        (result, result2)
    }

    pub fn create_grid_point_and_line_data(&self, show_known: bool, show_band: bool, show_far: bool, create_lines: bool) -> (Vec<Vertex_vvvc_nnnn>, Vec<Vertex_vvvc>) {

        // Cubes.
        let mut grid_cubes: Vec<Vertex_vvvc_nnnn> = Vec::new();

        // Lines.
        let mut lines: Vec<Vertex_vvvc> = Vec::new();

        for i in 0..self.dimension.0.end {
        for j in 0..self.dimension.1.end {
        for k in 0..self.dimension.2.end {
            let item = self.get_cell(i, j, k);
            let grid_pos = self.ijk_to_xyz(i, j, k).unwrap(); 
            match item {
                Cell::Known(dist, dir) => {
                    let sign = if dist < 0.0 { false } else { true }; 
                    let grid_pos = self.ijk_to_xyz(i, j, k).unwrap(); 
                    let triangle_point = grid_pos - dir * dist.abs(); 
                    let cell_color = match sign { true => encode_rgba_u32(255, 50, 50, 255), false => encode_rgba_u32(50, 50, 200, 255) } ; 
                    let cube = create_cube_triangles(grid_pos , 0.006, cell_color);
                    grid_cubes.extend(cube);
                    lines.push(Vertex_vvvc {
                        position: [grid_pos.x, grid_pos.y, grid_pos.z],
                        color: cell_color,// encode_rgba_u32(255, 50, 50, 255),
                    });
                    lines.push(Vertex_vvvc {
                        position: [triangle_point.x, triangle_point.y, triangle_point.z], 
                        color: cell_color, //encode_rgba_u32(100, 50, 255, 255),
                    });
                    
                }
                Cell::Band(dist, dir) => {
                    
                }
                Cell::Far() => {
                    // let cell_color = encode_rgba_u32(50, 155, 50, 255); 
                    // let cube = create_cube_triangles(grid_pos , 0.001, cell_color);
                    // result.extend(cube);
                }
            }
        }}};
        (grid_cubes, lines)
    }
}
