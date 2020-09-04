use std::collections::BinaryHeap;
use std::cmp::Reverse;
use ordered_float::NotNan;
use cgmath::{prelude::*,Vector3, Vector4};
use std::collections::HashMap;
use crate::bvh::{BBox, Triangle}; 
//use crate::misc::{Vertex_vvvc};
use crate::misc::*;
use crate::array3D::*;

const SQ3: f32 = 1.73205080757;

type MinNonNan = Reverse<NotNan<f32>>;

#[derive(Clone, Copy)]
pub enum Cell {
    Known(f32, Vector3<f32>, bool), // (distance, normal_vector, is_plus)
    Band(f32, bool, bool ), // (distance, in_heap, is_plus)
    Far(bool), // in_heap
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

//#[derive(Debug)]
//pub struct CellIndex {
//    i: u32,
//    j: u32,
//    k: u32,
//}
//
//#[derive(Debug)]
//pub struct Coordinate {
//    x: f32,
//    y: f32,
//    z: f32,
//}

/// Data structure for holding the information about the fmm state.
pub struct FmmDomain {
    //pub dimension: (std::ops::Range<u32>, std::ops::Range<u32>, std::ops::Range<u32>),
    //base_position: Vector3<f32>,
    //grid_length: f32,
    //cells: Vec<Cell>,
    domain: Array3D<Cell>,
    heap: BinaryHeap<std::cmp::Reverse<(ordered_float::NotNan<f32>, u32)>>,
    aabbs: Vec<BBox>,
    //heap: BinaryHeap<Reverse<(MinNonNan,u32)>>,
    //boundary_points: Option<Vec<f32>>,
    min_value: f32,
    max_value: f32,
    error_grids: Vec<(CellIndex, Vec<CellIndex>)>, // (error index, chosen neighbor indices)
}

impl FmmDomain {

    pub fn new(dimension: (u32, u32, u32), base_position: Vector3<f32>, grid_length: f32) -> Self {
        assert!(grid_length > 0.0, "FmmDomain.new: grid_length {} > 0.0", grid_length);

        let data: Vec<Cell> = vec![Cell::Far(false) ; (dimension.0 * dimension.1 * dimension.2) as usize];

        let mut array: Array3D<Cell> = Array3D::new(
            (dimension.0, dimension.1, dimension.2),
            Coordinate {x: base_position.x, y: base_position.y, z: base_position.z },
            grid_length);

        array.data = data;

        // Initialize cell points with far values.

        // Triangle aabb:s for debugging.
        let mut aabbs: Vec<BBox> = Vec::new(); 
        let error_grids: Vec<(CellIndex, Vec<CellIndex>)> = Vec::new();

        // // Initialize boundary points.
        // let mut boundary_points = None;
        // if boundary {
        //     boundary_points = Some(vec![0.0 ; ((dimension.0 + 1) * (dimension.1 + 1) * (dimension.2 + 1)) as usize]); 
        // }
        
        let heap: BinaryHeap<std::cmp::Reverse<(ordered_float::NotNan<f32>, u32)>> = BinaryHeap::new();

        let min_value = 0.0;
        let max_value = 0.0;

        Self {
            //dimension: (0..dimension.0, 0..dimension.1, 0..dimension.2),
            //base_position: base_position ,
            //grid_length: grid_length,
            //cells: cells,
            domain: array,
            heap: heap,
            //boundary_points: boundary_points,
            aabbs,
            min_value,
            max_value,
            error_grids,
        }
    }

    /// SOME NEW FUNCTIONS.

    /// Map given cell_index to a scaled and positioned x,y,z koordinates.
    //pub fn map_ijk_xyz(&self, cell_index: &CellIndex) -> Option<Coordinate> {
    //     
    //    self.validate_cell_index(&cell_index).ok()?;

    //    let x = cell_index.i as f32 * self.grid_length + self.base_position.x;
    //    let y = cell_index.j as f32 * self.grid_length + self.base_position.y;
    //    let z = cell_index.k as f32 * self.grid_length + self.base_position.z;

    //    Some(Coordinate {
    //        x: x,
    //        y: y,
    //        z: z,
    //    })
    //}

    ///// Map given coordinate to CellIndex.
    //pub fn map_xyz_ijk(&self, coordinate: &Coordinate, ceil: bool) -> Option<CellIndex> {
    //     
    //    // Move and scale coordinate to match CellIndex space.
    //    let x = (((coordinate.x - self.base_position.x) / self.grid_length) + 0.01) as f32;
    //    let y = (((coordinate.y - self.base_position.y) / self.grid_length) + 0.01) as f32;
    //    let z = (((coordinate.z - self.base_position.z) / self.grid_length) + 0.01) as f32;

    //    assert!(x >= 0.0 && y >= 0.0 && z >= 0.0, format!("map_xyz_ijk :: {} >= 0.0 && {} >= 0.0 && {} >= 0.0", x, y, z)); 

    //    let i = x as u32;
    //    let j = y as u32;
    //    let k = z as u32;

    //    let cell_index =
    //        if !ceil {
    //            CellIndex { i: x as u32, j: y as u32, k: z as u32 }
    //        }
    //        else {
    //            CellIndex { i: x.ceil() as u32, j: y.ceil() as u32, k: z.ceil() as u32 }
    //        };

    //    self.validate_cell_index(&cell_index).ok()?;

    //    Some(cell_index)
    //}

    ///// Boundary check for given cell index.
    //pub fn validate_cell_index(&self, cell_index: &CellIndex) -> Result<(), String> {
    //    if cell_index.i >= self.dimension.0.end ||
    //       cell_index.j >= self.dimension.1.end ||
    //       cell_index.k >= self.dimension.2.end  {
    //        let range = format!("[{}, {}, {}] - [{}, {}, {}]", 
    //            self.dimension.0.start,
    //            self.dimension.1.start,
    //            self.dimension.2.start,
    //            self.dimension.0.end,
    //            self.dimension.1.end,
    //            self.dimension.2.end
    //        );
    //        Err(format!("validate_cell_index :: cell index out of bounds. CellIndex(i: {}, j: {}, k: {}) not in range {}.", 
    //                cell_index.i, cell_index.j, cell_index.k, range))
    //       }
    //       else {
    //            Ok(())
    //       }
    //}

    ///// Set cell value at (i, j, k) position.
    //pub fn set_data(&mut self, i: u32, j: u32, k: u32, cell: Cell) {
    //    let index = self.get_array_index(i, j, k).unwrap() as usize; 
    //    self.cells[index] = cell;
    //}

    //pub fn set_data_ind(&mut self, index: u32) {
    //    
    //}

    ///// Get cell value at (i, j, k) position. TODO: the index should always be checked and return
    ///// Optional<Cell>.
    //pub fn get_data(&self, i: u32, j: u32, k: u32) -> Option<Cell> {
    //    if let Ok(index) = self.get_array_index(i, j, k) { // TODO: check?   
    //        Some(self.cells[index as usize])
    //    }
    //    else {
    //        None
    //    }
    //}

    //pub fn cell_to_aabb(&self, i:u32, j:u32, k:u32) -> Option<BBox> {
    //    None   
    //}

    ///// Map given world position to nearest cell array index coordinates. Returns None if given point is outside
    ///// computational domain. SHOULD WORK NOW
    //pub fn xyz_to_ijk(&self, world_pos: &Vector3<f32>, snap_to_farest: bool) -> Option<(u32, u32, u32)> {

    //    let mut i: u32 = 0;
    //    let mut j: u32 = 0;
    //    let mut k: u32 = 0;
    //    // println!("nuapurit!");
    //    // let neighbors = self.xyz_to_all_nearest_ijk(&world_pos);
    //    // let mut closest: Option<(u32, u32, u32)> = None; 

    //    // match neighbors {
    //    //     None => { return None; }
    //    //     Some((v_ijk)) => {
    //    //         let mut distance = 1000000000.0;
    //    //         for (r, q, s) in &v_ijk {
    //    //             let t = self.ijk_to_xyz(*r, *q, *s).unwrap(); 
    //    //             let d = world_pos.distance(t);
    //    //             if d < distance { closest = Some((*r, *q, *s)); } 
    //    //             distance = d;
    //    //         }
    //    //     }
    //    // }

    //    //closest


    //    if snap_to_farest {
    //        i = ((world_pos.x + self.grid_length * 0.5 - self.base_position.x) / self.grid_length).ceil() as u32;
    //        j = ((world_pos.y + self.grid_length * 0.5 - self.base_position.y) / self.grid_length).ceil() as u32;
    //        k = ((world_pos.z + self.grid_length * 0.5 - self.base_position.z) / self.grid_length).ceil() as u32;
    //    }
    //    else {
    //        // i = ((world_pos.x - self.base_position.x + 0.5) / self.grid_length).floor() as u32;
    //        // j = ((world_pos.y - self.base_position.y + 0.5) / self.grid_length).floor() as u32;
    //        // k = ((world_pos.z - self.base_position.z + 0.5) / self.grid_length).floor() as u32;
    //        // i = ((world_pos.x - self.base_position.x) / self.grid_length).floor() as u32;
    //        // j = ((world_pos.y - self.base_position.y) / self.grid_length).floor() as u32;
    //        // k = ((world_pos.z - self.base_position.z) / self.grid_length).floor() as u32;
    //        let x = (((world_pos.x - self.base_position.x) / self.grid_length) + 0.01) as f32;
    //        let y = (((world_pos.y - self.base_position.y) / self.grid_length) + 0.01) as f32;
    //        let z = (((world_pos.z - self.base_position.z) / self.grid_length) + 0.01) as f32;
    //        println!("x :: {}, y :: {}, z :: {}", x, y, z);
    //        i = x as u32;
    //        j = y as u32;
    //        k = z as u32;
    //    }
    //    //println!("i :: {}, j :: {}, k :: {}", i, j, k);

    //    match self.check_cell_index(i, j, k) {
    //        Ok(()) => { 
    //            return Some((i, j, k));
    //        }
    //        Err(_) => {
    //            //println!("{}", s);
    //            return None;
    //        }
    //    }
    //}

    ///// Map cell array index coordinates to world position. Returns None if i, j, k is outside
    ///// index space. Should work now. 
    //pub fn ijk_to_xyz(&self, i: u32, j: u32, k: u32) -> Option<Vector3<f32>> {

    //    match self.check_cell_index(i, j, k) {
    //        Ok(()) => { 
    //            let x = i as f32 * self.grid_length + self.base_position.x;
    //            let y = j as f32 * self.grid_length + self.base_position.y;
    //            let z = k as f32 * self.grid_length + self.base_position.z;
    //            //println!("ijk_to_xyz({}, {}, {}) == ({}, {}, {})", i, j, k, x, y, z);
    //            return Some(Vector3::<f32>::new(x, y, z));
    //        }
    //        _ => {
    //            println!("({}, {}, {}) are outside" , i, j, k);
    //            return None;
    //        }
    //    }
    //}

    ///// Get the nearest cell point from given world coordinates (xyz). Returns None if given world
    ///// position is outside the computational domain. Should work now.
    //pub fn xyz_to_nearest_cell_xyz(&self, world_pos: &Vector3<f32>) -> Option<Vector3<f32>> {
    //     
    //    if let Some((i,j,k)) = self.xyz_to_ijk(&world_pos, false) {
    //        if let Some(v) = self.ijk_to_xyz(i, j, k) {
    //            return Some(v);
    //        }
    //        else { // is this necessery? TODO: remove if this is never reached.
    //            return None;
    //        }
    //    }
    //    None
    //}

    //// :noremap <F12> :wall \| !./native_compile.sh && ./runNative.sh<CR>

    ///// Get all nearest cell array indices which satisfies the condition: dist(world_pos,
    ///// cell_point) <= sqrt(3) * cube_length / 2. Return None if the given world position is too far
    ///// away from any cell point.
    //pub fn xyz_to_all_nearest_ijk(&self, world_pos: &Vector3<f32>) -> Option<Vec<(u32, u32, u32)>> {

    //    // If the given position is inside the computational domain.
    //    if let Some((i, j, k)) = self.xyz_to_ijk(&world_pos, false) {

    //        // Seach for all 26 neighbors. Only test for those indices that lies inside
    //        // computational domain. 
    //       
    //        let mut cell_indices: Vec<(u32, u32, u32)> = Vec::new();
    //        //println!("nearest:");

    //        // TODO: is this necessery?
    //        let a_range = std::ops::Range::<i32> { start: i as i32 - 1, end: i as i32 + 2 };
    //        let b_range = std::ops::Range::<i32> { start: j as i32 - 1, end: j as i32 + 2 };
    //        let c_range = std::ops::Range::<i32> { start: k as i32 - 1, end: k as i32 + 2 };

    //        for a in a_range {
    //        for b in b_range.clone() {
    //        for c in c_range.clone() {

    //            // Skip negative indices.
    //            if a < 0 || b < 0 || c < 0 { continue; }

    //            if let Ok(()) = self.check_cell_index(a as u32, b as u32, c as u32) {
    //                let cell_point = self.ijk_to_xyz(a as u32, b as u32, c as u32).unwrap();
    //                if world_pos.distance(cell_point) <= SQ3 * self.grid_length * 0.5 { 
    //                    println!("cell_point == ({}, {}, {})", cell_point.x, cell_point.y, cell_point.z);
    //                    cell_indices.push((a as u32, b as u32, c as u32));
    //                    println!("({}, {}, {}) added", a, b, c);
    //                }
    //            }
    //        }}};

    //        Some(cell_indices)
    //    }
    //    else {
    //        None
    //    }
    //}

    ///// Get all nearest cell array indices which lies inside computational domain.
    //pub fn ijk_neighbors(&self, i: u32, j: u32, k: u32) -> Option<Vec<(u32, u32, u32)>> {

    //    // If the given position is inside the computational domain.
    //    if let Ok(()) = self.check_cell_index(i, j, k) {

    //        // Seach for all 6 neighbors. Only test for those indices that lies inside
    //        // computational domain. 
    //       
    //        let mut cell_indices: Vec<(u32, u32, u32)> = Vec::new();

    //        // Negative indices are now very big numbers (u32: -1 is a big number). It works, but it could be better to
    //        // reject negative indices.

    //        let mut neighbor_indices = Vec::new();   
    //        neighbor_indices.push((i+1, j  , k)); 
    //        neighbor_indices.push((i  , j+1, k)); 
    //        neighbor_indices.push((i  , j  , k+1)); 

    //        if i != 0 { neighbor_indices.push((i-1, j  , k)); } 
    //        if j != 0 { neighbor_indices.push((i  , j-1, k)); } 
    //        if k != 0 { neighbor_indices.push((i  , j  , k-1)); } 

    //        // let neighbor_indices = [
    //        //     (i+1, j,   k), // neigbor on the right
    //        //     (i-1, j,   k), // neigbor on the left
    //        //     (i  , j+1, k), // neigbor on the up direction.
    //        //     (i  , j-1, k), // neigbor on the down direction.
    //        //     (i  , j  , k+1), // neigbor on the front direction.
    //        //     (i  , j  , k-1), // neigbor on the back direction.
    //        // ];

    //        for neighbor in neighbor_indices.iter() {

    //            // Skip negative indices.
    //            // if i < neighbor.0 || neighbor.1 < 0 || neighbor.2 < 0 { continue; }

    //            if let Ok(()) = self.check_cell_index(neighbor.0, neighbor.1, neighbor.2) {
    //                //println!("cell_indicex.push(({}, {}, {}))", neighbor.0, neighbor.1, neighbor.2);
    //                cell_indices.push((neighbor.0, neighbor.1, neighbor.2));
    //            }
    //        }

    //        Some(cell_indices)
    //    }
    //    // i, j, k is outside the computational domain. 
    //    else {
    //        panic!(format!("ijk_neighbors:: ({}, {}, {}) is outside computational domain ({}, {}, {})", 
    //            i, j, k, self.dimension.0.end, self.dimension.1.end, self.dimension.2.end));
    //        //None
    //    }
    //}

    /// Return all world coordinates which are inside a cell centered ball r less than the length
    /// from cell center to the nearest cell cube corner. Returns None if given world coordinate
    /// is too far away from any cell center. Seems to work.
    //pub fn xyz_to_all_nearest_cells_xyz(&self, world_pos: &Vector3<f32>) -> Option<Vec<Vector3<f32>>> {

    //    let mut result: Vec<Vector3<f32>> = Vec::new();

    //    // Find all nearest cell i, j, k with distance condition. 
    //    if let Some(v_ijk) = self.xyz_to_all_nearest_ijk(&world_pos) {
    //        for i in v_ijk.iter() {
    //            result.push(self.ijk_to_xyz(i.0, i.1, i.2).unwrap());
    //            let temp = self.ijk_to_xyz(i.0, i.1, i.2).unwrap();
    //        }
    //    }

    //    // At least one found.
    //    if result.len() > 0 { Some(result) }

    //    // The point is too far from computational context. No cell found.
    //    else { None }
    //}

    /// Get the nearest cell point based on given w_pos coordinate.
    // pub fn get_neighbor_grid_points(&self, w_pos: &Vector3<f32>) -> (Vec<Vertex_vvvc_nnnn>, Vec<Vertex_vvvc>) {

    //     // Cubes.
    //     let mut result: Vec<Vertex_vvvc_nnnn> = Vec::new();

    //     // Lines.
    //     let mut result2: Vec<Vertex_vvvc> = Vec::new();

    //     // let coords = self.xyz_to_all_nearest_cells_xyz(&w_pos);
    //     // match coords {
    //     //     Some(v_cell_positions) => { 
    //     //         let cell_color = encode_rgba_u32(255, 50, 50, 255); 
    //     //         for cell_pos in v_cell_positions.iter() {
    //     //             let cube = create_cube_triangles(cell_pos - Vector3::<f32>::new(0.004, 0.004, 0.004), 0.008, cell_color);
    //     //             result.extend(cube);
    //     //             result2.push(Vertex_vvvc {
    //     //                 position: [cell_pos.x, cell_pos.y, cell_pos.z],
    //     //                 color: encode_rgba_u32(255, 50, 50, 255),
    //     //             });
    //     //             result2.push(Vertex_vvvc {
    //     //                 position: [w_pos.x, w_pos.y, w_pos.z], 
    //     //                 color: encode_rgba_u32(100, 50, 255, 255),
    //     //             });
    //     //         }
    //     //     },
    //     //    _ => { println!("nuapira ei löövy"); }
    //     // }

    //     // Only one cell center.
    //     let coord = self.xyz_to_nearest_cell_xyz(&w_pos);
    //     match coord {
    //         Some(cell_pos) => { 

    //             let cell_color = encode_rgba_u32(255, 50, 50, 255); 
    //             let cube = create_cube_triangles(cell_pos, 0.006, cell_color);
    //             result.extend(cube);
    //             result2.push(Vertex_vvvc {
    //                 position: [cell_pos.x, cell_pos.y, cell_pos.z],
    //                 color: encode_rgba_u32(255, 50, 50, 255),
    //             });
    //             result2.push(Vertex_vvvc {
    //                 position: [w_pos.x, w_pos.y, w_pos.z], 
    //                 color: encode_rgba_u32(100, 50, 255, 255),
    //             });
    //         },
    //        _ => { println!("nuapira ei löövy"); }
    //     }
    //     (result, result2)
    // }

    /// Create vvvv data from cell data. Cell color is encoded to in vvvc form, where vvv part is
    /// the coordinates xyz and c is encoded rgba information. 
    pub fn cells_to_vvvc_nnnn(&self) -> Vec<Vertex_vvvc_nnnn> {

        let mut result: Vec<Vertex_vvvc_nnnn> = Vec::new();
        for i in 0..self.domain.dimension.0.end { 
        for j in 0..self.domain.dimension.1.end { 
        for k in 0..self.domain.dimension.2.end { 
            let cell = self.domain.get_data(CellIndex{i: i, j: j, k: k}).unwrap(); //cells[self.get_array_index(i, j, k).unwrap() as usize]; 
            let cell_color = match cell {
                Cell::Known(_,_,_) => { encode_rgba_u32(0, 255, 0, 255) } 
                Cell::Band(_,_,_)  => { encode_rgba_u32(255, 255, 0, 255) }  
                Cell::Far(_)    => { encode_rgba_u32(255, 0, 255, 0) }  
            };
            let cell_ws_pos = self.domain.map_ijk_xyz(&CellIndex {i: i, j: j, k: k}).unwrap();
            let cube = create_cube_triangles(Vector3::<f32>::new(cell_ws_pos.x, cell_ws_pos.y, cell_ws_pos.z), 0.004, cell_color);

            result.extend(cube);
        }}};
        result
    }

    /// Create vvvc cell cube data. 
    pub fn boundary_lines(&self) -> Vec<Vertex_vvvc> {

        let mut result: Vec<Vertex_vvvc> = Vec::new();

        let offset = self.domain.grid_length * 0.5;
        let grid_offset = self.domain.grid_length;
        for i in 0..self.domain.dimension.0.end { 
        for j in 0..self.domain.dimension.1.end { 
        for k in 0..self.domain.dimension.2.end { 

            // Create cube.
            let cell_ws_pos = self.domain.map_ijk_xyz(&CellIndex {i: i, j: j, k: k}).unwrap();

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
    //pub fn check_cell_index(&self, i: u32, j: u32, k: u32) -> Result<(), String> {

    //    let x_ok = self.dimension.0.contains(&i);
    //    let y_ok = self.dimension.1.contains(&j);
    //    let z_ok = self.dimension.2.contains(&k);

    //    if !x_ok {
    //        return Err(format!("FmmDomain::check_cell_index(i: {}, j: {}, k: {}) :: i not in range 0..{}", i, j ,k, self.dimension.0.end).to_string());
    //    }
    //    if !y_ok {
    //        return Err(format!("FmmDomain::check_cell_index(i: {}, j: {}, k: {}) :: j not in range 0..{}", i, j ,k, self.dimension.0.end).to_string());
    //    }
    //    if !z_ok {
    //        return Err(format!("FmmDomain::check_cell_index(i: {}, j: {}, k: {}) :: k not in range 0..{}", i, j ,k, self.dimension.0.end).to_string());
    //    }
    //    
    //    Ok(())
    //}

    ///// Get array index for cell coordinates i,j,k. 
    //pub fn get_array_index(&self, i: u32, j: u32, k: u32) -> Result<u32, String> {

    //    // Check if index is inside domain bounds.
    //    match self.check_cell_index(i, j, k) {
    //        Err(s) => {
    //            let mut error = format!("FmmDomain::get_array_index({}, {}, {})\n", i, j, k).to_string();
    //            error.push_str(&s);
    //            return Err(error);
    //            }
    //        _ => { }
    //    }

    //    let x_range = self.dimension.0.end; 
    //    let y_range = self.dimension.1.end; 
    //    let index = i + j * x_range + k * x_range * y_range;

    //    Ok(index)
    //}

    ///// Get cell coordinates i, j, k.
    //pub fn array_index_to_ijk(&self, array_index: u32) -> Option<(u32, u32, u32)> {
    //    if array_index >= self.cells.len() as u32 {
    //        None
    //    }
    //    else {
    //        let i = array_index % self.dimension.0.end;
    //        let j = (array_index  / self.dimension.0.end ) % self.dimension.1.end;
    //        let k = array_index / (self.dimension.0.end * self.dimension.1.end);
    //        Some((i, j, k))
    //    }
    //}

    // Unused.
    pub fn get_domain_bounding_box(&self) -> BBox {
        let base_position = Vector3::<f32>::new(self.domain.base_position.x, self.domain.base_position.y, self.domain.base_position.z);
        let bbox = BBox::create_from_line(
            &base_position,
            &(base_position
                + Vector3::<f32>::new(
                    self.domain.dimension.0.end as f32 * self.domain.grid_length,
                    self.domain.dimension.1.end as f32 * self.domain.grid_length,
                    self.domain.dimension.2.end as f32 * self.domain.grid_length)
            ),
        );

        bbox
    }

    // Update initial state with given triangle (a,b,c). If some of triangle vertices are outside
    // the computational domain, the triangle is rejected.
    pub fn add_triangles(&mut self, triangles: &Vec<Triangle>) {

        let domain_aabb = self.get_domain_bounding_box();

        //let mut tr_counter = 0;
        //let mut tr_stop = 1;
        for tr in triangles {
            // if tr_counter == tr_stop { break; }
            // tr_counter += 1;
            // Triangle is inside computational domain.
            if domain_aabb.includes_point(&tr.a, true) && domain_aabb.includes_point(&tr.b, true) && domain_aabb.includes_point(&tr.c, true) {

                // TODO: implement from triangle type.
                let aabb = BBox::create_from_triangle(&tr.a, &tr.b, &tr.c);

                // expand to nearest cell grid points.

                //let min_indices = self.xyz_to_ijk(&aabb.min, true);
                //let min_indices = self.xyz_to_ijk(&aabb.min, true);

                //let mut i_min = 0;
                //let mut j_min = 0;
                //let mut k_min = 0;
                //let mut i_max = 0;
                //let mut j_max = 0;
                //let mut k_max = 0;

                //// if there is something wrong with indeces, ignore this triangle.
                //match min_indices {
                //    Some((some_i_min, some_j_min, some_k_min)) => {
                //        i_min = some_i_min - 1; // TODO: check if some_i_min is zero. 
                //        j_min = some_j_min - 1; // TODO: check if some_i_min is zero.
                //        k_min = some_k_min - 1; // TODO: check if some_i_min is zero.
                //    },
                //    None => { println!("TROUBLES!!!"); continue; },
                //}

                ////let max_indices = self.xyz_to_ijk(&aabb.max, true);
                //let max_indices = self.xyz_to_ijk(&aabb.max, true);
                //match max_indices {
                //    Some((some_i_max, some_j_max, some_k_max)) => {
                //        i_max = some_i_max;
                //        j_max = some_j_max;
                //        k_max = some_k_max;
                //        
                //    },
                //    None => { println!("TROUBLES!!!");continue; },
                //}

                let triangle_bounds = self.domain.aabb_to_range(&aabb).unwrap(); // -> Option<(std::ops::Range<u32>, std::ops::Range<u32>, std::ops::Range<u32>)> {

                let min_xyz = self.domain.map_ijk_xyz(&CellIndex {i: triangle_bounds.0.start, j: triangle_bounds.1.start, k: triangle_bounds.2.start}).unwrap();
                let max_xyz = self.domain.map_ijk_xyz(&CellIndex {i: triangle_bounds.0.end, j: triangle_bounds.1.end, k: triangle_bounds.2.end}).unwrap();
                let aabb_extended_tr = BBox::create_from_line(&Vector3::<f32>::new(min_xyz.x, min_xyz.y, min_xyz.z),
                                                              &Vector3::<f32>::new(max_xyz.x, max_xyz.y, max_xyz.z));
                self.aabbs.push(aabb_extended_tr);

                

                ////let mut temp_cells: HashMap<u32, (Vec<Vertex_vvvc_nnnn>, Vec<Vertex_vvvc>)> = HashMap::new();

                // Update the closest cell points to the triangle.
                for q in triangle_bounds.0.start..triangle_bounds.0.end+1 {
                for r in triangle_bounds.1.start..triangle_bounds.1.end+1 {
                for s in triangle_bounds.2.start..triangle_bounds.2.end+1 {

                    let cell_index = CellIndex { i:q, j:r, k:s };
                    let cell_pos = self.domain.map_ijk_xyz(&cell_index).unwrap();

                    // Test if cell and cell coordinates are correct.
                    //let inverse_cell_pos = self.xyz_to_ijk(&cell_pos, false).unwrap();
                    //if (q, r, s) != inverse_cell_pos {
                    //    //panic!(format!("inverse vailed! Reason :: ({}, {}, {}) != ({}, {}, {})", q, r, s, inverse_cell_pos.0, inverse_cell_pos.1, inverse_cell_pos.2));
                    //    println!("inverse vailed! Reason :: Inv ({}, {}, {}) != Original ({}, {}, {})", q, r, s, inverse_cell_pos.0, inverse_cell_pos.1, inverse_cell_pos.2);
                    //}

                    let (distance, sign) = tr.distance_to_triangle(&Vector3::<f32>::new(cell_pos.x, cell_pos.y, cell_pos.z));
                    let signed_distance = match sign { true => distance, false => -distance }; 
                    let triangle_point = tr.closest_point_to_triangle(&Vector3::<f32>::new(cell_pos.x, cell_pos.y, cell_pos.z));
                    //if distance >= self.domain.grid_length.sqrt() {
                    if distance >= self.domain.grid_length {
                    //if distance > SQ3 * 0.5 * self.domain.grid_length {
                    //if distance > SQ3 * self.domain.grid_length {
                    //let three: f32 = 3.0;
                    //if distance > three.powf(1.0/3.0) as f32 * self.grid_length {
                        continue;
                    }
                    else {
                        //if distance.abs() > 2.0 { println!("distance == {}", distance); }
                        let mut cell = self.domain.get_data(cell_index).unwrap(); //TODO: check?

                        let mut new_found = false;
                        let normal = (Vector3::<f32>::new(cell_pos.x, cell_pos.y, cell_pos.z) - triangle_point).normalize(); //ac.cross(ab).normalize();
                        match cell {
                            Cell::Known(val,_,_) => {
                                if distance < val.abs() { cell = Cell::Known(distance, normal, sign); self.domain.set_data(cell_index, cell); new_found = true; } 
                                //if distance < val.abs() { cell = Cell::Known(signed_distance, normal, true); self.domain.set_data(cell_index, cell).unwrap(); new_found = true; } 
                            }
                            Cell::Band(val,_,_) => {
                                if distance < val.abs() { cell = Cell::Known(distance, normal, sign); self.domain.set_data(cell_index, cell); new_found = true; } 
                                //if distance < val.abs() { cell = Cell::Known(signed_distance, normal, true); self.domain.set_data(cell_index, cell).unwrap(); new_found = true; } 
                            }
                            Cell::Far(_) => {
                                cell = Cell::Known(distance, normal, sign); self.domain.set_data(cell_index, cell); new_found = true;
                                //cell = Cell::Known(signed_distance, normal, true); self.domain.set_data(cell_index, cell).unwrap(); new_found = true;
                            }
                        }
                    }
                }}};
            }
        }
    }

    pub fn create_grid_point_and_line_data(&self, show_known: bool, show_band: bool, show_far: bool, create_lines: bool) -> (Vec<Vertex_vvvc_nnnn>, Vec<Vertex_vvvc>) {

        // Cubes.
        let mut grid_cubes: Vec<Vertex_vvvc_nnnn> = Vec::new();

        // Lines.
        let mut lines: Vec<Vertex_vvvc> = Vec::new();


        for i in 0..self.domain.dimension.0.end {
        for j in 0..self.domain.dimension.1.end {
        for k in 0..self.domain.dimension.2.end {
            let cell_index = CellIndex {i: i, j: j, k: k};
            let item = self.domain.get_data(cell_index).unwrap(); // TODO: check?
            let grid_pos = self.domain.map_ijk_xyz(&cell_index).unwrap(); 
            match item {
                Cell::Known(dist, dir, is_plus) => {
                    if show_known {

                        //let sign = if dist < 0.0 { false } else { true }; 
                        let grid_pos = self.domain.map_ijk_xyz(&cell_index).unwrap(); 
                        let triangle_point = Vector3::<f32>::new(grid_pos.x, grid_pos.y, grid_pos.z) - dir * dist.abs(); 
                        let cell_color = match is_plus { true => encode_rgba_u32(255, 50, 50, 255), false => encode_rgba_u32(50, 50, 200, 255) } ; 
                        let cube = create_cube_triangles(Vector3::<f32>::new(grid_pos.x, grid_pos.y, grid_pos.z) , 0.008, cell_color);
                        grid_cubes.extend(cube);
                        lines.push(Vertex_vvvc {
                            position: [grid_pos.x, grid_pos.y, grid_pos.z],
                            color: cell_color,
                        });
                        lines.push(Vertex_vvvc {
                            position: [triangle_point.x, triangle_point.y, triangle_point.z], 
                            color: cell_color,
                        });
                    }
                }
                Cell::Band(dist, dir, _) => {
                    if show_band {    
                        let grid_pos = self.domain.map_ijk_xyz(&cell_index).unwrap(); 
                        let cell_color = encode_rgba_u32(50, 250, 80, 255); 
                        let cube = create_cube_triangles(Vector3::<f32>::new(grid_pos.x, grid_pos.y, grid_pos.z) , 0.008, cell_color);
                        grid_cubes.extend(cube);
                    }
                }
                Cell::Far(_) => {
                    if show_far {    
                        let grid_pos = self.domain.map_ijk_xyz(&cell_index).unwrap(); 
                        let cell_color = encode_rgba_u32(50, 50, 255, 255); 
                        let cube = create_cube_triangles(Vector3::<f32>::new(grid_pos.x, grid_pos.y, grid_pos.z) , 0.008, cell_color);
                        grid_cubes.extend(cube);
                    }
                    // let cell_color = encode_rgba_u32(50, 155, 50, 255); 
                    // let cube = create_cube_triangles(grid_pos , 0.001, cell_color);
                    // result.extend(cube);
                }
            }
        }}};

    // error_grids: Vec<CellIndex>,
    // error_grids_neighbors: Vec<CellIndex>,
        for (err_cell_index, err_neighbors) in self.error_grids.iter() {
            let grid_pos = self.domain.map_ijk_xyz(&err_cell_index).unwrap(); 
            let cell_color = encode_rgba_u32(255, 0, 0, 255); 
            let cube = create_cube_triangles(Vector3::<f32>::new(grid_pos.x, grid_pos.y, grid_pos.z) , 0.018, cell_color);
            grid_cubes.extend(cube);
            for err_neig in err_neighbors {
                let grid_pos_n = self.domain.map_ijk_xyz(&err_neig).unwrap(); 
                let cell_color_n = encode_rgba_u32(255, 255, 0, 255); 
                let cube = create_cube_triangles(Vector3::<f32>::new(grid_pos_n.x, grid_pos_n.y, grid_pos_n.z) , 0.018, cell_color_n);
                grid_cubes.extend(cube);

                lines.push(Vertex_vvvc {
                    position: [grid_pos.x, grid_pos.y, grid_pos.z],
                    color: cell_color,
                });
                lines.push(Vertex_vvvc {
                    position: [grid_pos_n.x, grid_pos_n.y, grid_pos_n.z], 
                    color: cell_color,
                });
            }

        }
        (grid_cubes, lines)
    }
    pub fn fmm_domain_swap_signs(&mut self) {
        for i in 0..self.domain.dimension.0.end {
        for j in 0..self.domain.dimension.1.end {
        for k in 0..self.domain.dimension.2.end {

            let cell_index = CellIndex {i: i, j: j, k: k};

            // Swap the signs of knonw cells.
            if let Some(Cell::Known(dist, v, s)) = self.domain.get_data(cell_index) {
                let swapped_cell = Cell::Known(dist, v, !s);
                //let swapped_cell = Cell::Known(-dist, v, !s);
                self.domain.set_data(cell_index, swapped_cell);
            }
        }}};

    }

    ///
    pub fn fmm_initialize_heap(&mut self) {
        for i in 0..self.domain.dimension.0.end {
        for j in 0..self.domain.dimension.1.end {
        for k in 0..self.domain.dimension.2.end {

            let cell_index = CellIndex {i: i, j: j, k: k};

            // We found a known point with distance. Only update the cell points wiith posivie values.
            if let Some(Cell::Known(dist, _, sign)) = self.domain.get_data(cell_index)  {
                // println!("We found a knwonw point ({}, {}, {})", i, j, k);


                //if dist >= 0.0 {
                if sign {
                    //println!("dist == {}", dist);
                    self.fmm_update_neighbors(i, j, k);  
                }
            }
        }}};
    }

    /// Update neighbors of a known point.
    pub fn fmm_update_neighbors(&mut self, i: u32, j: u32, k: u32) {

        let debug_cell = self.domain.get_data(CellIndex {i: i, j: j, k: k}).unwrap();
        let mut sign = true;
        match debug_cell {
            Cell::Known(val, _, is_plus) => { sign = is_plus; /* println!("fmm_update_neigbors {}, {}, {}, {} :: OK", val, i, j, k); */ }
            Cell::Band(_, _, _) => { panic!("fmm_update_neighbors({}, {}, {}) with a band point", i, j, k); }
            Cell::Far(_) => { panic!("fmm_update_neighbors({}, {}, {}) with a far point", i, j, k); }
        }

        if sign {
            // Find neighbors.
            let neighbors = self.domain.get_neigbors_from_ijk(CellIndex{i: i, j: j, k: k}); 
            //println!("FMM_UPDATE_NEIGHBORS ({}, {}, {})", i, j, k);
            // let this_cell = self.domain.get_data(CellIndex{i: i, j: j, k: k}).unwrap();
            // match this_cell {
            //     Cell::Known(_,_,_) => { println!("This is a knwon grid point."); }
            //     Cell::Band(_,_,_) => { println!("This is a band grid point."); }
            //     Cell::Far(_) => { println!("This is a far grid point."); }
            // }
            for cell_index in neighbors.iter() {
                //let cell_index = CellIndex {i: *a, j: *b, k: *c};
                // Get neighbor cells.
                let cell = self.domain.get_data(*cell_index).unwrap(); // TODO: Checking?
                //println!("And the neighbor cell_index are :: ({}, {}, {})", cell_index.i, cell_index.j, cell_index.k);

                // If 
                match cell {
                    // A far point found. Update to band.
                    Cell::Far(_) => {

                        ////println!("Going to solve quadratic Far CellIndex{{ i: {}, j: {}, k: {} }}", cell_index.i, cell_index.j, cell_index.k);
                        let mut phi_temp = self.fmm_solve_quadratic(cell_index.i, cell_index.j, cell_index.k, 666.0, true);
                        //if phi_temp == 666.0 || phi_temp == -666.0 {
                        //    panic!("NOOOOOOO!!!");
                        //}
                        //println!("Updating Far point ({}, {}, {}) to Band with value {}", *a, *b, *c, phi_temp); 

                        // Add to heap and update current grid point to be Band point.
                        //let joo =   Reverse((NotNan::new(1.0).unwrap(), 66));
                        let value = Reverse((NotNan::new(phi_temp).unwrap(), self.domain.cell_index_to_array_index(*cell_index).unwrap()));
    
                        self.heap.push(value); 
                        //println!("UPDATING neighbor cell_index to band :: ({}, {}, {}) with value of {}", cell_index.i, cell_index.j, cell_index.k, phi_temp);
                        self.domain.set_data(*cell_index, Cell::Band(phi_temp, true, sign));
                    }
                    Cell::Band(dist, in_heap, is_plus) => {
                        ////println!("Going to solve quadratic Band CellIndex{{ i: {}, j: {}, k: {} }} :: dist == {}", cell_index.i, cell_index.j, cell_index.k, dist);
                        let mut phi_temp = self.fmm_solve_quadratic(cell_index.i, cell_index.j, cell_index.k, dist, true);
                        //if phi_temp == 666.0 || phi_temp == -666.0 {
                        //    panic!("NOOOOOOO2!!!");
                        //}
                        if phi_temp < dist || !in_heap {

                            //let updated_cell = Cell::Band(phi_temp, true);  

                            // We add this value to the heap even the contains already an phi value
                            // for this cell.
                            let value = Reverse((NotNan::new(phi_temp).unwrap(), self.domain.cell_index_to_array_index(*cell_index).unwrap()));
                            self.heap.push(value); 
                            //println!("UPDATING neighbor cell_index to band :: ({}, {}, {}) with value of {}", cell_index.i, cell_index.j, cell_index.k, phi_temp);
                            self.domain.set_data(*cell_index, Cell::Band(phi_temp, true, is_plus));
                            //println!("Updating Band point ({}, {}, {}) to Band with new value {}", *a, *b, *c, phi_temp); 

                            // // We need to update the old value in the heap.
                            // if in_heap {
                            //  
                            // }
                        }
                    }

                    // Ignore known cells and cells that doesn't exist.
                    _ => { /* println!("A knwon cell. Skipping."); */ }
                    // Cell::Known(_, _) => { println!("Found a neighbor a Known ({}, {}, {}) neighbor", a, b, c); }
                    // Cell::Band(_, _) => { println!("Found a neighbor a Band ({}, {}, {}) neighbor", a, b, c); }
                }
            }
            
            //; :: ({}, {}, {}", i, j, k);
        }
    }

    pub fn fmm_solve_quadratic2(&mut self, i: u32, j: u32, k: u32) -> f32 {

        // For debugging.
        let mut chosen_neighbors: Vec<CellIndex> = Vec::new();

        let mut nd = 0;
        let d = 0;
        let mut this_phi = 666.0;

        // let this_cell = self.domain.get_data(CellIndex {i: i, j: j, k: k}); 
        // match this_cell {
        //     Some(Cell::band(dist, _, sign) => {

        //     }
        //     Some(Cell::Far(_) => {

        //     }
        // }

        if let Some(Cell::Band(dist, _, sign)) = self.domain.get_data(CellIndex {i: i, j: j, k: k}) {
            this_phi = dist;
        }

        // Calculate the phi values.
        //    0     1     2     3     4     5    6
        // +-----------------------------------------+
        // | ijk | i-1 | i+1 | j-1 | j+1 | k-1 | k+1 |
        // +-----------------------------------------+
        //

        let mut phis = [ 0.0 ; 3];
        let mut pointer = 0;
        let mut x_dir_found = false;
        let mut y_dir_found = false;
        let mut z_dir_found = false;
        //phis[0] = this_phi;

        // Get all neihbors.
        //let neighbors = self.domain.get_neigbors_from_ijk(CellIndex {i: i, j: j, k: k});

        //for i in neighbors.len() {
        //    match neighbors[i] {
        //        Cell::Known(dist, _, plus) => {
        //            
        //        }
        //        _ => {
        //            // Nothing to do.
        //        }
        //    }
        //}

        let mut x_minus_chosen = false;
        let mut x_plus_chosen = false;
        // Must be greater than 0. See the substraction i - 1. Check for the left side of the cell. 
        if i > 0 {

            // Found a known point from the left.
            if let Some(Cell::Known(dist, _, sign)) = self.domain.get_data(CellIndex {i: i-1, j: j, k: k}) {
                println!("CellIndex {{i: {}, j: {}, k: {}, dist: {}}}", i-1, j, k, dist);
                assert!(sign, "pah");
                                    // Update value to array if its smaller than current phi.
                if dist < this_phi {
                    x_minus_chosen = true;
                    phis[pointer] = dist; 
                    x_dir_found = true;
                }
            }
        }

        // Found a known point from the right side.
        if let Some(Cell::Known(dist, _, sign)) = self.domain.get_data(CellIndex {i: i+1, j: j, k: k}) {
                println!("CellIndex {{i: {}, j: {}, k: {}, dist: {}}}", i+1, j, k, dist);
                assert!(sign, "pah");
                if dist < phis[pointer]  {
                    x_plus_chosen = true;
                    x_minus_chosen = false;
                    phis[pointer] = dist; 
                    x_dir_found = true;
                }
                else if dist < this_phi {
                    phis[pointer] = dist; 
                    x_dir_found = true;
                    x_plus_chosen = true;
                    x_minus_chosen = false;
                }
        }
        if x_dir_found { pointer += 1; }

        let mut y_minus_chosen = false;
        let mut y_plus_chosen = false;
        // Check for y-direction.
        if j > 0 {

            // Found a known point from the down.
            if let Some(Cell::Known(dist, _, sign)) = self.domain.get_data(CellIndex {i: i, j: j-1, k: k}) {
                println!("CellIndex {{i: {}, j: {}, k: {}, dist: {}}}", i, j-1, k, dist);
                assert!(sign, "pah");
                // Update value to array if its smaller than current phi.
                if dist < this_phi {
                    phis[pointer] = dist; 
                    y_dir_found = true;
                    y_minus_chosen = true;
                }
            }
        }

        // Found a known point from the up side.
        if let Some(Cell::Known(dist, _, sign)) = self.domain.get_data(CellIndex {i: i, j: j+1, k: k}) {
                println!("CellIndex {{i: {}, j: {}, k: {}, dist: {}}}", i, j+1, k, dist);
                assert!(sign, "pah");
                if dist < phis[pointer]  {
                    phis[pointer] = dist; 
                    y_dir_found = true;
                    y_plus_chosen = true;
                    y_minus_chosen = false;
                }
                else if dist < this_phi {
                    phis[pointer] = dist; 
                    y_dir_found = true;
                    y_plus_chosen = true;
                    y_minus_chosen = false;
                }
        }
        if y_dir_found { pointer += 1; }

        let mut z_minus_chosen = false;
        let mut z_plus_chosen = false;
        // Check for z-direction.
        if k > 0 {

            // Found a known point from the -z direction.
            if let Some(Cell::Known(dist, _, sign)) = self.domain.get_data(CellIndex {i: i, j: j, k: k-1}) {
                // Update value to array if its smaller than current phi.
                println!("CellIndex {{i: {}, j: {}, k: {}, dist: {}}}", i, j, k-1, dist);
                assert!(sign == true, "pah");
                if dist < this_phi {
                    phis[pointer] = dist; 
                    z_dir_found = true;
                    z_minus_chosen = true;
                }
            }
        }

        // Found a known point from the +z direction.
        if let Some(Cell::Known(dist, _, sign)) = self.domain.get_data(CellIndex {i: i, j: j, k: k+1}) {
                println!("CellIndex {{i: {}, j: {}, k: {}, dist: {}}}", i, j, k+1, dist);
                assert!(sign == true, "pah");
                if dist < phis[pointer] {
                    phis[pointer] = dist; 
                    z_dir_found = true;
                    z_plus_chosen = true;
                    z_minus_chosen = false;
                }
                else if dist < this_phi {
                    phis[pointer] = dist; 
                    z_dir_found = true;
                    z_plus_chosen = true;
                    z_minus_chosen = false;
                }
        }
        //if z_dir_found { pointer += 1; }

        if !x_dir_found && !y_dir_found && !z_dir_found{
            println!("phis[0] == {}, phis[1] == {}, phis[2] == {}, this_phi == {}", phis[0], phis[1], phis[2], this_phi); 

            panic!("Impossible! Solve quadradic all neighbors are not Known");
        }
        let mut id = 0;
        let mut phi_t = 666.0;

        phis.sort_by(|a, b| a.partial_cmp(b).unwrap());

        while id < pointer+1 {
            let a = (id + 1) as f32;
            let mut b = 0.0;
            let mut c = 0.0;

            for i in 0..id+1 {
                b += phis[i]; 
            }
            for i in 0..id+1 {
                c += phis[i].powf(2.0) - 1.0; // - self.domain.grid_length.powf(2.0); 
            }
            //c -= 1.0 / self.domain.grid_length.powf(2.0);

            let det = b.powf(2.0) - a*c;

            if det >= 0.0 {
                phi_t = (b + det.sqrt()) / a;  
                if id < pointer && phi_t > phis[id+1] {
                    id += 1;
                }
                else {

                    return phi_t;
                }
            }
        }

        if phi_t == 666.0 {

            if x_plus_chosen {
                println!("puhs x_plus");
                chosen_neighbors.push(CellIndex {i: i+1, j: j, k: k});
            }
            if x_minus_chosen {
                println!("puhs x_minus");
                chosen_neighbors.push(CellIndex {i: i-1, j: j, k: k});
            }
            if y_plus_chosen {
                println!("puhs y_plus");
                chosen_neighbors.push(CellIndex {i: i, j: j+1, k: k});
            }
            if y_minus_chosen {
                println!("puhs y_minus");
                chosen_neighbors.push(CellIndex {i: i, j: j-1, k: k});
            }
            if z_plus_chosen {
                println!("puhs z_plus");
                chosen_neighbors.push(CellIndex {i: i, j: j, k: k+1});
            }
            if z_minus_chosen {
                println!("puhs z_minus");
                chosen_neighbors.push(CellIndex {i: i, j: j-1, k: k-1});
            }

            self.error_grids.push((CellIndex {i: i, j: j, k: k}, chosen_neighbors)); // (error index, chosen neighbor indices)
            println!("juu on 666");
         }

        phi_t 
        
    }

    pub fn fmm_solve_quadratic(&mut self, i: u32, j: u32, k: u32, this_value: f32, fix_band: bool) -> f32 {

        //let this_cell = self.get_data(i, j, k).unwrap(); 

        // For debugging.
        let mut chosen_neighbors: Vec<CellIndex> = Vec::new();
        // x-direction.
        let mut phis = [0.0, 0.0, 0.0];
        let mut hs: [f32; 3] = [0.0, 0.0, 0.0];
        let mut d0: i32 = 0;
        let mut d1: i32 = 0;
        let mut d2: i32 = 0;
        let mut dist_x_minus = 0.0;
        let mut dist_x_plus  = 0.0;
        let h = 1.0 / self.domain.grid_length;
        if i > 0 { 
            if let Some(cell) = self.domain.get_data(CellIndex {i: i-1, j: j, k: k}) {
                match cell {
                    Cell::Known(dist,_,sign) => {
                        //assert!(sign, "A negative sign!");
                        if sign {
                            dist_x_minus = dist; d0 = -1;
                        }
                        // println!("KNOWN::CellIndex {{ i: {}, j: {}, k: {} }} is a known point. The dist is {}.", i-1, j, k, dist);
                        }
                    _ => { /*println!("CellIndex {{ i: {}, j: {}, k: {} }} is not a known point.", i-1, j, k);*/ } 
                }
            }
        }
        if let Some(cell) = self.domain.get_data(CellIndex {i: i+1, j: j, k: k}) {
            match cell {
                Cell::Known(dist,_,sign) => {
                    //assert!(sign, "A negative sign!");
                    if sign {
                        dist_x_plus = dist;
                        // println!("KNOWN::CellIndex {{ i: {}, j: {}, k: {} }} is a known point. The dist is {}.", i+1, j, k, dist);
                        if d0 == 0 { d0 = 1; }
                        else if dist_x_plus < dist_x_minus { d0 = 1; }
                    }
                }
                _ => { /* println!("CellIndex {{ i: {}, j: {}, k: {} }} is not a known point.", i+1, j, k); */}
            }
        }
        if d0 == -1  {
            phis[0] = dist_x_minus;  
            chosen_neighbors.push(CellIndex {i: i-1, j: j, k: k});
            hs[0] = h;
        }
        if d0 == 1 {
            phis[0] = dist_x_plus;  
            chosen_neighbors.push(CellIndex {i: i+1, j: j, k: k});
            hs[0] = h;
        }

        // y-direction.
        let mut dist_y_minus = 0.0;
        let mut dist_y_plus  = 0.0;

        if j > 0 {
            if let Some(cell) = self.domain.get_data(CellIndex {i: i, j: j-1, k: k}) {
                match cell {
                    Cell::Known(dist,_,sign) => {
                    //assert!(sign, "A negative sign!");
                        if sign {
                            dist_y_minus = dist; d1 = -1;
                            // println!("KNOWN::CellIndex {{ i: {}, j: {}, k: {} }} is a known point. The dist is {}.", i, j-1, k, dist);
                            }
                        }
                    _ => { /* println!("CellIndex {{ i: {}, j: {}, k: {} }} is not a known point.", i, j-1, k); */}
                }
            }
        }
        if let Some(cell) = self.domain.get_data(CellIndex {i: i, j: j+1, k: k}) {
            match cell {
                Cell::Known(dist,_,sign) => {
                    //assert!(sign, "A negative sign!");
                    if sign {
                        dist_y_plus = dist;
                        // println!("KNOWN::CellIndex {{ i: {}, j: {}, k: {} }} is a known point. The dist is {}.", i, j+1, k, dist);
                        if d1 == 0 { d1 = 1; }
                        else if dist_y_plus < dist_y_minus { d1 = 1; }
                    }
                }
                _ => { /* println!("CellIndex {{ i: {}, j: {}, k: {} }} is not a known point.", i, j+1, k); */}
            }
        }
        if d1 == -1  {
            phis[1] = dist_y_minus;  
            chosen_neighbors.push(CellIndex {i: i, j: j-1, k: k});
            hs[1] = h;
        }
        if d1 == 1 {
            phis[1] = dist_y_plus;  
            chosen_neighbors.push(CellIndex {i: i, j: j+1, k: k});
            hs[1] = h;
        }

        // z-direction.
        d2 = 0;
        let mut dist_z_minus = 0.0;
        let mut dist_z_plus  = 0.0;
        if k > 0 {
            if let Some(cell) = self.domain.get_data(CellIndex {i: i, j: j, k: k-1}) {
                match cell {
                    Cell::Known(dist,_,sign) => {
                    //assert!(sign, "A negative sign!");
                        if sign {
                            dist_z_minus = dist; d2 = -1;
                            //println!("KNOWN::CellIndex {{ i: {}, j: {}, k: {} }} is a known point. The dist is {}.", i, j, k-1, dist);
                        }
                    }
                    _ => { /* println!("CellIndex {{ i: {}, j: {}, k: {} }} is not a known point.", i, j, k-1); */ }
                }
            }
        }
        if let Some(cell) = self.domain.get_data(CellIndex {i: i, j: j, k: k+1}) {
            match cell {
                Cell::Known(dist,_,sign) => {
                    //assert!(sign, "A negative sign!");
                    if sign {
                        dist_z_plus = dist;
                        //println!("KNOWN::CellIndex {{ i: {}, j: {}, k: {} }} is a known point. The dist is {}.", i, j, k+1, dist);
                        if d2 == 0 { d2 = 1; }
                        else if dist_z_plus < dist_z_minus { d2 = 1; }
                    }
                }
                _ => { /* println!("CellIndex {{ i: {}, j: {}, k: {} }} is not a known point.", i, j, k+1); */}
            }
        }
        if d2 == -1  {
            phis[2] = dist_z_minus;  
            chosen_neighbors.push(CellIndex {i: i, j: j, k: k-1});
            hs[2] = h;
        }
        if d2 == 1 {
            phis[2] = dist_z_plus;  
            chosen_neighbors.push(CellIndex {i: i, j: j, k: k+1});
            hs[2] = h;
        }

        if d0 == 0 && d1 == 0 && d2 == 0 {
            panic!("Impossible! Solve quadradic all neighbors are not Known");
        }

        // println!("phis :: ({}, {}, {})", phis[0], phis[1], phis[2]);
        // println!("hs :: ({}, {}, {})", hs[0], hs[1], hs[2]);

        let mut a = 0.0;
        let mut b = 0.0;
        let mut c = 0.0;

        for i in 0..3 {
            a += hs[i].powf(2.0);
            //println!("i.powf(2.0) == {}", i.powf(2.0));
        }
        for i in 0..3 {
            b += hs[i].powf(2.0)*phis[i]; 
        }
        b *= -2.0;
        //println!("b == {}", b);
        
        for i in 0..3 {
            c += hs[i].powf(2.0) * phis[i].powf(2.0); // h.powf(2.0); // 1.0;
        }

        //c -= self.domain.grid_length.powf(2.0); // hs[0].powf(2.0);
        //c -= (1.0/0.1 as f32).powf(2.0); // 1.0; // Speed function 1/(f_i,j,k)^2
        c -= 1.0; // Speed function 1/(f_i,j,k)^2
        //println!("c == {}", c);

        //let mut result = Cell::Band(0.0, Vector3::<f32>::new(0.0, 0.0, 0.0));
        let mut final_distance = 777.0;

        let discriminant = b.powf(2.0) - (4.0*a*c);
        // println!("discriminant == {}", discriminant);
        // println!("2*a == {}", 2.0*a);

        if discriminant >= 0.0 {
            let t_phi = (-1.0*b + discriminant.sqrt()) / (2.0*a); 
            //println!("t_phi == {}", t_phi);
            if phis[0] < t_phi && phis[1] < t_phi && phis[2] < t_phi {
                final_distance = t_phi; 
            }
            else {
                if fix_band {
                    return t_phi;
                }
                else {
                    self.error_grids.push((CellIndex{i: i, j: j, k:k}, chosen_neighbors));
                    println!("*********************************");
                    println!("discriminant == {}", discriminant);
                    println!("this_phi == {}", this_value);
                    println!("t_phi == {}", t_phi);
                    println!("hs[0] == {}", hs[0]);
                    println!("hs[1] == {}", hs[1]);
                    println!("hs[2] == {}", hs[2]);
                    println!("phis[0] == {}", phis[0]);
                    println!("phis[1] == {}", phis[1]);
                    println!("phis[2] == {}", phis[2]);
                }
            }
        }
        else {
            self.error_grids.push((CellIndex{i: i, j: j, k:k}, chosen_neighbors));
            println!("++++++++++++++++++++++++");
            println!("this_phi == {}", this_value);
            println!("hs[0] == {}", hs[0]);
            println!("hs[1] == {}", hs[1]);
            println!("hs[2] == {}", hs[2]);
            println!("phis[0] == {}", phis[0]);
            println!("phis[1] == {}", phis[1]);
            println!("phis[2] == {}", phis[2]);
        }
        // let result = Cell::Band(final_distance, false);
        // self.set_data(i, j, k, result);
        // result
        final_distance
    }

    pub fn fmm_march_narrow_band(&mut self) {
        println!("fmm_march_narrow_band. heap.len() ==  {})", self.heap.len());
        while !self.heap.is_empty() {
            if let Some(Reverse((x,u))) = self.heap.pop() { //{ (x.into_inner(), u) },
                let phi = x.into_inner();
                // println!("Popped ({}, {})", phi, u);
                // let (i,j,k) = self.array_index_to_ijk(u).unwrap();  // CHECK!!!
                if let Some(Cell::Band(p, h, is_plus)) = self.domain.get_data_index(u) {

                    // The cell point is already known.
                    //if let Cell::Known(_,_,_) = self.domain.get_data_index(h).unwrap() {
                    //    continue;
                    //}
                    
                    self.domain.set_data_index(u, Cell::Known(phi, Vector3::<f32>::new(0.0, 0.0, 0.0), is_plus));
                    let cell_index = self.domain.array_index_to_cell_index(u).unwrap();
                    self.fmm_update_neighbors(cell_index.i, cell_index.j, cell_index.k);
                }
                //else {
                //    //println!("fmm_march_narrow_band :: A non band cell in heap.");
                //    match self.domain.get_data(i, j, k) {
                //        Some(Cell::Known(_, _, _)) => {  } 
                //        Some(Cell::Band(_, _, _)) => { } 
                //        Some(Cell::Far(_)) => { }  
                //        None => { println!("A None in heap."); } 
                //    }
                //    // panic!("fmm_march_narrow_band :: A non band cell in heap.");
                //}
            }
            else {
                panic!("fmm_march_narrow_band :: an error occurred while heap.pop())");
            }
        }
    }

    /// [a1..a2] -> [b1..b2]. s value to scale.
    fn map_range(a1: f32, a2: f32, b1: f32, b2: f32, s: f32) -> f32 {
        b1 + (s - a1) * (b2 - b1) / (a2 - a1)
    }

    pub fn fmm_data_to_f32(&mut self) -> Vec<f32> {

        for cell in self.domain.data.iter() {
            match cell {
                Cell::Known(dist, _, sign) => {
                    let sign = if *sign { 1.0 } else { -1.0 };
                    let distance = *dist * sign;
                    if distance < self.min_value {
                        //println!("Changin min value {} -> {}", self.min_value, distance);
                        self.min_value = distance;
                    }
                    if distance > self.max_value {
                        //println!("Changin max value {} -> {}", self.max_value, distance);
                        self.max_value = distance;
                    }
                }
                _ => { }
            }
        }

        println!("MIN VALUE :: {}", self.min_value);
        println!("MAX VALUE :: {}", self.max_value);

        let mut result: Vec<f32> = Vec::new();
        for i in 0..self.domain.data.len() {
            if let Cell::Known(dist, _, is_plus) = self.domain.data[i] {
                let sign = if is_plus { 1.0 } else { -1.0 };
                result.push(
                    // FmmDomain::map_range(
                    //     -(self.domain.dimension.0.end as f32 * self.domain.grid_length),
                    //     self.domain.dimension.0.end as f32 * self.domain.grid_length,
                    //     0.0,
                    //     1.0,
                    //     dist * sign)
                    FmmDomain::map_range(
                        self.min_value,
                        self.max_value,
                        0.0,
                        1.0,
                        dist * sign)
                );
                // println!("dist :: {}",dist);
                // println!("scaled::dist {}",FmmDomain::map_range(0.0, self.dimension.0.end as f32 * self.grid_length, 0.0, 1.0, dist));
                result.push(0.0);
                result.push(0.0);
                result.push(0.0);
            }
            else { /* panic!("fmm_data_to_vec4 :: Not all cells are known!"); */ 
                result.push(0.0);
                result.push(0.0);
                result.push(0.0);
                result.push(0.0);
            }
        }
        println!("map_range_for_zero == {}", FmmDomain::map_range(
                        self.min_value,
                        self.max_value,
                        0.0,
                        1.0,
                        0.0));
        result
    }

    pub fn create_triangle_aabb(&self) -> Vec<f32> {
        let mut result: Vec<f32> = Vec::new();
        for i in 0..self.aabbs.len() {
            result.extend(&self.aabbs[i].to_lines());
        }
        result
    }
    
    pub fn heap_test() {
        let mut heap = BinaryHeap::new(); 
        let joo = Reverse((NotNan::new(1.0).unwrap(), 66));
        let ei = Reverse((NotNan::new(1.0).unwrap(), 55));
        let ehheh = Reverse((NotNan::new(1.1).unwrap(), 11));
        //if heap == joo { println!(""); }
        //println!("{}", joo);
        heap.push(joo);
        heap.push(ei);
        heap.push(ehheh);
    
        let mut lets_continue = true;
        while lets_continue {
            let result = match heap.pop() {
                Some(Reverse((x,u))) => { (x.into_inner(), u) },
                _ => {
                    println!("Nyt onpi heappi tyhjä!");
                    lets_continue = false;
                    (0.0, 666)
                }
            };
            println!("HEAP TEST :: {}, {}", result.0, result.1);
        }
    }
}

