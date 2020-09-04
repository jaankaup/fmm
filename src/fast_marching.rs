//////use cgmath::structure::MetricSpace;
//////fn d_minus_x() 
////use std::collections::BinaryHeap;
////use std::cmp::Reverse;
////use cgmath::{prelude::*,Vector3, Vector4};
////use ordered_float::NotNan;
//////use misc::VertexType;
////use crate::misc::VertexType;
////use crate::bvh::{Triangle, BBox, Plane}; 
////
////type MinNonNan = Reverse<NotNan<f32>>;
////
////#[derive(Clone, Copy)]
////pub enum Cell {
////    Known(f32),
////    Band(f32),
////    Far(),
////}
////
////
//////   Initial state
//////  
//////   I :: Surface ( b(x) == 0 )
//////   + :: Ghost point (boundary condition b(x))
//////   F :: Far
//////   K :: Known
//////  
//////                                    w(x) < 0                                           w(x) > 0
//////  
//////    5.5  +---------+---------+---------+---------+---------+---------I---------+---------+---------+---------+--
//////         |         |         |         |         |         |       II|         |         |         |         |       
//////         |         |         |         |         |         |      I  |         |         |         |         |       
//////    5.0  |    F    |    F    |    F    |    F    |    F    |   F I   |    K    |    F    |    F    |    F    |       
//////         |         |         |         |         |         |   II    |         |         |         |         |       
//////         |         |         |         |         |         | II      |         |         |         |         |       
//////    4.5  +---------+---------+---------+---------+---------+I--------+---------+---------+---------+---------+--
//////         |         |         |         |         |         I         |         |         |         |         |       
//////         |         |         |         |         |        I|         |         |         |         |         |       
//////    4.0  |    F    |    F    |    F    |    F    |    F  I |    K    |    F    |    F    |    F    |    F    |       
//////         |         |         |         |         |       I |         |         |         |         |         |       
//////         |         |         |         |         |        I|         |         |         |         |         |       
//////    3.5  +---------+---------+---------+---------+---------I---------+---------+---------+---------+---------+--
//////         |         |         |         |         |         I         |         |         |         |         |       
//////         |         |         |         |         |         |I        |         |         |         |         |       
//////    3.0  |    F    |    F    |    F    |    F    |    F    | I  K    |    F    |    F    |    F    |    F    |       
//////         |         |         |         |         |         |  I      |         |         |         |         |       
//////         |         |         |         |         |         |  I      |         |         |         |         |       
//////    2.5  +---------+---------+---------+---------+---------+--I------+---------+---------+---------+---------+--
//////         |         |         |         |         |         | I       |         |         |         |         |       
//////         |         |         |         |         |         |I        |         |         |         |         |       
//////    2.0  |    F    |    F    |    F    |    F    |    F    I    K    |    F    |    F    |    F    |    F    |       
//////         |         |         |         |         |         I         |         |         |         |         |       
//////         |         |         |         |         |        I|         |         |         |         |         |       
//////    1.5  +---------+---------+---------+---------+-------I-+---------+---------+---------+---------+---------+--
//////         |         |         |         |         |      I  |         |         |         |         |         |       
//////         |         |         |         |         |     I   |         |         |         |         |         |       
//////    1.0  |    F    |    F    |    F    |    F    |  F I    |    K    |    F    |    F    |    F    |    F    |       
//////         |         |         |         |         |    I    |         |         |         |         |         |       
//////         |         |         |         |         |   I     |         |         |         |         |         |       
//////    0.5  +---------+---------+---------+---------+---I-----+---------+---------+---------+---------+---------+--
//////  
//////        0.5  1.0  1.5  2.0  2.5  3.0  3.5  4.0  4.5  5.0  5.5  6.0  6.5  7.0  7.5  8.0  8.5  9.0  9.5  10.0  10.5
//////  
//////
////pub struct InterfaceT {
////    pub points: Vec<f32>,
////}
////
////// Computational domanin for fast marching. Includes both cells and boundary points. 
////pub struct DomainE {
////    pub min_x: u32, // Minimum x-coordinate for cells. 
////    pub max_x: u32, // Minimum y-coordinate for cells. 
////    pub min_y: u32, // Minimum z-coordinate for cells. 
////    pub max_y: u32, // Maximum x-coordinate for cells. 
////    pub min_z: u32, // Maximum y-coordinate for cells. 
////    pub max_z: u32, // Maximum z-coordinate for cells. 
////    pub scale_factor: f32,
////    pub cells: Vec<Cell>,
////    pub boundary_points: InterfaceT,
////    pub min_boundary_value: f32, // for debuging. Keep track of the minimun value of b(x).
////    pub max_boundary_value: f32, // for debuging. Keep track of the maximum value of b(x).
////    pub heap: BinaryHeap<MinNonNan>,
////}
////
////impl DomainE {
////    pub fn new(min_x: u32, max_x: u32, min_y: u32, max_y: u32, min_z: u32, max_z: u32, scale_factor: f32) -> Self {
////
////        assert!(min_x < max_x, "min_x :: {} < max_x :: {}", min_x, max_x);
////        assert!(min_y < max_y, "min_y :: {} < max_y :: {}", min_y, max_y);
////        assert!(min_z < max_z, "min_z :: {} < max_z :: {}", min_z, max_x);
////
////        // Delta for x positions.
////        let delta_x = max_x - min_x;
////
////        // Delta for y positions.
////        let delta_y = max_y - min_y;
////
////        // Delta for z positions.
////        let delta_z = max_z - min_z;
////
////        // Initialize cell points with far values.
////        let cells: Vec<Cell> = vec![Cell::Far() ; ((delta_x+1) * (delta_y+1) * (delta_z+1)) as usize];
////
////        // Initialize boundary points with zeroes.
////        let boundary_points: Vec<f32> = vec![0.0 ; ((delta_x+2) * (delta_y+2) * (delta_z+2)) as usize];
////
////        let heap: BinaryHeap<MinNonNan> = BinaryHeap::new();
////
////        Self {
////            min_x: min_x,
////            max_x: max_x,
////            min_y: min_y,
////            max_y: max_y,
////            min_z: min_z,
////            max_z: max_z,
////            scale_factor: scale_factor,
////            cells: cells,
////            boundary_points: InterfaceT { points: boundary_points, },
////            min_boundary_value: 0.0,
////            max_boundary_value: 0.0,
////            heap: heap,
////        }
////    }
////
////    fn set_cell(&mut self, i: u32, j: u32, k: u32, cell: &Cell) {
////        self.check_cell_index(i, j, k, true); // TODO: this should be checked when range is generated. TODO: remove in the future.
////        let index = self.get_cell_index(i, j, k).unwrap(); // as usize;
////        self.cells[index] = *cell;
////    }
////
////    fn get_x_range(&self) -> std::ops::Range<u32> {
////        let width = self.max_x - self.min_x + 1;
////        std::ops::Range {start: 0, end: width }
////    }
////
////    fn get_y_range(&self) -> std::ops::Range<u32> {
////        let height = self.max_y - self.min_y + 1;
////        std::ops::Range {start: 0, end: height }
////    }
////
////    fn get_z_range(&self) -> std::ops::Range<u32> {
////        let depth = self.max_z - self.min_z + 1;
////        std::ops::Range {start: 0, end: depth }
////    }
////
////    fn get_boundary_x_range(&self) -> std::ops::Range<u32> {
////        let width = self.max_x - self.min_x + 2;
////        std::ops::Range {start: 0, end: width }
////    }
////
////    fn get_boundary_y_range(&self) -> std::ops::Range<u32> {
////        let height = self.max_y - self.min_y + 2;
////        std::ops::Range {start: 0, end: height }
////    }
////
////    fn get_boundary_z_range(&self) -> std::ops::Range<u32> {
////        let depth = self.max_z - self.min_z + 2;
////        std::ops::Range {start: 0, end: depth }
////    }
////
////    // 
////    pub fn get_index_ranges_aabb(&self, aabb: &BBox) -> (std::ops::Range<u32>,  std::ops::Range<u32>, std::ops::Range<u32>) {
////
////        let (x_min, y_min, z_max) = self.get_cell_indices_wc(&aabb.min).unwrap();// -> Option<(u32, u32, u32)> {
////        let (x_max, y_max, z_min) = self.get_cell_indices_wc(&aabb.max).unwrap();// -> Option<(u32, u32, u32)> {
////        // println!("Range_x :: [{}, {}]", x_min, x_max);
////        // println!("Range_y :: [{}, {}]", y_min, y_max);
////        // println!("Range_z :: [{}, {}]", z_min, z_max);
////        (std::ops::Range {start: x_min, end: x_max},
////         std::ops::Range {start: y_min, end: y_max},
////         std::ops::Range {start: z_min, end: z_max})
////    }
////
////
////    fn check_cell_index(&self, i: u32, j: u32, k: u32, assert: bool) -> bool {
////        let x_ok = i >= self.min_x || i <= self.max_x;
////        let y_ok = j >= self.min_y || j <= self.max_y;
////        let z_ok = k >= self.min_z || k <= self.max_z;
////
////        if assert {
////            assert!(i >= self.min_x || i <= self.max_x, "check_cell_index:: i :: {} not in range [{}, {}]", i, self.min_x, self.max_x);
////            assert!(j >= self.min_y || j <= self.max_y, "check_cell_index:: j :: {} not in range [{}, {}]", j, self.min_y, self.max_y);
////            assert!(k >= self.min_z || k <= self.max_z, "check_cell_index:: k :: {} not in range [{}, {}]", k, self.min_z, self.max_z);
////        }
////
////        x_ok && y_ok && z_ok
////    }
////
////    fn check_boundary_index(&self, i: u32, j: u32, k: u32) {
////        assert!(i >= self.min_x || i <= self.max_x + 1, "check_cell_index:: i :: {} not in range [{}, {}]", i, self.min_x, self.max_x);
////        assert!(j >= self.min_y || j <= self.max_y + 1, "check_cell_index:: j :: {} not in range [{}, {}]", j, self.min_y, self.max_y);
////        assert!(k >= self.min_z || k <= self.max_z + 1, "check_cell_index:: k :: {} not in range [{}, {}]", k, self.min_z, self.max_z);
////    }
////
////    fn get_cell_coordinates(&self, i: u32, j: u32, k: u32) -> Vector4<f32> {
////        self.check_cell_index(i, j, k, true); // TODO: this should be checked when range is generated. TODO: remove in the future.
////        let result: Vector4<f32> = Vector4::new(
////            (self.min_x + i) as f32 * self.scale_factor,
////            (self.min_y + j) as f32 * self.scale_factor,      
////            (self.min_z as i32 - k as i32) as f32 * self.scale_factor,
////            1.0
////        );
////
////        result
////    }
////
////    /// Get the nearest cell indices from world coordinates.
////    fn get_cell_indices_wc(&self, w_pos: &Vector3<f32>) -> Option<(u32, u32, u32)> {
////        let x_index = (w_pos.x / self.scale_factor).floor() as u32; 
////        let y_index = (w_pos.y / self.scale_factor).floor() as u32; 
////        let z_index = (-w_pos.z / self.scale_factor).floor() as u32; 
////        if x_index == 0 || y_index == 0 || z_index == 0 {
////            return None
////        }
////        if !self.check_cell_index(x_index - 1, y_index - 1, z_index - 1, false) { None } 
////        else { Some((x_index - 1, y_index - 1, z_index - 1)) }
////    }
////
////    fn get_boundary_coordinates(&self, i: u32, j: u32, k: u32) -> Vector4<f32> {
////        self.check_boundary_index(i, j, k); // TODO: this should be checked when range is generated. TODO: remove in the future.
////        let result: Vector4<f32> = Vector4::new(
////            ((self.min_x + i) as f32 - 0.5) * self.scale_factor,
////            ((self.min_y + j) as f32 - 0.5) * self.scale_factor,
////            ((self.min_z as i32 - k as i32) as f32 + 0.5) * self.scale_factor,
////            1.0
////        );
////
////        result
////    }
////
////    /// Get cell index from point (i, j, k).
////    fn get_cell_index(&self, i: u32, j: u32, k: u32) -> Result<usize, String> {
////        self.check_cell_index(i, j, k, true); // TODO: this should be checked when range is generated. TODO: remove in the future.
////        let x_range = self.get_x_range(); 
////        let y_range = self.get_y_range(); 
////        let index = i + j * x_range.end + k * x_range.end * y_range.end;
////        if index >= self.cells.len() as u32 {
////             Err(format!("DomainE::get_cell_index({}, {}, {}). Index out of bounds index {} > cells.len() {}.", i, j ,k, index, self.cells.len()))
////        }
////        else {
////            Ok(index as usize)
////        }
////    }
////
////    /// Get boundary index from point (i, j, k).
////    fn get_boundary_index(&self, i: u32, j: u32, k: u32) -> Result<usize, String> {
////        self.check_boundary_index(i, j, k); // TODO: this should be checked when range is generated. TODO: remove in the future.
////        let x_range = self.get_boundary_x_range(); 
////        let y_range = self.get_boundary_y_range(); 
////        let index = i + j * x_range.end + k * x_range.end * y_range.end;
////        if index >= self.boundary_points.points.len() as u32 {
////             Err(format!("DomainE::get_bondary_index({}, {}, {}). Index out of bounds index {} > boundary_points.points.len() {}.",
////             i,
////             j,
////             k,
////             index,
////             self.boundary_points.points.len()))
////        }
////        else {
////            Ok(index as usize)
////        }
////    }
////
////    /// Get cell from index (i,j,k)
////    fn get_cell(&self, i: u32, j: u32, k: u32) -> Cell {
////        let index = self.get_cell_index(i, j, k).unwrap(); 
////        self.cells[index]
////    }
////
////    /// Get boundary value from index (i,j,k)
////    fn get_boundary_value(&self, i: u32, j: u32, k: u32) -> f32 {
////        let index = self.get_boundary_index(i, j, k).unwrap(); 
////        self.boundary_points.points[index]
////    }
////
////    // Create Vec<f32> from cell values in vvvv cccc format. vvvv :: pos, cccc :: color.
////    // Far cell point = red.  
////    // Known cell point = greeb.  
////    // Band cell point = blud.  
////    pub fn cells_to_vec(&self) -> Vec<f32> {
////
////        let mut result: Vec<f32> = Vec::new(); 
////
////        for k in self.get_z_range() {
////        for j in self.get_y_range() {
////        for i in self.get_x_range() {
////
////            let value = self.get_cell(i, j, k);
////
////            //let cell_coordinates = self.get_cell_coordinates(i, j, k);
////            let coordinates = self.get_cell_coordinates(i, j, k);
////            result.push(coordinates.x);
////            result.push(coordinates.y);
////            result.push(coordinates.z);
////            result.push(coordinates.w);
////            
////            // The color of the cell. Far points are red, Known green and Band cell blue.
////            result.push(match value {Cell::Far()    => 1.0, _ => 0.0 }); // red
////            result.push(match value {Cell::Known(_) => 1.0, _ => 0.0 }); // green
////            result.push(match value {Cell::Band(_)  => 1.0, _ => 0.0 }); // blue
////            result.push(1.0 as f32); // alpha
////        }}};
////        result
////    }
////
////    /// Create Vec<f32> from boundary values in vvvc format. vvv: pos, c :: [0.0, 1.0] a scaled/normalized boundary point value.
////    pub fn boundary_points_to_vec(&self) -> Vec<f32> {
////        println!("Converting boundary points to vec<f32>.");
////
////        let mut result: Vec<f32> = Vec::new(); 
////
////        for k in self.get_boundary_z_range() {
////        for j in self.get_boundary_y_range() {
////        for i in self.get_boundary_x_range() {
////
////            let value = self.get_boundary_value(i,j,k);
////
////            let boundary_coordinates = self.get_boundary_coordinates(i, j, k);
////            let normalized_value = clamp(1.0 / (value.abs() + 1.0), 0.1, 1.0).powf(8.0);
////
////            result.push(boundary_coordinates.x);
////            result.push(boundary_coordinates.y);
////            result.push(boundary_coordinates.z);
////            result.push(normalized_value);
////        }}};
////        result
////    }
////
////    pub fn boundary_grid_to_vec(&self) -> Vec<f32> {
////
////        println!("Converting boundary grid lines to vec<f32>.");
////        let boundary_width = self.max_x - self.min_x + 2;
////        let boundary_height = self.max_y - self.min_y + 2;
////        let boundary_depth = self.max_z - self.min_z + 2;
////
////        let mut result: Vec<f32> = Vec::new(); 
////
////        // Horizontal lines.
////        for k in 0..boundary_depth  {
////        for j in 0..boundary_height {
////            result.push(((self.min_x) as f32 - 0.5) * self.scale_factor);
////            result.push(((self.min_y + j) as f32 - 0.5) * self.scale_factor);
////            result.push(((self.min_z as i32 - k as i32) as f32 + 0.5) * self.scale_factor);
////            result.push(0.1);
////            result.push(((self.max_x + 1) as f32- 0.5) * self.scale_factor);
////            result.push(((self.min_y + j) as f32 - 0.5) * self.scale_factor);
////            result.push(((self.min_z as i32 - k as i32) as f32 + 0.5) * self.scale_factor);
////            result.push(0.1);
////        }};
////
////        // Vertical lines.
////        for k in 0..boundary_depth  {
////        for i in 0..boundary_width {
////            result.push(((self.min_x + i) as f32 - 0.5) * self.scale_factor);
////            result.push(((self.min_y) as f32 - 0.5) * self.scale_factor);
////            result.push(((self.min_z as i32 - k as i32) as f32 + 0.5) * self.scale_factor);
////            result.push(0.1);
////            result.push(((self.min_x + i) as f32- 0.5) * self.scale_factor);
////            result.push(((self.max_y + 1) as f32 - 0.5) * self.scale_factor);
////            result.push(((self.min_z as i32 - k as i32) as f32 + 0.5) * self.scale_factor);
////            result.push(0.1);
////        }};
////
////        // Depth lines.
////        for j in 0..boundary_height  {
////        for i in 0..boundary_width {
////            result.push(((self.min_x + i) as f32 - 0.5) * self.scale_factor);
////            result.push(((self.min_y + j) as f32 - 0.5) * self.scale_factor);
////            result.push(((self.min_z) as f32 + 0.5) * self.scale_factor);
////            result.push(0.1);
////            result.push(((self.min_x + i) as f32- 0.5) * self.scale_factor);
////            result.push(((self.min_y + j) as f32 - 0.5) * self.scale_factor);
////            result.push(((-(self.max_z as i32) + 1 as i32) as f32 + 0.5) * self.scale_factor);
////            result.push(0.1);
////        }};
////
////        result
////    }
////
////    /// Initialize values for boundary points.
////    pub fn initialize_boundary<F: Fn(f32, f32, f32) -> f32>(&mut self, b: F) {
////
////        let mut min_value: f32 = 0.0;
////        let mut max_value: f32 = 0.0;
////
////        for k in self.get_boundary_x_range() {
////        for j in self.get_boundary_y_range() {
////        for i in self.get_boundary_z_range() {
////
////            let index  = self.get_boundary_index(i, j, k).unwrap();
////            // let index = (i + j * boundary_width + k * boundary_width * boundary_height) as usize;
////            let boundary_coordinates = self.get_boundary_coordinates(i, j, k);
////            let value = b(
////               boundary_coordinates.x,
////               boundary_coordinates.y,
////               boundary_coordinates.z
////            );
////            if value < min_value { min_value = value; }
////            if value > max_value { max_value = value; }
////            self.boundary_points.points[index] = value; 
////        }}};
////
////        self.min_boundary_value = min_value;
////        self.max_boundary_value = max_value;
////    }
////
////    // Neighbor points and the nearest neighbor as Vec.
////    pub fn get_neighbor_grid_points(&self, p: &Vector3<f32>) -> Vec<f32> {
////        let x_minus = (p.x / self.scale_factor).floor() * self.scale_factor;
////        let y_minus = (p.y / self.scale_factor).floor() * self.scale_factor;
////        let z_minus = (p.z / self.scale_factor).floor() * self.scale_factor;
////        let x_plus  = (p.x / self.scale_factor).ceil()  * self.scale_factor;
////        let y_plus  = (p.y / self.scale_factor).ceil()  * self.scale_factor;
////        let z_plus  = (p.z / self.scale_factor).ceil()  * self.scale_factor;
////
////        let p0 = Vector3::<f32>::new(x_minus, y_minus, z_minus);
////        let p1 = Vector3::<f32>::new(x_minus, y_plus, z_minus);
////        let p2 = Vector3::<f32>::new(x_plus,  y_plus, z_minus);
////        let p3 = Vector3::<f32>::new(x_minus, y_plus, z_minus);
////
////        let p4 = Vector3::<f32>::new(x_minus, y_minus, z_plus);
////        let p5 = Vector3::<f32>::new(x_minus, y_plus , z_plus);
////        let p6 = Vector3::<f32>::new(x_plus , y_plus , z_plus);
////        let p7 = Vector3::<f32>::new(x_plus , y_minus, z_plus);
////
////        let mut result: Vec<Vector3<f32>> = Vec::new(); 
////        result.push(p0);
////        result.push(p1);
////        result.push(p2);
////        result.push(p3);
////        result.push(p4);
////        result.push(p5);
////        result.push(p6);
////        result.push(p7);
////
////        let mut min_distance: f32 = p0.distance(*p);
////        let mut closest_points: Vec<Vector3<f32>> = Vec::new();
////        let mut line_data: Vec<f32> = Vec::new();
////
////        for neighbor in result.iter() {
////            let temp_dist = p.distance(*neighbor);  
////            if temp_dist < self.scale_factor {
////                closest_points.push(neighbor.clone());
////            }
////            // if temp_dist < min_distance { 
////            //     min_distance = temp_dist;
////            //     closest_point = neighbor.clone();
////            // }
////        }
////
////        for canditate in closest_points.iter() {
////            line_data.push(p.x);
////            line_data.push(p.y);
////            line_data.push(p.z);
////            line_data.push(1.0);
////
////            line_data.push(canditate.x);
////            line_data.push(canditate.y);
////            line_data.push(canditate.z);
////            line_data.push(1.0);
////        }
////
////        //let (i, j, k) = self.get_cell_indices_wc(&closest_point).unwrap();
////        //println!("Index is ({}, {}, {})", i, j, k);
////
////        // line_data.push(p.x);
////        // line_data.push(p.y);
////        // line_data.push(p.z);
////        // line_data.push(1.0);
////
////        // line_data.push(closest_point.x);
////        // line_data.push(closest_point.y);
////        // line_data.push(closest_point.z);
////        // line_data.push(1.0);
////
////        // line_data.push(p.x + 0.001);
////        // line_data.push(p.y + 0.001);
////        // line_data.push(p.z + 0.001);
////        // line_data.push(1.0);
////
////        // line_data.push(closest_point.x);
////        // line_data.push(closest_point.y);
////        // line_data.push(closest_point.z);
////        // line_data.push(1.0);
////
////        // line_data.push(p.x - 0.001);
////        // line_data.push(p.y - 0.001);
////        // line_data.push(p.z - 0.001);
////        // line_data.push(1.0);
////
////        // line_data.push(closest_point.x);
////        // line_data.push(closest_point.y);
////        // line_data.push(closest_point.z);
////        // line_data.push(1.0);
////
////        // line_data.push(p.x + 0.001);
////        // line_data.push(p.y + 0.001);
////        // line_data.push(p.z + 0.001);
////        // line_data.push(1.0);
////
////        // line_data.push(p.x - 0.001);
////        // line_data.push(p.y - 0.001);
////        // line_data.push(p.z - 0.001);
////        // line_data.push(1.0);
////        
////        line_data
////    }
////
////    pub fn initialize_from_triangle_list(&mut self, vertex_list: &Vec<f32>, vt: &VertexType) -> (Vec<f32>, Vec<f32>, Vec<Plane>) {
////
////        let mut aabbs: Vec<BBox> = Vec::new(); 
////        let mut triangles: Vec<Triangle> = Vec::new(); 
////        let mut planes: Vec<Plane> = Vec::new(); 
////
////        match vt {
////
////            VertexType::vvv() => {
////
////                let modulo = vertex_list.len() % 9;
////                if modulo != 0 { panic!("initialize_from_triangle_list. vvv: triangle_list % 9 == {}", modulo) }
////
////                for i in 0..vertex_list.len() / 9 {
////                    let a: Vector3<f32> = Vector3::<f32>::new(vertex_list[i*9],     vertex_list[i*9 + 1], vertex_list[i*9 + 2]);
////                    let b: Vector3<f32> = Vector3::<f32>::new(vertex_list[i*9 + 3], vertex_list[i*9 + 4], vertex_list[i*9 + 5]);
////                    let c: Vector3<f32> = Vector3::<f32>::new(vertex_list[i*9 + 6], vertex_list[i*9 + 7], vertex_list[i*9 + 8]);
////
////                    // Triangle is outside the computational domain.
////
////                    if self.get_cell_indices_wc(&a) == None ||
////                       self.get_cell_indices_wc(&b) == None ||
////                       self.get_cell_indices_wc(&c) == None {
////                            println!("Vertex is outside computational domain.");
////                            continue;
////                    }
////                    
////                    let (a1, b1, c1) = self.get_cell_indices_wc(&a).unwrap();
////                    let (a2, b2, c2) = self.get_cell_indices_wc(&b).unwrap();
////                    let (a3, b3, c3) = self.get_cell_indices_wc(&c).unwrap();
////
////                    let min_x_index = a1.min(a2).min(a3); 
////                    let min_y_index = b1.min(b2).min(b3); 
////                    let min_z_index = c1.min(c2).min(c3); 
////
////                    let max_x_index = a1.max(a2).max(a3); 
////                    let max_y_index = b1.max(b2).max(b3); 
////                    let max_z_index = c1.max(c2).max(c3); 
////
////                    let range_x = std::ops::Range {start: min_x_index, end: max_x_index };
////                    let range_y = std::ops::Range {start: min_y_index, end: max_y_index };
////                    let range_z = std::ops::Range {start: min_z_index, end: max_z_index };
////
////                    println!("(a1 :: {}, b1 :: {}, c1 :: {})", a1, b1, c1);
////                    println!("(a2 :: {}, b2 :: {}, c2 :: {})", a2, b2, c2);
////                    println!("(a3 :: {}, b3 :: {}, c3 :: {})", a3, b3, c3);
////
////                    println!("(range_x :: [{} , {}]", min_x_index, max_x_index);
////                    println!("(range_y :: [{} , {}]", min_y_index, max_y_index);
////                    println!("(range_z :: [{} , {}]", min_z_index, max_z_index);
////                    
////                    triangles.push(Triangle {
////                        a: a,
////                        b: b,
////                        c: c,
////                    });
////
////                    let mut aabb = BBox::create_from_triangle(&a, &b, &c);
////                    aabb.expand_to_nearest_grids(self.scale_factor);
////                    aabbs.push(aabb);
////                    //aabbs.push(aabb.clone());
////
////                    let plane = Plane::new(&a, &b, &c); 
////                    planes.push(plane);
////                }
////                
////                // for i in 0..triangles.len() {
////                //     aabb.push(BBox::new(
////                // }
////            }
////            VertexType::vvvv() => {
////                let modulo = (vertex_list.len() + 1) % 4;
////                if modulo != 0 { panic!("initialize_from_triangle_list. vvvv: triangle_list % 4 == {}", modulo) }
////                unimplemented!()
////            }
////            VertexType::vvvnnn() => {
////                let modulo = (vertex_list.len() + 1) % 6;
////                if modulo != 0 { panic!("initialize_from_triangle_list. vvvnnn: triangle_list % 6 == {}", modulo) }
////                unimplemented!()
////            }
////            VertexType::vvvvnnnn() => {
////                let modulo = (vertex_list.len() + 1) % 8;
////                if modulo != 0 { panic!("initialize_from_triangle_list. vvvnnn: triangle_list % 8 == {}", modulo) }
////                unimplemented!()
////            }
////        }
////        let mut tr: Vec<f32> = Vec::new();
////        let mut bb: Vec<f32> = Vec::new();
////        for i in 0..triangles.len() {
////            tr.extend(&triangles[i].to_f32_vec(&VertexType::vvvvnnnn()));
////        }
////        for i in 0..aabbs.len() {
////            bb.extend(&aabbs[i].to_lines());
////            let (mut q,mut w,mut e) = self.get_index_ranges_aabb(&aabbs[i]);
////            // println!("pahhaaa");
////            // println!("{} {} {}", q.start, w.start, e.start);
////            // println!("{} {} {}", q.end, w.end, e.end);
////            for x in q.start..q.end {
////                for y in w.start..w.end {
////                    for z in e.start..e.end {
////                        // println!("pyhyy");
////                        // let coords = self.get_cell_coordinates(x, y, z); //i: u32, j: u32, k: u32)
////                        // println!("({}, {}, {})", coords.x, coords.y, coords.z);
////                        // let (i, j, k) = self.get_cell_indices_wc(&Vector3::<f32>::new(coords.x, coords.y, coords.z)).unwrap();
////                        // println!("({}, {}, {})", i, j, k);
////                        // self.set_cell(i, j, k, &Cell::Known(1.0));
////                    }
////                }
////            }
////            let (q,w,e) = self.get_index_ranges_aabb(&aabbs[i]);
////        }
////        (tr, bb, planes)
////        // let triangles =   
////    }
////} // DomainE impl
////
////// pub fn initialize_boundary_points<F: FnOnce(f32, f32, f32) -> f32>(domain: &mut DomainE, f: F) -> f32 {
//////     let x = 0.5;
//////     let y = 1.5;
//////     let z = 2.5;
//////     f(x,y,z)
////// }
////
////// Insert a value explicitly.
////// fn add_initial_value(domain: DomainE, position: Vector3<u32>, value: Vector3<f32>) {
//////     // assert!(position.x == 0 || position.y == 0 || position.z == 0 || position.x > domain.dim_x || position.y > domain.dim_y || position.z > domain.dim_z,
//////     //     "Position ({}, {}, {}) not in domain range (0, 0, 0) .. ({}, {}, {}).",
//////     //     position.x,
//////     //     position.y,
//////     //     position.z,
//////     //     domain.dim_x,
//////     //     domain.dim_y,
//////     //     domain.dim_z
//////     // );
////// }
////
////// pub fn is_inside_range(value: f32, min: f32, max: f32) -> bool {
//////     assert!(max > min, "is_inside_range({}, {}, {}). min < max :: {} < {}", value, min, max, min, max);
//////     value >= min && value <= max 
////// }
////
////pub fn clamp(value: f32, a: f32, b: f32) -> f32 {
////    let min = if value < a { a } else { value }; 
////    let result = if min > b { b } else { min }; 
////    result
////}
////
////
////// pub fn aabb(a: &Vector4<f32>, b: &Vector4<f32>, c: &Vector4<f32>) {
////// 
//////     let mut min_x = a.x;
//////     let mut min_y = a.y;
//////     let mut min_z = a.z;
//////     let mut max_x = a.x;
//////     let mut max_y = a.y;
//////     let mut max_z = a.z;
////// 
//////    if b.x < min_x { min_x = b.x } 
//////    if b.y < min_y { min_y = b.y } 
//////    if b.z < min_z { min_z = b.z } 
////// 
//////    if c.x < min_x { min_x = c.x } 
//////    if c.y < min_y { min_y = c.y } 
//////    if c.z < min_z { min_z = c.z } 
////// 
////// }
////
//////fn get_cell(i: u32, j: u32, k: u32, domain: &DomainE) -> Result<Cell, String> {
//////    let index = (i + j * boundary_width + k * boundary_width * boundary_height) as usize;
//////
//////}
////
////// Update the neighbors of a KNOWN point:
//////fn update_neighbors(i: u32, j: u32, k: u32, domain: &mut DomainE) {
//////    let cell = domain.
//////    assert!(match cell { Cell::Known(_) => true, _ => false },
//////        "In update_neighbors function: cell must be known.");
//////}
////
//////pub fn initialize_interface(domain: &mut DomainE, initial_values: &Vec<Vector3<f32>>) {
//////
//////    domain.cells = vec![Cell::Far() ; (domain.dim_x * domain.dim_y * domain.dim_z) as usize];
//////
//////    let test_point0: Vector3<f32> = Vector3::new(5.5,10.0,-11.5);
//////
//////    //println!("initialize boundary points :: {}", initialize_boundary_points(|x, y, z| x+y+z));
//////
//////    //for i in 0..domain.dim_x {
//////    //for j in 0..domain.dim_y {
//////    //for k in 0..domain.dim_z {
//////    //    let index = (i + j * domain.dim_y + j * domain.dim_y * k * domain.dim_z) as usize;
//////    //    if is adjancent to domain
//////    //    domain.cells[index] = Cell::Known(55.0);
//////    //}}};
//////}
////
////// Narrow band Chopp: Computing Minimal Surfaces via Level Set Curvature Flow
//////
////// Ainoastaan upwind pisteitä voidaan käyttää laskettaessa narrow band arvoja. Näin varmistetaan
////// oikea viskositeetti ratkaisu (1502.07303.pdf) sivu 8.
