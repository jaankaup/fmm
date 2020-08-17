//fn d_minus_x() 
use std::collections::BinaryHeap;
use std::cmp::Reverse;
use cgmath::{Vector3, Vector4};
use ordered_float::NotNan;

type MinNonNan = Reverse<NotNan<f32>>;

#[derive(Clone, Copy)]
pub enum Cell {
    Known(f32),
    Band(f32),
    Far(),
}


//   Initial state
//  
//   I :: Surface ( b(x) == 0 )
//   + :: Ghost point (boundary condition b(x))
//   F :: Far
//   K :: Known
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
//
pub struct InterfaceT {
    pub points: Vec<f32>,
}

// Computational domanin for fast marching. Includes both cells and boundary points. 
pub struct DomainE {
    pub min_x: u32, // Minimum x-coordinate for cells. 
    pub max_x: u32, // Minimum y-coordinate for cells. 
    pub min_y: u32, // Minimum z-coordinate for cells. 
    pub max_y: u32, // Maximum x-coordinate for cells. 
    pub min_z: u32, // Maximum y-coordinate for cells. 
    pub max_z: u32, // Maximum z-coordinate for cells. 
    pub scale_factor: f32,
    pub cells: Vec<Cell>,
    pub boundary_points: InterfaceT,
    pub min_boundary_value: f32, // for debuging. Keep track of the minimun value of b(x).
    pub max_boundary_value: f32, // for debuging. Keep track of the maximum value of b(x).
    pub heap: BinaryHeap<MinNonNan>,
}

impl DomainE {
    pub fn new(min_x: u32, max_x: u32, min_y: u32, max_y: u32, min_z: u32, max_z: u32, scale_factor: f32) -> Self {

        assert!(min_x < max_x, "min_x :: {} < max_x :: {}", min_x, max_x);
        assert!(min_y < max_y, "min_y :: {} < max_y :: {}", min_y, max_y);
        assert!(min_z < max_z, "min_z :: {} < max_z :: {}", min_z, max_x);

        // Delta for x positions.
        let delta_x = max_x - min_x;

        // Delta for y positions.
        let delta_y = max_y - min_y;

        // Delta for z positions.
        let delta_z = max_z - min_z;

        // Initialize cell points with far values.
        let cells: Vec<Cell> = vec![Cell::Far() ; ((delta_x+1) * (delta_y+1) * (delta_z+1)) as usize];

        // Initialize boundary points with zeroes.
        let boundary_points: Vec<f32> = vec![0.0 ; ((delta_x+2) * (delta_y+2) * (delta_z+2)) as usize];

        let heap: BinaryHeap<MinNonNan> = BinaryHeap::new();

        Self {
            min_x: min_x,
            max_x: max_x,
            min_y: min_y,
            max_y: max_y,
            min_z: min_z,
            max_z: max_z,
            scale_factor: scale_factor,
            cells: cells,
            boundary_points: InterfaceT { points: boundary_points, },
            min_boundary_value: 0.0,
            max_boundary_value: 0.0,
            heap: heap,
        }
    }

    fn get_x_range(&self) -> std::ops::Range<u32> {
        let width = self.max_x - self.min_x + 1;
        std::ops::Range {start: 0, end: width }
    }

    fn get_y_range(&self) -> std::ops::Range<u32> {
        let height = self.max_y - self.min_y + 1;
        std::ops::Range {start: 0, end: height }
    }

    fn get_z_range(&self) -> std::ops::Range<u32> {
        let depth = self.max_z - self.min_z + 1;
        std::ops::Range {start: 0, end: depth }
    }

    fn get_boundary_x_range(&self) -> std::ops::Range<u32> {
        let width = self.max_x - self.min_x + 2;
        std::ops::Range {start: 0, end: width }
    }

    fn get_boundary_y_range(&self) -> std::ops::Range<u32> {
        let height = self.max_y - self.min_y + 2;
        std::ops::Range {start: 0, end: height }
    }

    fn get_boundary_z_range(&self) -> std::ops::Range<u32> {
        let depth = self.max_z - self.min_z + 2;
        std::ops::Range {start: 0, end: depth }
    }

    fn check_cell_index(&self, i: u32, j: u32, k: u32) {
        assert!(i >= self.min_x || i <= self.max_x, "check_cell_index:: i :: {} not in range [{}, {}]", i, self.min_x, self.max_x);
        assert!(j >= self.min_y || j <= self.max_y, "check_cell_index:: j :: {} not in range [{}, {}]", j, self.min_y, self.max_y);
        assert!(k >= self.min_z || k <= self.max_z, "check_cell_index:: k :: {} not in range [{}, {}]", k, self.min_z, self.max_z);
    }

    fn check_boundary_index(&self, i: u32, j: u32, k: u32) {
        assert!(i >= self.min_x || i <= self.max_x + 1, "check_cell_index:: i :: {} not in range [{}, {}]", i, self.min_x, self.max_x);
        assert!(j >= self.min_y || j <= self.max_y + 1, "check_cell_index:: j :: {} not in range [{}, {}]", j, self.min_y, self.max_y);
        assert!(k >= self.min_z || k <= self.max_z + 1, "check_cell_index:: k :: {} not in range [{}, {}]", k, self.min_z, self.max_z);
    }

    fn get_cell_coordinates(&self, i: u32, j: u32, k: u32) -> Vector4<f32> {
        self.check_cell_index(i, j, k); // TODO: this should be checked when range is generated. TODO: remove in the future.
        let result: Vector4<f32> = Vector4::new(
            (self.min_x + i) as f32 * self.scale_factor,
            (self.min_y + j) as f32 * self.scale_factor,      
            (self.min_z as i32 - k as i32) as f32 * self.scale_factor,
            1.0
        );

        result
    }

    fn get_boundary_coordinates(&self, i: u32, j: u32, k: u32) -> Vector4<f32> {
        self.check_boundary_index(i, j, k); // TODO: this should be checked when range is generated. TODO: remove in the future.
        let result: Vector4<f32> = Vector4::new(
            ((self.min_x + i) as f32 - 0.5) * self.scale_factor,
            ((self.min_y + j) as f32 - 0.5) * self.scale_factor,
            ((self.min_z as i32 - k as i32) as f32 + 0.5) * self.scale_factor,
            1.0
        );

        result
    }

    /// Get cell index from point (i, j, k).
    fn get_cell_index(&self, i: u32, j: u32, k: u32) -> Result<usize, String> {
        self.check_cell_index(i, j, k); // TODO: this should be checked when range is generated. TODO: remove in the future.
        let x_range = self.get_x_range(); 
        let y_range = self.get_y_range(); 
        let index = i + j * x_range.end + k * x_range.end * y_range.end;
        if index >= self.cells.len() as u32 {
             Err(format!("DomainE::get_cell_index({}, {}, {}). Index out of bounds index {} > cells.len() {}.", i, j ,k, index, self.cells.len()))
        }
        else {
            Ok(index as usize)
        }
    }

    /// Get boundary index from point (i, j, k).
    fn get_boundary_index(&self, i: u32, j: u32, k: u32) -> Result<usize, String> {
        self.check_boundary_index(i, j, k); // TODO: this should be checked when range is generated. TODO: remove in the future.
        let x_range = self.get_boundary_x_range(); 
        let y_range = self.get_boundary_y_range(); 
        let index = i + j * x_range.end + k * x_range.end * y_range.end;
        if index >= self.boundary_points.points.len() as u32 {
             Err(format!("DomainE::get_bondary_index({}, {}, {}). Index out of bounds index {} > boundary_points.points.len() {}.",
             i,
             j,
             k,
             index,
             self.boundary_points.points.len()))
        }
        else {
            Ok(index as usize)
        }
    }

    /// Get cell from index (i,j,k)
    fn get_cell(&self, i: u32, j: u32, k: u32) -> Cell {
        let index = self.get_cell_index(i, j, k).unwrap(); 
        self.cells[index]
    }

    /// Get boundary value from index (i,j,k)
    fn get_boundary_value(&self, i: u32, j: u32, k: u32) -> f32 {
        let index = self.get_boundary_index(i, j, k).unwrap(); 
        self.boundary_points.points[index]
    }

    // Create Vec<f32> from cell values in vvvv cccc format. vvvv :: pos, cccc :: color.
    // Far cell point = red.  
    // Known cell point = greeb.  
    // Band cell point = blud.  
    pub fn cells_to_vec(&self) -> Vec<f32> {

        let mut result: Vec<f32> = Vec::new(); 

        for k in self.get_z_range() {
        for j in self.get_y_range() {
        for i in self.get_x_range() {

            let value = self.get_cell(i, j, k);

            //let cell_coordinates = self.get_cell_coordinates(i, j, k);
            let coordinates = self.get_cell_coordinates(i, j, k);
            result.push(coordinates.x);
            result.push(coordinates.y);
            result.push(coordinates.z);
            result.push(coordinates.w);
            
            // The color of the cell. Far points are red, Known green and Band cell blue.
            result.push(match value {Cell::Far()    => 1.0, _ => 0.0 }); // red
            result.push(match value {Cell::Known(_) => 1.0, _ => 0.0 }); // green
            result.push(match value {Cell::Band(_)  => 1.0, _ => 0.0 }); // blue
            result.push(1.0 as f32); // alpha
        }}};
        result
    }

    /// Create Vec<f32> from boundary values in vvvc format. vvv: pos, c :: [0.0, 1.0] a scaled/normalized boundary point value.
    pub fn boundary_points_to_vec(&self) -> Vec<f32> {
        println!("Converting boundary points to vec<f32>.");

        let mut result: Vec<f32> = Vec::new(); 

        for k in self.get_boundary_z_range() {
        for j in self.get_boundary_y_range() {
        for i in self.get_boundary_x_range() {

            let value = self.get_boundary_value(i,j,k);

            let boundary_coordinates = self.get_boundary_coordinates(i, j, k);
            let normalized_value = clamp(1.0 / (value.abs() + 1.0), 0.1, 1.0).powf(8.0);

            result.push(boundary_coordinates.x);
            result.push(boundary_coordinates.y);
            result.push(boundary_coordinates.z);
            result.push(normalized_value);
        }}};
        result
    }

    pub fn boundary_grid_to_vec(&self) -> Vec<f32> {

        println!("Converting boundary grid lines to vec<f32>.");
        let boundary_width = self.max_x - self.min_x + 2;
        let boundary_height = self.max_y - self.min_y + 2;
        let boundary_depth = self.max_z - self.min_z + 2;

        let mut result: Vec<f32> = Vec::new(); 

        // Horizontal lines.
        for k in 0..boundary_depth  {
        for j in 0..boundary_height {
            result.push(((self.min_x) as f32 - 0.5) * self.scale_factor);
            result.push(((self.min_y + j) as f32 - 0.5) * self.scale_factor);
            result.push(((self.min_z as i32 - k as i32) as f32 + 0.5) * self.scale_factor);
            result.push(0.1);
            result.push(((self.max_x + 1) as f32- 0.5) * self.scale_factor);
            result.push(((self.min_y + j) as f32 - 0.5) * self.scale_factor);
            result.push(((self.min_z as i32 - k as i32) as f32 + 0.5) * self.scale_factor);
            result.push(0.1);
        }};

        // Vertical lines.
        for k in 0..boundary_depth  {
        for i in 0..boundary_width {
            result.push(((self.min_x + i) as f32 - 0.5) * self.scale_factor);
            result.push(((self.min_y) as f32 - 0.5) * self.scale_factor);
            result.push(((self.min_z as i32 - k as i32) as f32 + 0.5) * self.scale_factor);
            result.push(0.1);
            result.push(((self.min_x + i) as f32- 0.5) * self.scale_factor);
            result.push(((self.max_y + 1) as f32 - 0.5) * self.scale_factor);
            result.push(((self.min_z as i32 - k as i32) as f32 + 0.5) * self.scale_factor);
            result.push(0.1);
        }};

        // Depth lines.
        for j in 0..boundary_height  {
        for i in 0..boundary_width {
            result.push(((self.min_x + i) as f32 - 0.5) * self.scale_factor);
            result.push(((self.min_y + j) as f32 - 0.5) * self.scale_factor);
            result.push(((self.min_z) as f32 + 0.5) * self.scale_factor);
            result.push(0.1);
            result.push(((self.min_x + i) as f32- 0.5) * self.scale_factor);
            result.push(((self.min_y + j) as f32 - 0.5) * self.scale_factor);
            result.push(((-(self.max_z as i32) + 1 as i32) as f32 + 0.5) * self.scale_factor);
            result.push(0.1);
        }};

        result
    }

    /// Initialize values for boundary points.
    pub fn initialize_boundary<F: Fn(f32, f32, f32) -> f32>(&mut self, b: F) {

        // let boundary_width = self.max_x - self.min_x + 2;
        // let boundary_height = self.max_y - self.min_y + 2;
        // let boundary_depth = self.max_z - self.min_z + 2;
        let mut min_value: f32 = 0.0;
        let mut max_value: f32 = 0.0;

        for k in self.get_boundary_x_range() {
        for j in self.get_boundary_y_range() {
        for i in self.get_boundary_z_range() {

            let index  = self.get_boundary_index(i, j, k).unwrap();
            // let index = (i + j * boundary_width + k * boundary_width * boundary_height) as usize;
            let boundary_coordinates = self.get_boundary_coordinates(i, j, k);
            let value = b(
               boundary_coordinates.x,
               boundary_coordinates.y,
               boundary_coordinates.z
            );
            if value < min_value { min_value = value; }
            if value > max_value { max_value = value; }
            self.boundary_points.points[index] = value; 
        }}};

        self.min_boundary_value = min_value;
        self.max_boundary_value = max_value;
    }
} // DomainE impl

// pub fn initialize_boundary_points<F: FnOnce(f32, f32, f32) -> f32>(domain: &mut DomainE, f: F) -> f32 {
//     let x = 0.5;
//     let y = 1.5;
//     let z = 2.5;
//     f(x,y,z)
// }

// Insert a value explicitly.
// fn add_initial_value(domain: DomainE, position: Vector3<u32>, value: Vector3<f32>) {
//     // assert!(position.x == 0 || position.y == 0 || position.z == 0 || position.x > domain.dim_x || position.y > domain.dim_y || position.z > domain.dim_z,
//     //     "Position ({}, {}, {}) not in domain range (0, 0, 0) .. ({}, {}, {}).",
//     //     position.x,
//     //     position.y,
//     //     position.z,
//     //     domain.dim_x,
//     //     domain.dim_y,
//     //     domain.dim_z
//     // );
// }

// pub fn is_inside_range(value: f32, min: f32, max: f32) -> bool {
//     assert!(max > min, "is_inside_range({}, {}, {}). min < max :: {} < {}", value, min, max, min, max);
//     value >= min && value <= max 
// }

pub fn clamp(value: f32, a: f32, b: f32) -> f32 {
    let min = if value < a { a } else { value }; 
    let result = if min > b { b } else { min }; 
    result
}

pub fn heap_test() {
    let mut heap = BinaryHeap::new(); 
    let joo: MinNonNan = Reverse(NotNan::new(1.0).unwrap());
    let ei: MinNonNan = Reverse(NotNan::new(0.5).unwrap());
    let ehheh: MinNonNan = Reverse(NotNan::new(1.1).unwrap());
    heap.push(joo);
    heap.push(ei);
    heap.push(ehheh);

    let mut lets_continue = true;
    while lets_continue {
        let result = match heap.pop() {
            Some(Reverse(x)) => { x.into_inner() },
            _ => {
                println!("Nyt onpi heappi tyhjä!");
                lets_continue = false;
                0.0
            }
        };
        println!("HEAP TEST :: {}", result);
    }
}

pub fn aabb(a: &Vector4<f32>, b: &Vector4<f32>, c: &Vector4<f32>) {

    let mut min_x = a.x;
    let mut min_y = a.y;
    let mut min_z = a.z;
    let mut max_x = a.x;
    let mut max_y = a.y;
    let mut max_z = a.z;

   if b.x < min_x { min_x = b.x } 
   if b.y < min_y { min_y = b.y } 
   if b.z < min_z { min_z = b.z } 

   if c.x < min_x { min_x = c.x } 
   if c.y < min_y { min_y = c.y } 
   if c.z < min_z { min_z = c.z } 

}

//fn get_cell(i: u32, j: u32, k: u32, domain: &DomainE) -> Result<Cell, String> {
//    let index = (i + j * boundary_width + k * boundary_width * boundary_height) as usize;
//
//}

// Update the neighbors of a KNOWN point:
//fn update_neighbors(i: u32, j: u32, k: u32, domain: &mut DomainE) {
//    let cell = domain.
//    assert!(match cell { Cell::Known(_) => true, _ => false },
//        "In update_neighbors function: cell must be known.");
//}

//pub fn initialize_interface(domain: &mut DomainE, initial_values: &Vec<Vector3<f32>>) {
//
//    domain.cells = vec![Cell::Far() ; (domain.dim_x * domain.dim_y * domain.dim_z) as usize];
//
//    let test_point0: Vector3<f32> = Vector3::new(5.5,10.0,-11.5);
//
//    //println!("initialize boundary points :: {}", initialize_boundary_points(|x, y, z| x+y+z));
//
//    //for i in 0..domain.dim_x {
//    //for j in 0..domain.dim_y {
//    //for k in 0..domain.dim_z {
//    //    let index = (i + j * domain.dim_y + j * domain.dim_y * k * domain.dim_z) as usize;
//    //    if is adjancent to domain
//    //    domain.cells[index] = Cell::Known(55.0);
//    //}}};
//}

// Narrow band Chopp: Computing Minimal Surfaces via Level Set Curvature Flow
//
// Ainoastaan upwind pisteitä voidaan käyttää laskettaessa narrow band arvoja. Näin varmistetaan
// oikea viskositeetti ratkaisu (1502.07303.pdf) sivu 8.
