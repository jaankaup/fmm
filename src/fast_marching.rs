//fn d_minus_x() 

use cgmath::{Vector3, Vector4};

#[derive(Clone, Copy)]
pub enum Cell {
    Known(f32),
    Band(f32),
    Far(),
}

//#[derive(Clone, Copy)]
//pub struct Cell {
//    cell_type: Cell_Type,
//    coord: Vector3<u32>,
//}
//
//impl Cell {
//    pub fn new(x: u32, y: u32, z: u32, t: Cell_Type) -> Self {
//        let cell_type = t;
//        let coord = Vector3::<u32>::new(x, y, z);  
//
//        Self {
//            cell_type: cell_type,
//            coord: coord,
//        }
//    }
//}

//  
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

// Computational domanin for fast marching. Includes both cells and ghost points. 
pub struct DomainE {
    pub min_x: u32, // Minimum x-coordinate for cells. 
    pub max_x: u32, // Minimum y-coordinate for cells. 
    pub min_y: u32, // Minimum z-coordinate for cells. 
    pub max_y: u32, // Maximum x-coordinate for cells. 
    pub min_z: u32, // Maximum y-coordinate for cells. 
    pub max_z: u32, // Maximum z-coordinate for cells. 
    pub scale_factor: f32,
    pub cells: Vec<Cell>,
    pub ghost_points: InterfaceT,
    pub min_ghost_value: f32, // for debuging. Keep track of the minimun value of b(x).
    pub max_ghost_value: f32, // for debuging. Keep track of the maximum value of b(x).
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

        // Initialize ghost points with zeroes.
        let ghost_points: Vec<f32> = vec![0.0 ; ((delta_x+2) * (delta_y+2) * (delta_z+2)) as usize];

        Self {
            min_x: min_x,
            max_x: max_x,
            min_y: min_y,
            max_y: max_y,
            min_z: min_z,
            max_z: max_z,
            scale_factor: scale_factor,
            cells: cells,
            ghost_points: InterfaceT { points: ghost_points, },
            min_ghost_value: 0.0,
            max_ghost_value: 0.0,
        }
    }

    // Create Vec<f32> from cell values in vvvv cccc format. vvvv :: pos, cccc :: color.
    // Far cell point = red.  
    // Known cell point = greeb.  
    // Band cell point = blud.  
    pub fn cells_to_vec(&self) -> Vec<f32> {
        println!("Converting cell points to vec<f32>.");

        let boundary_width = self.max_x - self.min_x + 1;
        let boundary_height = self.max_y - self.min_y + 1;
        let boundary_depth = self.max_z - self.min_z + 1;

        let mut result: Vec<f32> = Vec::new(); 

        for k in 0..boundary_depth  {
        for j in 0..boundary_height {
        for i in 0..boundary_width  {
            let index = (i + j * boundary_width + k * boundary_width * boundary_height) as usize;
            let value = self.cells[index];

            result.push((self.min_x + i) as f32 * self.scale_factor);
            result.push((self.min_y + j) as f32 * self.scale_factor);
            result.push((self.min_z as i32 - k as i32) as f32 * self.scale_factor);
            result.push((1.0) as f32);
            println!("cell_point ({}, {}, {}, {})",
                (self.min_x + i) as f32,
                (self.min_y + j) as f32,
                (self.min_z as i32 - k as i32) as f32,
                1.0);
            
            result.push(match value {Cell::Far()   => 1.0, _ => 0.0 }); // red
            result.push(match value {Cell::Known(_) => 1.0, _ => 0.0 }); // green
            result.push(match value {Cell::Band(_)  => 1.0, _ => 0.0 }); // blue
            result.push(1.0 as f32); // alpha
        }}};
        result
    }

    // Create Vec<f32> from cell values in vvvc format. vvv: pos, c :: [0.0, 1.0] a scaled/normalized ghost point value.
    pub fn ghost_points_to_vec(&self) -> Vec<f32> {
        println!("Converting ghost points to vec<f32>.");

        let boundary_width = self.max_x - self.min_x + 2;
        let boundary_height = self.max_y - self.min_y + 2;
        let boundary_depth = self.max_z - self.min_z + 2;

        let mut result: Vec<f32> = Vec::new(); 

        for k in 0..boundary_depth  {
        for j in 0..boundary_height {
        for i in 0..boundary_width  {
            let index = (i + j * boundary_width + k * boundary_width * boundary_height) as usize;
            let value = self.ghost_points.points[index];

            let normalized_value = clamp(1.0 / (value.abs() + 1.0), 0.1, 1.0).powf(8.0);
            let vec = Vector4::<f32> {
                x: (self.min_x + i) as f32 - 0.5,
                y: (self.min_y + j) as f32 - 0.5,
                z: (self.min_z as i32 - k as i32) as f32 + 0.5,
                w: normalized_value as f32
            };    
            println!("ghost_point ({}, {}, {}, {})", vec.x, vec.y, vec.z, vec.w);
            result.push(((self.min_x + i) as f32 - 0.5) * self.scale_factor);
            result.push(((self.min_y + j) as f32 - 0.5) * self.scale_factor);
            result.push(((self.min_z as i32 - k as i32) as f32 + 0.5) * self.scale_factor);
            result.push(normalized_value);
        }}};
        result
    }

    pub fn ghost_grid_to_vec(&self) -> Vec<f32> {

        println!("Converting ghost grid lines to vec<f32>.");
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

    pub fn initialize_boundary<F: Fn(f32, f32, f32) -> f32>(&mut self, b: F) {

        let boundary_width = self.max_x - self.min_x + 2;
        let boundary_height = self.max_y - self.min_y + 2;
        let boundary_depth = self.max_z - self.min_z + 2;
        let mut min_value: f32 = 0.0;
        let mut max_value: f32 = 0.0;

        for k in 0..boundary_width  {
        for j in 0..boundary_height {
        for i in 0..boundary_depth  {

            let index = (i + j * boundary_width + k * boundary_width * boundary_height) as usize;
            let value = b(
               ((self.min_x + i) as f32 - 0.5) * self.scale_factor,
               ((self.min_y + j) as f32 - 0.5) * self.scale_factor,
               ((self.min_z + k) as f32 - 0.5) * self.scale_factor); 
            if value < min_value { min_value = value; }
            if value > max_value { max_value = value; }
            self.ghost_points.points[index] = value; 
        }}};

        for k in 0..boundary_width  {
        for j in 0..boundary_height {
        for i in 0..boundary_depth  {

            let index = (i + j * boundary_width + k * boundary_width * boundary_height) as usize;
            //println!("({}, {}, {}) == {}", i as f32 + 0.5, j as f32 + 0.5, k as f32 + 0.5, self.ghost_points.points[index]);  

        }}};
        self.min_ghost_value = min_value;
        self.max_ghost_value = max_value;
    }
} // DomainE impl

pub fn initialize_ghost_points<F: FnOnce(f32, f32, f32) -> f32>(domain: &mut DomainE, f: F) -> f32 {
    let x = 0.5;
    let y = 1.5;
    let z = 2.5;
    f(x,y,z)
}

// Insert a value explicitly.
fn add_initial_value(domain: DomainE, position: Vector3<u32>, value: Vector3<f32>) {
    // assert!(position.x == 0 || position.y == 0 || position.z == 0 || position.x > domain.dim_x || position.y > domain.dim_y || position.z > domain.dim_z,
    //     "Position ({}, {}, {}) not in domain range (0, 0, 0) .. ({}, {}, {}).",
    //     position.x,
    //     position.y,
    //     position.z,
    //     domain.dim_x,
    //     domain.dim_y,
    //     domain.dim_z
    // );
}

pub fn is_inside_range(value: f32, min: f32, max: f32) -> bool {
    assert!(max > min, "is_inside_range({}, {}, {}). min < max :: {} < {}", value, min, max, min, max);
    value >= min && value <= max 
}

pub fn clamp(value: f32, a: f32, b: f32) -> f32 {
    let min = if value < a { a } else { value }; 
    let result = if min > b { b } else { min }; 
    result
}

fn dx_minus(x: u32, y: u32, z: u32, domain: &DomainE) -> f32 {
    //assert!(x == 0 || y > domain.dim_y || z == 0 || position.x > domain.dim_x || position.y > domain.dim_y || position.z > domain.dim_z,
    123.0
}
//pub fn initialize_interface(domain: &mut DomainE, initial_values: &Vec<Vector3<f32>>) {
//
//    domain.cells = vec![Cell::Far() ; (domain.dim_x * domain.dim_y * domain.dim_z) as usize];
//
//    let test_point0: Vector3<f32> = Vector3::new(5.5,10.0,-11.5);
//
//    //println!("initialize ghost points :: {}", initialize_ghost_points(|x, y, z| x+y+z));
//
//    //for i in 0..domain.dim_x {
//    //for j in 0..domain.dim_y {
//    //for k in 0..domain.dim_z {
//    //    let index = (i + j * domain.dim_y + j * domain.dim_y * k * domain.dim_z) as usize;
//    //    if is adjancent to domain
//    //    domain.cells[index] = Cell::Known(55.0);
//    //}}};
//}
