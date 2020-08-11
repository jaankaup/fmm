//fn d_minus_x() 

use cgmath::{Vector3, Vector4};

#[derive(Clone, Copy)]
pub enum Cell {
    Known(f32),
    Band(Option<f32>),
    Far(),
}

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

pub struct DomainE {
    pub dim_x: u32,
    pub dim_y: u32,
    pub dim_z: u32,
    pub cells: Vec<Cell>,
    pub ghost_points: InterfaceT,
    pub min_ghost_value: f32, // for debuging. Keep track of the minimun value of b(x).
    pub max_ghost_value: f32, // for debuging. Keep track of the maximum value of b(x).
}

impl DomainE {
    pub fn new(width: u32, height: u32, depth: u32) -> Self {

        assert!(width != 0 || height != 0 || depth != 0, "width, height and depth must be > 0");

        let cells: Vec<Cell> = vec![Cell::Far() ; (width * height * depth) as usize];
        let ghost_points: Vec<f32> = vec![0.0 ; ((width+1) * (height+1) * (depth+1)) as usize];

        Self {
            dim_x: width,
            dim_y: height,
            dim_z: depth,
            cells: cells,
            ghost_points: InterfaceT { points: ghost_points, },
            min_ghost_value: 0.0,
            max_ghost_value: 0.0,
        }
    }

    pub fn ghost_points_to_vec(&self) -> Vec<f32> {

        let boundary_width = self.dim_x + 1;
        let boundary_height = self.dim_y + 1;
        let boundary_depth = self.dim_z + 1;

        let mut result: Vec<f32> = Vec::new(); 
        //let mut result: Vec<f32> = Vec::with_capacity((boundary_width * boundary_height * boundary_depth * 4) as usize); 
        let ratio = (self.max_ghost_value - self.min_ghost_value).abs();
        println!("ratio == {}", ratio);
        let mut counter = 0;

        for k in 0..boundary_width  {
        for j in 0..boundary_height {
        for i in 0..boundary_depth  {

            let index = (i + j * boundary_width + k * boundary_width * boundary_height) as usize;
            let value = self.ghost_points.points[index];
            //let mut normalized_value: f32 = 0.0; 
            //if ratio > 0.0 {
            //    normalized_value = (value + self.min_ghost_value.abs()) / ratio;     
            //}
            let maximum_value = if self.max_ghost_value < self.min_ghost_value.abs() { self.min_ghost_value.abs() } else { self.max_ghost_value };
            let normalized_value = clamp(1.0 / (value.abs() + 1.0) , 0.0, 1.0).powf(2.0);
            let vec = Vector4::<f32> {
                x: i as f32 + 0.5,
                y: j as f32 + 0.5,
                z: k as f32 + 0.5,
                w: normalized_value as f32
            };    
            println!("ghost_point ({}, {}, {}, {})", vec.x, vec.y, vec.z, vec.w);
            result.push(i as f32 + 0.5);
            result.push(j as f32 + 0.5);
            result.push(k as f32 + 0.5);
            result.push(normalized_value);
            counter = counter + 1;
            //result.push(vec);
        }}};
        //println!("min/max ({}, {})", self.min_ghost_value, self.max_ghost_value);
        //println!("counter == {}", counter);
        result
    }

    pub fn initialize_boundary<F: Fn(f32, f32, f32) -> f32>(&mut self, b: F) {

        let boundary_width = self.dim_x + 1;
        let boundary_height = self.dim_y + 1;
        let boundary_depth = self.dim_z + 1;
        let mut min_value: f32 = 0.0;
        let mut max_value: f32 = 0.0;

        for k in 0..boundary_width  {
        for j in 0..boundary_height {
        for i in 0..boundary_depth  {

            let index = (i + j * boundary_width + k * boundary_width * boundary_height) as usize;
            let value = b(i as f32 + 0.5, j as f32 + 0.5, k as f32 + 0.5); 
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
    assert!(position.x == 0 || position.y == 0 || position.z == 0 || position.x > domain.dim_x || position.y > domain.dim_y || position.z > domain.dim_z,
        "Position ({}, {}, {}) not in domain range (0, 0, 0) .. ({}, {}, {}).",
        position.x,
        position.y,
        position.z,
        domain.dim_x,
        domain.dim_y,
        domain.dim_z
    );
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
