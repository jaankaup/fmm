use std::convert::TryInto;
use bytemuck::{Pod, Zeroable};
use cgmath::{prelude::*, Vector3, Vector4};
//use zerocopy::{AsBytes, FromBytes};

/// A trait for things that can copy and convert a wgpu-rs buffer to
/// a std::Vec. 
pub trait Convert2Vec where Self: std::marker::Sized {
    fn convert(data: &[u8]) -> Vec<Self>;  
}

/// A macro for creating Convert2Vec for specific a primitive 
/// number type. Note that the type must implement from_ne_bytes.
/// This works only in async functions. This cannot be used
/// in winit event_loop! Use it before entering event_loop.
macro_rules! impl_convert {
  ($to_type:ty) => {
    impl Convert2Vec for $to_type {
      fn convert(data: &[u8]) -> Vec<Self> {
            let result = data
                .chunks_exact(std::mem::size_of::<Self>())
                .map(|b| *bytemuck::try_from_bytes::<Self>(b).unwrap())
                //.map(|b| Self::from_ne_bytes(b.try_into().unwrap())) //s::<Self>(b).unwrap())
                .collect();
            result
      }
    }
  }
}

// macro_rules! impl_convert {
//   ($to_type:ty) => {
//     impl Convert2Array for $to_type {
//       fn convert(data: &[u8]) -> Vec<Self> {
//         bytemuck::cast_slice::<u8, Self>(data);
//         //    let result = data
//         //        .chunks_exact(std::mem::size_of::<Self>())
//         //        .map(|b| *bytemuck::try_from_bytes::<Self>(b).unwrap())
//         //        .collect();
//         //    result
//       }
//     }
//   }
// }

impl_convert!{Vertex_vvvv_nnnn}
impl_convert!{f32}
impl_convert!{u32}
impl_convert!{u8}

pub enum VertexType {
    vvv(),
    vvvv(),
    vvvnnn(),
    vvvvnnnn(),
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Vertex_vvvc {
    pub position: [f32 ; 3],
    pub color: u32,
}

unsafe impl bytemuck::Zeroable for Vertex_vvvc {}
unsafe impl bytemuck::Pod for Vertex_vvvc {}

#[repr(C)]
#[derive(Debug, Clone, Copy)] //, AsBytes, FromBytes)]
pub struct Vertex_vvvc_nnnn {
    pub position: [f32 ; 3],
    pub color: u32,
    pub normal: [f32 ; 4],
}

unsafe impl bytemuck::Zeroable for Vertex_vvvc_nnnn {}
unsafe impl bytemuck::Pod for Vertex_vvvc_nnnn {}

#[repr(C)]
#[derive(Debug, Clone, Copy)] //, FromBytes, AsBytes)] //, AsBytes, FromBytes)]
pub struct Vertex_vvvv_nnnn {
    pub position: [f32 ; 4],
    pub normal: [f32 ; 4],
}

//impl FromBytes for Vertex_vvvv_nnnn {
//    fn from_ne_bytes_(bytes: &[u8]) -> Option<Self> {
//        let s = std::mem::size_of::<Self>();
//
//        // The size matches.
//        if s == bytes.len() {
//            let mut buf: Vec<
//        }
//        let vvvv = bytes.try_into(
//    }
//}

unsafe impl bytemuck::Zeroable for Vertex_vvvv_nnnn {}
unsafe impl bytemuck::Pod for Vertex_vvvv_nnnn {}

pub fn create_cube_triangles(pos: Vector3<f32>, cube_length: f32, cube_color: u32) -> Vec<Vertex_vvvc_nnnn> {

    let mut result: Vec<Vertex_vvvc_nnnn> = Vec::new(); 

    // Create cube.
    let p0 = pos - Vector3::<f32>::new(cube_length * 0.5, cube_length * 0.5,cube_length * 0.5);
    let p1 = p0 + Vector3::<f32>::new(0.0,         cube_length, 0.0);
    let p2 = p0 + Vector3::<f32>::new(cube_length, cube_length, 0.0);
    let p3 = p0 + Vector3::<f32>::new(cube_length, 0.0        , 0.0);

    let p4 = p0 + Vector3::<f32>::new(0.0,         0.0,         cube_length);
    let p5 = p0 + Vector3::<f32>::new(0.0,         cube_length, cube_length);
    let p6 = p0 + Vector3::<f32>::new(cube_length, cube_length, cube_length);
    let p7 = p0 + Vector3::<f32>::new(cube_length, 0.0,         cube_length);

    // p0, p3, p2, p0
    let front_0 = create_vvvc_nnnn_triangle(&p0, &p3, &p2, cube_color);
    let front_1 = create_vvvc_nnnn_triangle(&p0, &p2, &p1, cube_color);

    let back_0 = create_vvvc_nnnn_triangle(&p7, &p4, &p5, cube_color);
    let back_1 = create_vvvc_nnnn_triangle(&p7, &p5, &p6, cube_color);
     
    let top_0 = create_vvvc_nnnn_triangle(&p2, &p6, &p5, cube_color);
    let top_1 = create_vvvc_nnnn_triangle(&p2, &p5, &p1, cube_color);

    let bottom_0 = create_vvvc_nnnn_triangle(&p3, &p7, &p4, cube_color);
    let bottom_1 = create_vvvc_nnnn_triangle(&p3, &p4, &p0, cube_color);

    let left_0 = create_vvvc_nnnn_triangle(&p0, &p4, &p5, cube_color);
    let left_1 = create_vvvc_nnnn_triangle(&p0, &p5, &p1, cube_color);

    let right_0 = create_vvvc_nnnn_triangle(&p3, &p7, &p6, cube_color);
    let right_1 = create_vvvc_nnnn_triangle(&p3, &p6, &p2, cube_color);

    result.extend(front_0);
    result.extend(front_1);
    result.extend(back_0);
    result.extend(back_1);
    result.extend(top_0);
    result.extend(top_1);
    result.extend(bottom_0);
    result.extend(bottom_1);
    result.extend(left_0);
    result.extend(left_1);
    result.extend(right_0);
    result.extend(right_1);
    result
}

pub fn calculate_normal(a: &Vector3<f32>, b: &Vector3<f32>, c: &Vector3<f32>) -> Vector3<f32> {
    let u = b - a;
    let v = c - a;
    let result = u.cross(v).normalize();
    result
}

pub fn create_vvvc_nnnn_triangle(a: &Vector3<f32>, b: &Vector3<f32>, c: &Vector3<f32>, cube_color: u32) -> Vec<Vertex_vvvc_nnnn> {

    let normal = calculate_normal(&a, &b, &c); 
    //let normal = Vector3::<f32>::new(0.1, 0.1, 0.1);
    let mut result: Vec<Vertex_vvvc_nnnn> = Vec::new();
    //println!("position: [{}, {}, {}], color ..., normal[{}, {}, {}, {}]", a.x, a.y, a.z, normal.x, normal.y, normal.z , 0.0);
    //println!("position: [{}, {}, {}], color ..., normal[{}, {}, {}, {}]", b.x, b.y, b.z, normal.x, normal.y, normal.z , 0.0);
    //println!("position: [{}, {}, {}], color ..., normal[{}, {}, {}, {}]", c.x, c.y, c.z, normal.x, normal.y, normal.z , 0.0);
    //println!("position: [{}, {}, {}], color ..., normal[{}, {}, {}, {}]", a.x, a.y, a.z, normal.x, normal.y, normal.z , 0.0);
    result.push(Vertex_vvvc_nnnn {
        position: [a.x, a.y, a.z],
        color: cube_color,
        normal: [normal.x, normal.y, normal.z, 0.0],
    });
    result.push(Vertex_vvvc_nnnn {
        position: [b.x, b.y, b.z],
        color: cube_color,
        normal: [normal.x, normal.y, normal.z, 0.0],
    });
    result.push(Vertex_vvvc_nnnn {
        position: [c.x, c.y, c.z],
        color: cube_color,
        normal: [normal.x, normal.y, normal.z, 0.0],
    });
    // result.push(Vertex_vvvc_nnnn {
    //     position: [a.x, a.y, a.z],
    //     color: cube_color,
    //     normal: [normal.x, normal.y, normal.z, 0.0],
    // });
    result
}

///////////////////////////////////////////////////////////////////////////////////////

// #[repr(C)]
// #[derive(Clone, Copy)]
// pub struct Vertex {
//     _pos: [f32; 4],
//     _normal: [f32; 4],
// }
// 
// #[allow(dead_code)]
// pub fn vertex(pos: [f32; 3], nor: [f32; 3]) -> Vertex {
//     Vertex {
//         _pos: [pos[0], pos[1], pos[2], 1.0],
//         _normal: [nor[0], nor[1], nor[2], 0.0],
//     }
// }
// 
// unsafe impl Pod for Vertex {}
// unsafe impl Zeroable for Vertex {}

///////////////////////////////////////////////////////////////////////////////////////

// enum Vertex {
//     vvv([f32; 3]),
//     vvvv([f32; 4]),
//     vvvvnnnn([f32; 8]),
// }
// 
// trait Vertex_data {
// }

// pub struct Vertex {
//     pos: [f32; 4],
// }

/// Data for textured cube. vvvttnnn vvvttnnn vvvttnnn ...
#[allow(dead_code)]
pub fn create_cube() -> Vec<f32> {

    let v_data = [[1.0 , -1.0, -1.0],
                  [1.0 , -1.0, 1.0],
                  [-1.0, -1.0, 1.0],
                  [-1.0, -1.0, -1.0],
                  [1.0 , 1.0, -1.0],
                  [1.0, 1.0, 1.0],
                  [-1.0, 1.0, 1.0],
                  [-1.0, 1.0, -1.0],
    ];

    let t_data = [[0.748573,0.750412],
                 [0.749279,0.501284],
                 [0.999110,0.501077],
                 [0.999455,0.750380],
                 [0.250471,0.500702],
                 [0.249682,0.749677],
                 [0.001085,0.750380],
                 [0.001517,0.499994],
                 [0.499422,0.500239],
                 [0.500149,0.750166],
                 [0.748355,0.998230],
                 [0.500193,0.998728],
                 [0.498993,0.250415],
                 [0.748953,0.250920],
    ];
    
    let n_data = [ 
                  [0.0 , 0.0 , -1.0],
                  [-1.0, -0.0, 0.0],
                  [0.0, -0.0, 1.0],
                  [0.0, 0.0 , 1.0],
                  [1.0 , -0.0, 0.0],
                  [1.0 , 0.0 , 0.0],
                  [0.0 , 1.0 , 0.0],
                  [0.0, -1.0, 0.0],
    ];

    let mut vs: Vec<[f32; 3]> = Vec::new();
    let mut ts: Vec<[f32; 2]> = Vec::new();
    let mut vn: Vec<[f32; 3]> = Vec::new();

    vs.push(v_data[4]);
    ts.push(t_data[0]);
    vn.push(n_data[0]);
    vs.push(v_data[0]);
    ts.push(t_data[1]);
    vn.push(n_data[0]);
    vs.push(v_data[3]);
    ts.push(t_data[2]);
    vn.push(n_data[0]);

    // Face2
    //  f 5/1/1 4/3/1 8/4/1
    vs.push(v_data[4]);
    ts.push(t_data[0]);
    vn.push(n_data[0]);
    vs.push(v_data[3]);
    ts.push(t_data[2]);
    vn.push(n_data[0]);
    vs.push(v_data[7]);
    ts.push(t_data[3]);
    vn.push(n_data[0]);

    // Face3
    //  f 3/5/2 7/6/2 8/7/2
    vs.push(v_data[2]);
    ts.push(t_data[4]);
    vn.push(n_data[1]);
    vs.push(v_data[6]);
    ts.push(t_data[5]);
    vn.push(n_data[1]);
    vs.push(v_data[7]);
    ts.push(t_data[6]);
    vn.push(n_data[1]);

  // Face4
//  f 3/5/2 8/7/2 4/8/2
    vs.push(v_data[2]);
    ts.push(t_data[4]);
    vn.push(n_data[1]);
    vs.push(v_data[7]);
    ts.push(t_data[6]);
    vn.push(n_data[1]);
    vs.push(v_data[3]);
    ts.push(t_data[7]);
    vn.push(n_data[1]);

  // Face5
//  f 2/9/3 6/10/3 3/5/3
    vs.push(v_data[1]);
    ts.push(t_data[8]);
    vn.push(n_data[2]);
    vs.push(v_data[5]);
    ts.push(t_data[9]);
    vn.push(n_data[2]);
    vs.push(v_data[2]);
    ts.push(t_data[4]);
    vn.push(n_data[2]);

  // Face6
//  f 6/10/4 7/6/4 3/5/4
    vs.push(v_data[5]);
    ts.push(t_data[9]);
    vn.push(n_data[3]);
    vs.push(v_data[6]);
    ts.push(t_data[5]);
    vn.push(n_data[3]);
    vs.push(v_data[2]);
    ts.push(t_data[4]);
    vn.push(n_data[3]);

  // Face7
//  f 1/2/5 5/1/5 2/9/5
    vs.push(v_data[0]);
    ts.push(t_data[1]);
    vn.push(n_data[4]);
    vs.push(v_data[4]);
    ts.push(t_data[0]);
    vn.push(n_data[4]);
    vs.push(v_data[1]);
    ts.push(t_data[8]);
    vn.push(n_data[4]);

  // Face8
//  f 5/1/6 6/10/6 2/9/6
    vs.push(v_data[4]);
    ts.push(t_data[0]);
    vn.push(n_data[5]);
    vs.push(v_data[5]);
    ts.push(t_data[9]);
    vn.push(n_data[5]);
    vs.push(v_data[1]);
    ts.push(t_data[8]);
    vn.push(n_data[5]);

  // Face9
//  f 5/1/7 8/11/7 6/10/7
    vs.push(v_data[4]);
    ts.push(t_data[0]);
    vn.push(n_data[6]);
    vs.push(v_data[7]);
    ts.push(t_data[10]);
    vn.push(n_data[6]);
    vs.push(v_data[5]);
    ts.push(t_data[9]);
    vn.push(n_data[6]);

  // Face10
 // f 8/11/7 7/12/7 6/10/7
    vs.push(v_data[7]);
    ts.push(t_data[10]);
    vn.push(n_data[6]);
    vs.push(v_data[6]);
    ts.push(t_data[11]);
    vn.push(n_data[6]);
    vs.push(v_data[5]);
    ts.push(t_data[9]);
    vn.push(n_data[6]);

  // Face11
//  f 1/2/8 2/9/8 3/13/8
    vs.push(v_data[0]);
    ts.push(t_data[1]);
    vn.push(n_data[7]);
    vs.push(v_data[1]);
    ts.push(t_data[8]);
    vn.push(n_data[7]);
    vs.push(v_data[2]);
    ts.push(t_data[12]);
    vn.push(n_data[7]);

  // Face12
//  f 1/2/8 3/13/8 4/14/8
    vs.push(v_data[0]);
    ts.push(t_data[1]);
    vn.push(n_data[7]);
    vs.push(v_data[2]);
    ts.push(t_data[12]);
    vn.push(n_data[7]);
    vs.push(v_data[3]);
    ts.push(t_data[13]);
    vn.push(n_data[7]);

    let mut p_data: Vec<f32> = Vec::new();

    for i in 0..vs.len() {
        p_data.push(vs[i][0]); 
        p_data.push(vs[i][1]); 
        p_data.push(vs[i][2]); 
        p_data.push(ts[i][0]); 
        p_data.push(ts[i][1]); 
        p_data.push(vn[i][0]); 
        p_data.push(vn[i][1]); 
        p_data.push(vn[i][2]); 
    }

    p_data

}

///////////////////////////////////////////////////////////////////////////////////////

pub fn clamp(val: f32, min: f32, max: f32) -> f32 {
    let result  = if val >= max { max } else { val };
    let result2 = if result <= min { min } else { val };
    result2
}

// Encode vector to "rgba" uint. 
// Conversion: vec4(x,y,z,w) => Uint(xxyyzzww); 
pub fn encode_rgba_u32(r: u8, g: u8, b: u8, a: u8) -> u32 {

  // let mut result: f32 = 0.0;
  let mut color: u32 = 0;
  color = ((r as u32) << 24) | ((g as u32) << 16) | ((b as u32) << 8) | a as u32; 
  color

  // // Copy color bits to f32. This value is then decoded in shader.
  // unsafe { result = std::mem::transmute::<u32, f32>(color); }

  // result
}
