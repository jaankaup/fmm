use std::convert::TryInto;
use bytemuck::{Pod, Zeroable};

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
      //fn convert(&self, data: &mut [u8]) -> Vec<Self> {
      fn convert(data: &[u8]) -> Vec<Self> {
            let result = data
                .chunks_exact(std::mem::size_of::<Self>())
                .map(|b| Self::from_ne_bytes(b.try_into().unwrap()))
                .collect();
            result
      }
    }
  }
}

impl_convert!{f32}
impl_convert!{u32}
impl_convert!{u8}

///////////////////////////////////////////////////////////////////////////////////////

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Vertex {
    _pos: [f32; 4],
    _normal: [f32; 4],
}

#[allow(dead_code)]
pub fn vertex(pos: [f32; 3], nor: [f32; 3]) -> Vertex {
    Vertex {
        _pos: [pos[0], pos[1], pos[2], 1.0],
        _normal: [nor[0], nor[1], nor[2], 0.0],
    }
}

unsafe impl Pod for Vertex {}
unsafe impl Zeroable for Vertex {}

///////////////////////////////////////////////////////////////////////////////////////

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
