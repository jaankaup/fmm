use bytemuck::{Pod, Zeroable};

/// Uniform data for marching cubes.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Mc_uniform_data {
    pub base_position: cgmath::Vector4<f32>,
    pub isovalue: f32,
    pub cube_length: f32,
    pub joopajoo: f32,
    pub joopajoo2: f32,
}

unsafe impl Pod for Mc_uniform_data {}
unsafe impl Zeroable for Mc_uniform_data {}

