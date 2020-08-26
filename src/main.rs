//use futures::task::LocalSpawn;
use fmm::object_loader::load_triangles_from_obj;
use std::borrow::Cow::Borrowed;
//use futures::executor::LocalPool;
//use futures::executor::LocalSpawner;
use std::collections::HashMap;
use rand::prelude::*;
use std::time::{SystemTime, UNIX_EPOCH};
use jaankaup_hilbert::hilbert::hilbert_index_reverse;
use cgmath::{prelude::*, Vector3, Vector4};
use fmm::misc::*;
use fmm::buffer::*;
use fmm::texture::*;
use fmm::camera::*;
//use fmm::radix_sort::*;
use fmm::marching_cubes::*;
//use fmm::radix_sort::*;
use fmm::app_resources::*;
use fmm::fast_marching::*;
use fmm::bvh::*;
use fmm::fmm::*;
//use crate::misc::VertexType;
//use crate::bvh::{Triangle, BBox, Plane}; 

use winit::{
    event::{Event, WindowEvent,KeyboardInput,ElementState,VirtualKeyCode},
    event_loop::{ControlFlow, EventLoop},
    window::Window
};

// Ray camera resolution.
static CAMERA_RESOLUTION: (u32, u32) = (512,512);

enum Example {
    TwoTriangles,
    Cube,
    Mc,
    VolumetricNoise,
    FmmGhostPoints,
    CellPoints,
}

struct Fmm {
    boundary_points: bool,
    grids: bool,
    cells: bool,
    triangles: bool,
    aabb: bool,
}

struct Buffers {
    camera_uniform_buffer: BufferInfo,
    ray_camera_uniform_buffer: BufferInfo,
    mc_uniform_buffer: BufferInfo,
    mc_counter_buffer: BufferInfo,
    mc_output_buffer: BufferInfo,
    ray_march_output_buffer: BufferInfo,
    ray_debug_buffer: BufferInfo,
    fmm_points: BufferInfo,
    fmm_cell_points: BufferInfo,
    fmm_boundary_lines: BufferInfo,
}

// Size in bytes.
static BUFFERS:  Buffers = Buffers {
    camera_uniform_buffer:       BufferInfo { name: "camera_uniform_buffer",     size: None,},
    ray_camera_uniform_buffer:   BufferInfo { name: "ray_camera_uniform_buffer", size: None,},
    mc_uniform_buffer:           BufferInfo { name: "mc_uniform_buffer",         size: None,},
    mc_counter_buffer:           BufferInfo { name: "mc_counter_buffer",         size: None,},
    mc_output_buffer:            BufferInfo { name: "mc_output_buffer",          size: Some(64*64*64*24), },
    ray_march_output_buffer:     BufferInfo { name: "ray_march_output",          size: Some(CAMERA_RESOLUTION.0 as u32 * CAMERA_RESOLUTION.1 as u32 * 4),},
    ray_debug_buffer:            BufferInfo { name: "ray_debug_buffer",          size: Some(CAMERA_RESOLUTION.0 as u32 * CAMERA_RESOLUTION.1 as u32 * 4 * 4),},
    fmm_points:                  BufferInfo { name: "fmm_points",                size: None,},
    fmm_cell_points:             BufferInfo { name: "fmm_cell_points",           size: None,},
    fmm_boundary_lines:             BufferInfo { name: "fmm_boundary_lines",           size: None,},
};

#[derive(Clone, Copy)]
struct ShaderModuleInfo {
    name: &'static str,
    source_file: &'static str,
    _stage: &'static str, // TODO: remove? 
}

enum Resource {
    TextureView(&'static str),
    TextureSampler(&'static str),
    Buffer(&'static str),
}

struct VertexBufferInfo {
    vertex_buffer_name: String,
    _index_buffer: Option<String>,
    start_index: u32,
    end_index: u32,
    instances: u32,
}

struct RenderPass {
    pipeline: wgpu::RenderPipeline,
    bind_groups: Vec<wgpu::BindGroup>,
}

struct ComputePass {
    pipeline: wgpu::ComputePipeline,
    bind_groups: Vec<wgpu::BindGroup>,
    dispatch_x: u32,
    dispatch_y: u32,
    dispatch_z: u32,
}

impl RenderPass {
    fn execute(&self,
               encoder: &mut wgpu::CommandEncoder,
               frame: &wgpu::SwapChainTexture,
               multisampled_framebuffer: &wgpu::TextureView,
               textures: &HashMap<String, Texture>,
               buffers: &HashMap<String, Buffer>,
               vertex_buffer_info: &VertexBufferInfo,
               sample_count: u32,
               clear: bool) {

            let multi_sampled = multisampled(sample_count);

            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: Borrowed(&[
                    wgpu::RenderPassColorAttachmentDescriptor {
                            attachment: match multi_sampled { false => &frame.view, true => &multisampled_framebuffer, },
                            resolve_target: match multi_sampled { false => None, true => Some(&frame.view), },
                            ops: wgpu::Operations {
                                load: match clear {
                                    true => {
                                        wgpu::LoadOp::Clear(wgpu::Color { 
                                            r: 0.0,
                                            g: 0.0,
                                            b: 0.0,
                                            a: 1.0,
                                        })
                                    }
                                    false => {
                                        wgpu::LoadOp::Load
                                    }
                                },
                                store: true,
                            },
                    }
                ]),
                //depth_stencil_attachment: None,
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachmentDescriptor {
                    attachment: &textures.get(TEXTURES.depth.name).unwrap().view,
                    depth_ops: Some(wgpu::Operations {
                            load: match clear { true => wgpu::LoadOp::Clear(1.0), false => wgpu::LoadOp::Load }, 
                            store: true,
                    }),
                    stencil_ops: None,
                    }),
            });

            render_pass.set_pipeline(&self.pipeline);

            // Set bind groups.
            for (e, bgs) in self.bind_groups.iter().enumerate() {
                render_pass.set_bind_group(e as u32, &bgs, &[]);
            }

            // Set vertex buffer.
            render_pass.set_vertex_buffer(
                0,
                buffers.get(&vertex_buffer_info.vertex_buffer_name).unwrap().buffer.slice(..)
            );

            // TODO: handle index buffer.

            // Draw.
            render_pass.draw(vertex_buffer_info.start_index..vertex_buffer_info.end_index, 0..vertex_buffer_info.instances);
    }
}

impl ComputePass {

    fn execute(&self, encoder: &mut wgpu::CommandEncoder) {

        let mut ray_pass = encoder.begin_compute_pass();
        ray_pass.set_pipeline(&self.pipeline);
        for (e, bgs) in self.bind_groups.iter().enumerate() {
            ray_pass.set_bind_group(e as u32, &bgs, &[]);
        }
        ray_pass.dispatch(self.dispatch_x, self.dispatch_y, self.dispatch_z);
    }
}

struct BindGroupInfo {
    binding: u32,
    visibility: wgpu::ShaderStage,
    resource: Resource, 
    binding_type: wgpu::BindingType,
}

struct BufferInfo {
    name: &'static str,
    size: Option<u32>,
}

struct RenderPipelineInfo {
    vertex_shader: ShaderModuleInfo,
    fragment_shader: Option<ShaderModuleInfo>,
    bind_groups: Vec<Vec<BindGroupInfo>>,
    input_formats: Vec<(wgpu::VertexFormat, u64)>, 
}

struct ComputePipelineInfo {
    compute_shader: ShaderModuleInfo,
    bind_groups: Vec<Vec<BindGroupInfo>>,
}

static VTN_SHADERS: [ShaderModuleInfo; 2]  = [
    ShaderModuleInfo {name: "vtn_render_vert", source_file: "vtn_render_vert.spv", _stage: "vertex"},
    ShaderModuleInfo {name: "vtn_render_frag", source_file: "vtn_render_frag.spv", _stage: "frag"},
];

static MC_RENDER_SHADERS: [ShaderModuleInfo; 2]  = [
    ShaderModuleInfo {name: "mc_render_vert", source_file: "mc_render_vert.spv", _stage: "vertex"},
    ShaderModuleInfo {name: "mc_render_frag", source_file: "mc_render_frag.spv", _stage: "frag"},
];

static LINE_SHADERS: [ShaderModuleInfo; 2]  = [
    ShaderModuleInfo {name: "line_vert", source_file: "line_vert.spv", _stage: "vertex"},
    ShaderModuleInfo {name: "line_frag", source_file: "line_frag.spv", _stage: "frag"},
];

static LINE_SHADERS_VVVC_NNNN: [ShaderModuleInfo; 2]  = [
    ShaderModuleInfo {name: "cube_instanced_vert", source_file: "cube_instanced.vert.spv", _stage: "vertex"},
    ShaderModuleInfo {name: "cube_instanced_frag", source_file: "cube_instanced.frag.spv", _stage: "frag"},
];

static LINE_SHADERS_4PX: [ShaderModuleInfo; 2]  = [
    ShaderModuleInfo {name: "line_vert_4px", source_file: "line_vert_4px.spv", _stage: "vertex"},
    ShaderModuleInfo {name: "line_frag_4px", source_file: "line_frag_4px.spv", _stage: "frag"},
];

static LINE_SHADERS_VN: [ShaderModuleInfo; 2]  = [
    ShaderModuleInfo {name: "line_vert_vn", source_file: "line_vn.vert.spv", _stage: "vertex"},
    ShaderModuleInfo {name: "line_frag_vn", source_file: "line_vn.frag.spv", _stage: "frag"},
];

static MARCHING_CUBES_SHADER: ShaderModuleInfo  = ShaderModuleInfo { 
           name: "mc",
           source_file: "mc.spv",
           _stage: "compute",
};

static RAY_MARCH_SHADER: ShaderModuleInfo  = ShaderModuleInfo { 
           name: "ray_march",
           source_file: "ray.spv",
           _stage: "compute",
};

static SPHERE_TRACER_SHADER: ShaderModuleInfo  = ShaderModuleInfo { 
           name: "sphere_tracer",
           source_file: "sphere_tracer_comp.spv",
           _stage: "compute",
};

static GENERATE_3D_SHADER: ShaderModuleInfo  = ShaderModuleInfo { 
           name: "noise_shader",
           source_file: "generate_noise3d.spv",
           _stage: "compute",
};

static BITONIC_SHADER: ShaderModuleInfo  = ShaderModuleInfo { 
           name: "bitonic_shader",
           source_file: "local_sort_comp.spv",
           _stage: "compute",
};

static RADIX_SHADER: ShaderModuleInfo  = ShaderModuleInfo { 
           name: "radix_0",
           source_file: "radix_comp.spv",
           _stage: "compute",
};

fn multisampled(sample_count: u32) -> bool {
    match sample_count { 1 => false, 2 => true, 4 => true, 8 => true, 16 => true, _ => panic!("Illegal sample count {}.", sample_count) }
}

// Create a pipeline and binding groups for shader that renders a given texture to the screen.
// Should be used with two_triangles buffer. 
fn create_two_triangles_info(texture_name: &'static str, sample_count: u32) -> RenderPipelineInfo { 
    let two_triangles_info: RenderPipelineInfo = RenderPipelineInfo {
        vertex_shader: ShaderModuleInfo {
            name: "two_triangles_vert",
            source_file: "two_triangles_vert.spv",
            _stage: "vertex"
        }, 
        fragment_shader: Some(ShaderModuleInfo {
            name: "two_triangles_frag",
            source_file: "two_triangles_frag.spv",
            _stage: "frag"
        }), 
        bind_groups: vec![
                vec![ 
                    BindGroupInfo {
                        binding: 0,
                        visibility: wgpu::ShaderStage::FRAGMENT,
                        resource: Resource::TextureView(texture_name),
                        binding_type: wgpu::BindingType::SampledTexture {
                           multisampled: multisampled(sample_count),
                           component_type: wgpu::TextureComponentType::Float,
                           dimension: wgpu::TextureViewDimension::D2,
                        },
                    }, 
                    BindGroupInfo {
                        binding: 1,
                        visibility: wgpu::ShaderStage::FRAGMENT,
                        resource: Resource::TextureSampler(texture_name),
                        binding_type: wgpu::BindingType::Sampler {
                           comparison: false,
                        },
                    },
                ],
        ],
        input_formats: vec![
            (wgpu::VertexFormat::Float4, 4 * std::mem::size_of::<f32>() as u64),
            (wgpu::VertexFormat::Float4, 4 * std::mem::size_of::<f32>() as u64)
        ],
    };
    two_triangles_info
}

// Creates a pipeline and bindgroup infos for a shader that has a camera uniform and a float4 input..
fn vvvv_camera_info(camera_uniform: &'static str, vertex_shader_name: &'static str, fragment_shader_name: &'static str, _sample_count: u32) -> RenderPipelineInfo { 
    let line_info: RenderPipelineInfo = RenderPipelineInfo {
        vertex_shader: ShaderModuleInfo {
            name: vertex_shader_name, //LINE_SHADERS[0].name,
            source_file: "todo", // LINE_SHADERS[0].source_file,
            _stage: "vertex"
        }, 
        fragment_shader: Some(ShaderModuleInfo {
            name: fragment_shader_name, //LINE_SHADERS[1].name,
            source_file: "todo", //LINE_SHADERS[1].source_file,
            _stage: "frag"
        }), 
        bind_groups: vec![
               vec![
                   BindGroupInfo {
                            binding: 0,
                            visibility: wgpu::ShaderStage::VERTEX | wgpu::ShaderStage::FRAGMENT,
                            resource: Resource::Buffer(camera_uniform),
                            binding_type: wgpu::BindingType::UniformBuffer {
                               dynamic: false,
                               min_binding_size: None,
                            },
                   }, 
               ],
        ],
        input_formats: vec![
            (wgpu::VertexFormat::Float3, 3 * std::mem::size_of::<f32>() as u64),
            (wgpu::VertexFormat::Uint, std::mem::size_of::<u32>() as u64),
        ],
    };

    line_info
}

fn vvvc_nnnn_camera_info(camera_uniform: &'static str,
                         // instanced_buffer_name: &'static str,
                         vertex_shader_name: &'static str,
                         fragment_shader_name: &'static str,
                         _sample_count: u32) -> RenderPipelineInfo { 
    let line_info: RenderPipelineInfo = RenderPipelineInfo {
        vertex_shader: ShaderModuleInfo {
            name: vertex_shader_name,
            source_file: "todo",
            _stage: "vertex"
        }, 
        fragment_shader: Some(ShaderModuleInfo {
            name: fragment_shader_name,
            source_file: "todo",
            _stage: "frag"
        }), 
        bind_groups: vec![
               vec![
                   BindGroupInfo {
                            binding: 0,
                            visibility: wgpu::ShaderStage::VERTEX | wgpu::ShaderStage::FRAGMENT,
                            resource: Resource::Buffer(camera_uniform),
                            binding_type: wgpu::BindingType::UniformBuffer {
                               dynamic: false,
                               min_binding_size: None,
                            },
                   }, 
//                   BindGroupInfo {
//                            binding: 1,
//                            visibility: wgpu::ShaderStage::COMPUTE,
//                            resource: Resource::Buffer(instanced_buffer_name),
//                            binding_type: wgpu::BindingType::StorageBuffer {
//                               dynamic: false,
//                               readonly: false,
//                               min_binding_size: None,
//                            },
//                   }, 
               ],
        ],
        input_formats: vec![
            (wgpu::VertexFormat::Float3, 3 * std::mem::size_of::<f32>() as u64),
            (wgpu::VertexFormat::Uint, std::mem::size_of::<u32>() as u64),
            (wgpu::VertexFormat::Float4, 4 * std::mem::size_of::<f32>() as u64),
        ],
    };

    line_info
}

// Creates a pipeline and bindgroup infos for a shader that has a camera uniform and a float4, float4 input..
fn v4_v4_camera_info(camera_uniform: &'static str, vertex_shader_name: &'static str, fragment_shader_name: &'static str, _sample_count: u32) -> RenderPipelineInfo { 
    let line_info_vn: RenderPipelineInfo = RenderPipelineInfo {
        vertex_shader: ShaderModuleInfo {
            name: vertex_shader_name,
            source_file: "todo", //LINE_SHADERS_VN[0].source_file,
            _stage: "vertex"
        }, 
        fragment_shader: Some(ShaderModuleInfo {
            name: fragment_shader_name, //LINE_SHADERS_VN[1].name,
            source_file: "todo",
            _stage: "frag"
        }), 
        bind_groups: vec![
               vec![
                   BindGroupInfo {
                            binding: 0,
                            visibility: wgpu::ShaderStage::VERTEX | wgpu::ShaderStage::FRAGMENT,
                            resource: Resource::Buffer(camera_uniform),
                            binding_type: wgpu::BindingType::UniformBuffer {
                               dynamic: false,
                               min_binding_size: None,
                            },
                   }, 
               ],
        ],
        input_formats: vec![
            (wgpu::VertexFormat::Float4, 4 * std::mem::size_of::<f32>() as u64),
            (wgpu::VertexFormat::Float4, 4 * std::mem::size_of::<f32>() as u64)
        ],
    };

    line_info_vn
}

fn vtn_renderer_info(sample_count: u32) -> RenderPipelineInfo { 
   let vtn_renderer_info: RenderPipelineInfo = RenderPipelineInfo {
       vertex_shader: ShaderModuleInfo {
           name: VTN_SHADERS[0].name,
           source_file: VTN_SHADERS[0].source_file,
           _stage: "vertex"
       }, 
       fragment_shader: Some(ShaderModuleInfo {
           name: VTN_SHADERS[1].name,
           source_file: VTN_SHADERS[1].source_file,
           _stage: "frag"
       }), 
       bind_groups:
           vec![ 
               vec![
                   BindGroupInfo {
                            binding: 0,
                            visibility: wgpu::ShaderStage::VERTEX | wgpu::ShaderStage::FRAGMENT,
                            resource: Resource::Buffer(BUFFERS.camera_uniform_buffer.name),
                            binding_type: wgpu::BindingType::UniformBuffer {
                               dynamic: false,
                               min_binding_size: None,
                            },
                   }, 
               ],
               vec![
                   BindGroupInfo {
                            binding: 0,
                            visibility: wgpu::ShaderStage::FRAGMENT,
                            resource: Resource::TextureView(TEXTURES.grass.name),
                            binding_type: wgpu::BindingType::SampledTexture {
                               multisampled: multisampled(sample_count),
                               component_type: wgpu::TextureComponentType::Float,
                               dimension: wgpu::TextureViewDimension::D2,
                            },
                   }, 
                   BindGroupInfo {
                       binding: 1,
                       visibility: wgpu::ShaderStage::FRAGMENT,
                       resource: Resource::TextureSampler(TEXTURES.grass.name),
                       binding_type: wgpu::BindingType::Sampler {
                          comparison: false,
                       },
                   },
               ],
           ],
           input_formats: vec![
               (wgpu::VertexFormat::Float3, 3 * std::mem::size_of::<f32>() as u64),
               (wgpu::VertexFormat::Float2, 2 * std::mem::size_of::<f32>() as u64),
               (wgpu::VertexFormat::Float3, 3 * std::mem::size_of::<f32>() as u64)
           ],
    };

    vtn_renderer_info
}

fn ray_renderer_info(sample_count: u32) -> RenderPipelineInfo { 
    let ray_renderer_info: RenderPipelineInfo = RenderPipelineInfo {
        vertex_shader: ShaderModuleInfo {
            name: "two_triangles_vert",
            source_file: "two_triangles_vert.spv",
            _stage: "vertex"
        }, 
        fragment_shader: Some(ShaderModuleInfo {
            name: "two_triangles_frag",
            source_file: "two_triangles_frag.spv",
            _stage: "frag"
        }), 
        bind_groups: vec![
                vec![ 
                    BindGroupInfo {
                        binding: 0,
                        visibility: wgpu::ShaderStage::FRAGMENT,
                        resource: Resource::TextureView(TEXTURES.ray_texture.name),
                        binding_type: wgpu::BindingType::SampledTexture {
                           multisampled: multisampled(sample_count),
                           component_type: wgpu::TextureComponentType::Float,
                           dimension: wgpu::TextureViewDimension::D2,
                        },
                    }, 
                    BindGroupInfo {
                        binding: 1,
                        visibility: wgpu::ShaderStage::FRAGMENT,
                        resource: Resource::TextureSampler(TEXTURES.ray_texture.name),
                        binding_type: wgpu::BindingType::Sampler {
                           comparison: false,
                        },
                    },
                ],
        ],
        input_formats: vec![
            (wgpu::VertexFormat::Float4, 4 * std::mem::size_of::<f32>() as u64),
            (wgpu::VertexFormat::Float4, 4 * std::mem::size_of::<f32>() as u64)
        ],
    };

    ray_renderer_info
}

fn mc_renderer_info(sample_count: u32) -> RenderPipelineInfo { 
   let mc_renderer_info: RenderPipelineInfo = RenderPipelineInfo {
       vertex_shader: ShaderModuleInfo {
           name: MC_RENDER_SHADERS[0].name,
           source_file: MC_RENDER_SHADERS[0].source_file,
           _stage: "vertex"
       }, 
       fragment_shader: Some(ShaderModuleInfo {
           name: MC_RENDER_SHADERS[1].name,
           source_file: MC_RENDER_SHADERS[1].source_file,
           _stage: "frag"
       }), 
       bind_groups:
           vec![ 
               vec![
                   BindGroupInfo {
                            binding: 0,
                            visibility: wgpu::ShaderStage::VERTEX | wgpu::ShaderStage::FRAGMENT,
                            resource: Resource::Buffer(BUFFERS.camera_uniform_buffer.name),
                            binding_type: wgpu::BindingType::UniformBuffer {
                               dynamic: false,
                               min_binding_size: None,
                            },
                   }, 
               ],
               vec![
                   BindGroupInfo {
                            binding: 0,
                            visibility: wgpu::ShaderStage::FRAGMENT,
                            resource: Resource::TextureView(TEXTURES.grass.name),
                            binding_type: wgpu::BindingType::SampledTexture {
                               multisampled: multisampled(sample_count),
                               component_type: wgpu::TextureComponentType::Float,
                               dimension: wgpu::TextureViewDimension::D2,
                            },
                   }, 
                   BindGroupInfo {
                       binding: 1,
                       visibility: wgpu::ShaderStage::FRAGMENT,
                       resource: Resource::TextureSampler(TEXTURES.grass.name),
                       binding_type: wgpu::BindingType::Sampler {
                          comparison: false,
                       },
                   },
                   BindGroupInfo {
                       binding: 2,
                       visibility: wgpu::ShaderStage::FRAGMENT,
                       resource: Resource::TextureView(TEXTURES.rock.name),
                       binding_type: wgpu::BindingType::SampledTexture {
                          multisampled: multisampled(sample_count),
                          component_type: wgpu::TextureComponentType::Float,
                          dimension: wgpu::TextureViewDimension::D2,
                       },
                   },
                  BindGroupInfo {
                      binding: 3,
                      visibility: wgpu::ShaderStage::FRAGMENT,
                      resource: Resource::TextureSampler(TEXTURES.rock.name),
                      binding_type: wgpu::BindingType::Sampler {
                         comparison: false,
                      },
                  },
               ],
           ],
           input_formats: vec![
               (wgpu::VertexFormat::Float4, 4 * std::mem::size_of::<f32>() as u64),
               (wgpu::VertexFormat::Float4, 4 * std::mem::size_of::<f32>() as u64)
           ],
    };

    mc_renderer_info
}

fn marching_cubes_info() -> ComputePipelineInfo {
   let marching_cubes_info: ComputePipelineInfo = ComputePipelineInfo {
       compute_shader: ShaderModuleInfo {
           name: MARCHING_CUBES_SHADER.name,
           source_file: MARCHING_CUBES_SHADER.source_file,
           _stage: "compute"
       }, 
       bind_groups:
           vec![ 
               vec![
                   BindGroupInfo {
                            binding: 0,
                            visibility: wgpu::ShaderStage::COMPUTE,
                            resource: Resource::Buffer(BUFFERS.mc_uniform_buffer.name),
                            binding_type: wgpu::BindingType::UniformBuffer {
                               dynamic: false,
                               min_binding_size: None,
                            },
                   }, 
                   BindGroupInfo {
                            binding: 1,
                            visibility: wgpu::ShaderStage::COMPUTE,
                            resource: Resource::Buffer(BUFFERS.mc_counter_buffer.name),
                            binding_type: wgpu::BindingType::StorageBuffer {
                               dynamic: false,
                               readonly: false,
                               min_binding_size: None,
                            },
                   }, 
               ],
               vec![
                   BindGroupInfo {
                            binding: 0,
                            visibility: wgpu::ShaderStage::COMPUTE,
                            resource: Resource::Buffer(BUFFERS.mc_output_buffer.name),
                            binding_type: wgpu::BindingType::StorageBuffer {
                               dynamic: false,
                               readonly: false,
                               min_binding_size: None,
                            },
                   }, 
               ],
           ],
    };

    marching_cubes_info
}

fn ray_march_info(sample_count: u32) -> ComputePipelineInfo {
   let ray_march_info: ComputePipelineInfo = ComputePipelineInfo {
       compute_shader: ShaderModuleInfo {
           name: RAY_MARCH_SHADER.name,
           source_file: RAY_MARCH_SHADER.source_file,
           _stage: "compute"
       }, 
       bind_groups:
           vec![ 
               vec![
                   BindGroupInfo {
                        binding: 0,
                        visibility: wgpu::ShaderStage::COMPUTE,
                        resource: Resource::Buffer(BUFFERS.ray_camera_uniform_buffer.name),
                        binding_type: wgpu::BindingType::UniformBuffer {
                           dynamic: false,
                           min_binding_size: None, // wgpu::BufferSize::new(std::mem::size_of::<RayCameraUniform>() as u64) * 4,
                        },
                   },
               ],
               vec![
                   BindGroupInfo {
                        binding: 0,
                        visibility: wgpu::ShaderStage::COMPUTE,
                        resource: Resource::TextureView(TEXTURES.grass.name),
                        binding_type: wgpu::BindingType::SampledTexture {
                           multisampled: multisampled(sample_count),
                           component_type: wgpu::TextureComponentType::Float,
                           dimension: wgpu::TextureViewDimension::D2,
                        },
                   },
                   BindGroupInfo {
                        binding: 1,
                        visibility: wgpu::ShaderStage::COMPUTE,
                        resource: Resource::TextureView(TEXTURES.grass.name),
                        binding_type: wgpu::BindingType::SampledTexture {
                           multisampled: multisampled(sample_count),
                           component_type: wgpu::TextureComponentType::Float,
                           dimension: wgpu::TextureViewDimension::D2,
                        },
                   },
               ],
               vec![
                   BindGroupInfo {
                        binding: 0,
                        visibility: wgpu::ShaderStage::COMPUTE,
                        resource: Resource::Buffer(BUFFERS.ray_march_output_buffer.name),
                        binding_type: wgpu::BindingType::StorageBuffer {
                           dynamic: false,
                           readonly: false,
                           min_binding_size: wgpu::BufferSize::new(256*256*4),
                        },
                   },
               ],
               vec![
                   BindGroupInfo {
                        binding: 0,
                        visibility: wgpu::ShaderStage::COMPUTE,
                        resource: Resource::Buffer(BUFFERS.ray_debug_buffer.name),
                        binding_type: wgpu::BindingType::StorageBuffer {
                           dynamic: false,
                           readonly: false,
                           min_binding_size: wgpu::BufferSize::new(BUFFERS.ray_debug_buffer.size.unwrap().into()),
                        },
                   }, 
               ],
           ],
    };

    ray_march_info
}

fn create_render_pipeline_and_bind_groups(device: &wgpu::Device,
                                   sc_desc: &wgpu::SwapChainDescriptor,
                                   shaders: &HashMap<String, wgpu::ShaderModule>,
                                   textures: &HashMap<String, Texture>,
                                   buffers: &HashMap<String, Buffer>,
                                   rpi: &RenderPipelineInfo,
                                   primitive_topology: &wgpu::PrimitiveTopology,
                                   sample_count: u32)
    -> (Vec<wgpu::BindGroup>, wgpu::RenderPipeline) {
    
    print!("    * Creating bind groups ... ");
    
    let mut bind_group_layouts: Vec<wgpu::BindGroupLayout> = Vec::new();
    let mut bind_groups: Vec<wgpu::BindGroup> = Vec::new();
    
    // Loop over all bind_groups.
    for b_group in rpi.bind_groups.iter() {
    
        let layout_entries: Vec<wgpu::BindGroupLayoutEntry>
            = b_group.into_iter().map(|x| wgpu::BindGroupLayoutEntry::new(
                x.binding,
                x.visibility,
                x.binding_type.clone(),
              )).collect();
    

           device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
               entries: Borrowed(&layout_entries),
               label: None,
        });

        let texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: Borrowed(&layout_entries),
                label: None,
            });

        let bindings: Vec<wgpu::BindGroupEntry> 
            = b_group.into_iter().map(|x| wgpu::BindGroupEntry {
                binding: x.binding,
                resource: match x.resource {
                        Resource::TextureView(tw) =>  
                            wgpu::BindingResource::TextureView(&textures.get(tw).expect(&format!("Failed to get texture {}.", tw)).view),
                        Resource::TextureSampler(ts) => 
                            wgpu::BindingResource::Sampler(&textures.get(ts).expect(&format!("Failed to get texture {}.", ts)).sampler),
                        Resource::Buffer(b) => 
                            wgpu::BindingResource::Buffer(buffers.get(b).expect(&format!("Failed to get buffer {}.", b)).buffer.slice(..)),
                }
            }).collect();
    
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &texture_bind_group_layout,
            entries: Borrowed(&bindings),
            label: None,
        });
    
        bind_group_layouts.push(texture_bind_group_layout);
        bind_groups.push(bind_group);
    }
    
    println!(" OK'");
    
    print!("    * Creating pipeline ... ");
    
      let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
          bind_group_layouts: Borrowed(&bind_group_layouts.iter().collect::<Vec<_>>()), 
          push_constant_ranges: Borrowed(&[]),
      });
    
      // Crete vertex attributes.
      let mut stride: u64 = 0;
      let mut vertex_attributes: Vec<wgpu::VertexAttributeDescriptor> = Vec::new();
      println!("rpi.input.formats.len() == {}", rpi.input_formats.len());
      for i in 0..rpi.input_formats.len() {
          vertex_attributes.push(
              wgpu::VertexAttributeDescriptor {
                  format: rpi.input_formats[i].0,
                  offset: stride,
                  shader_location: i as u32,
              }
          );
          stride = stride + rpi.input_formats[i].1;  
          println!("stride {} :: {}", i, stride);
      }
    
      let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        layout: &render_pipeline_layout,
        vertex_stage: wgpu::ProgrammableStageDescriptor {
            module: &shaders.get(rpi.vertex_shader.name).expect(&format!("Failed to get vertex shader {}.", rpi.vertex_shader.name)),
            entry_point: Borrowed("main"),
        }, 
        fragment_stage: match rpi.fragment_shader {
            None => None,
            s    => Some(wgpu::ProgrammableStageDescriptor {
                            module: &shaders.get(s.unwrap().name).expect(&format!("Failed to fragment shader {}.", s.unwrap().name)),
                            entry_point: Borrowed("main"),
                    }),
        },
        rasterization_state: Some(wgpu::RasterizationStateDescriptor {
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: wgpu::CullMode::None, // Back
            ..Default::default()
        }),
        primitive_topology: *primitive_topology, //wgpu::PrimitiveTopology::TriangleList,
        color_states: Borrowed(&[
            wgpu::ColorStateDescriptor {
                format: sc_desc.format,
                color_blend: wgpu::BlendDescriptor::REPLACE,
                alpha_blend: wgpu::BlendDescriptor::REPLACE,
                write_mask: wgpu::ColorWrite::ALL,
            },
        ]),
        //depth_stencil_state: None,
        depth_stencil_state: Some(wgpu::DepthStencilStateDescriptor {
            format: Texture::DEPTH_FORMAT,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less, // Less
            stencil_front: wgpu::StencilStateFaceDescriptor::IGNORE,
            stencil_back: wgpu::StencilStateFaceDescriptor::IGNORE,
            stencil_read_mask: 0,
            stencil_write_mask: 0,
            //stencil_read_only: false,
        }),
        vertex_state: wgpu::VertexStateDescriptor {
            index_format: wgpu::IndexFormat::Uint16,
            vertex_buffers: Borrowed(&[wgpu::VertexBufferDescriptor {
                stride: stride,
                step_mode: wgpu::InputStepMode::Vertex,
                attributes: Borrowed(&vertex_attributes),
            }]),
        },
        sample_count: sample_count,
        sample_mask: !0,
        alpha_to_coverage_enabled: false,
      });
    

    println!(" OK'");
    (bind_groups, render_pipeline)
}

fn create_compute_pipeline_and_bind_groups(device: &wgpu::Device,
                                           shaders: &HashMap<String, wgpu::ShaderModule>,
                                           textures: &HashMap<String, Texture>,
                                           buffers: &HashMap<String, Buffer>,
                                           rpi: &ComputePipelineInfo)
    -> (Vec<wgpu::BindGroup>, wgpu::ComputePipeline) {

    print!("    * Creating compute bind groups ... ");

    let mut bind_group_layouts: Vec<wgpu::BindGroupLayout> = Vec::new();
    let mut bind_groups: Vec<wgpu::BindGroup> = Vec::new();

    // Loop over all bind_groups.
    for b_group in rpi.bind_groups.iter() {

        let layout_entries: Vec<wgpu::BindGroupLayoutEntry>
            = b_group.into_iter().map(|x| wgpu::BindGroupLayoutEntry::new(
                x.binding,
                x.visibility,
                x.binding_type.clone(),
              )).collect();

        let texture_bind_group_layout =
           device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
               entries: Borrowed(&layout_entries),
               label: None,
        });

        let bindings: Vec<wgpu::BindGroupEntry> 
            = b_group.into_iter().map(|x| wgpu::BindGroupEntry {
                binding: x.binding,
                resource: match x.resource {
                        Resource::TextureView(tw) =>  
                            wgpu::BindingResource::TextureView(&textures.get(tw).expect(&format!("Failed to get texture {}.", tw)).view),
                        Resource::TextureSampler(ts) => 
                            wgpu::BindingResource::Sampler(&textures.get(ts).expect(&format!("Failed to get texture {}.", ts)).sampler),
                        Resource::Buffer(b) => 
                            wgpu::BindingResource::Buffer(buffers.get(b).expect(&format!("Failed to get buffer {}.", b)).buffer.slice(..)),
                }
            }).collect();

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &texture_bind_group_layout,
            entries: Borrowed(&bindings),
            label: None,
        });

        bind_group_layouts.push(texture_bind_group_layout);
        bind_groups.push(bind_group);
    }

    println!(" OK'");

    print!("    * Creating compute pipeline ... ");

      let compute_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
          bind_group_layouts: Borrowed(&bind_group_layouts.iter().collect::<Vec<_>>()), 
          push_constant_ranges: Borrowed(&[]),
      });

      let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
          layout: &compute_pipeline_layout,
          compute_stage: wgpu::ProgrammableStageDescriptor {
              module: &shaders.get(rpi.compute_shader.name).unwrap(),
              entry_point: Borrowed("main"),
          },
      });
    

    println!(" OK'");
    (bind_groups, compute_pipeline)
}

/// The resources for graphics.
pub struct App {
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    sc_desc: wgpu::SwapChainDescriptor,
    swap_chain: wgpu::SwapChain,
    size: winit::dpi::PhysicalSize<u32>,
    buffers: HashMap<String,Buffer>,
    textures: HashMap<String,Texture>,
    camera: Camera,
    ray_camera: RayCamera,
    camera_controller: CameraController,
    camera_uniform: CameraUniform,
    example: Example,
    ray_camera_uniform: RayCameraUniform,
    time_counter: u128,
    time_sum: u128,
    time_sum_counter: u32,
    multisampled_framebuffer: wgpu::TextureView,
    sample_count: u32,
    render_passes: HashMap<String, RenderPass>,
    compute_passes: HashMap<String, ComputePass>,
    vertex_buffer_infos: HashMap<String, VertexBufferInfo>,
    fmm_debug_state: Fmm,
    fmm_domain: FMM_Domain,
//    pool: ,
//    spawner: ,
}

use Texture;  

impl App {

    /// Initializes the project resources and returns the intance for App object. 
    pub async fn new(window: &Window) -> Self {

        let start = SystemTime::now(); 
        let time_counter = start
            .duration_since(UNIX_EPOCH)
            .expect("Could't get the time.").as_nanos();
        let time_sum: u128 = 0;
        let time_sum_counter: u32 = 0;

        let sample_count = 1;
                                                                                  
        let example = Example::TwoTriangles;

        let fmm_debug_state =  Fmm {
            boundary_points: false,
            grids: false,
            cells: false,
            triangles: false,
            aabb: false,
        };

        // Create the surface, adapter, device and the queue.
        let (surface, device, queue, size) = create_sdqs(window).await;

        // Create the swap_chain_descriptor and swap_chain.
        let (sc_desc, swap_chain) = create_swap_chain(size, &surface, &device);

        // Create framebuffer for multisampling.
        let multisampled_framebuffer = create_multisampled_framebuffer(&device, &sc_desc, sample_count);
           
        // Storage for textures. It is important to load textures before creating bind groups.
        let mut textures = HashMap::new();
        create_textures(&device, &queue, &sc_desc, &mut textures, sample_count); 

        // Create shaders.
        let shaders = create_shaders(&device);

        // Storage for buffers.
        let mut buffers = HashMap::new();
        create_vertex_buffers(&device, &mut buffers);
        
        // The camera.
        let mut camera = Camera {
            pos: (1.0, 1.0, 1.0).into(),
            view: Vector3::new(0.0, 0.0, -1.0).normalize(),
            up: cgmath::Vector3::unit_y(),
            aspect: sc_desc.width as f32 / sc_desc.height as f32,
            fov: (45.0,45.0).into(),
            znear: 0.01,
            zfar: 1000.0,
        };

        // The camera controller.
        let camera_controller = CameraController::new(0.01,0.15);

        camera.view = Vector3::new(
            camera_controller.pitch.to_radians().cos() * camera_controller.yaw.to_radians().cos(),
            camera_controller.pitch.to_radians().sin(),
            camera_controller.pitch.to_radians().cos() * camera_controller.yaw.to_radians().sin()
        ).normalize();


        let mut camera_uniform = CameraUniform::new();
        camera_uniform.update_view_proj(&camera);

        let camera_buffer = Buffer::create_buffer_from_data::<CameraUniform>(
            &device,
            &[camera_uniform],
            wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST,
            None);

        buffers.insert(BUFFERS.camera_uniform_buffer.name.to_string(), camera_buffer);

        // The ray camera.
        let ray_camera = RayCamera {
            pos: (0.0, 5.0, 0.0).into(),
            view: Vector3::new(0.0, 0.0, -1.0).normalize(),
            up: cgmath::Vector3::unit_y(),
            fov: ((45.0 as f32).to_radians(), (45.0 as f32).to_radians()).into(),
            aperture_radius: 1.0, // this is only used in path tracing.
            focal_distance: 1.0, // camera distance to the camera screen.
        };

        let mut ray_camera_uniform = RayCameraUniform::new(); 
        ray_camera_uniform.update(&ray_camera);

        let ray_camera_buffer = Buffer::create_buffer_from_data::<RayCameraUniform>(
            &device,
            &[ray_camera_uniform],
            wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST,
            None);

        buffers.insert(BUFFERS.ray_camera_uniform_buffer.name.to_string(), ray_camera_buffer);

        let mut render_passes: HashMap<String, RenderPass> = HashMap::new();
        let mut compute_passes: HashMap<String, ComputePass> = HashMap::new();
        let mut vertex_buffer_infos: HashMap<String, VertexBufferInfo> = HashMap::new();

        /* DUMMY VB INFO AND BUFFER*/

        let dummy_buffer = Buffer::create_buffer_from_data::<f32>(
            &device,
            &[],
            wgpu::BufferUsage::VERTEX,
            None
        );
        buffers.insert("dummy_buffer".to_string(), dummy_buffer);

        let dummy_vb_info = VertexBufferInfo {
            vertex_buffer_name: "dummy_buffer".to_string(),
            _index_buffer: None,
            start_index: 0,
            end_index: 0,
            instances: 1,
        };

        vertex_buffer_infos.insert("dummy_vb_info".to_string(), dummy_vb_info);

        /* TWO TRIANGLES */

        println!("Creating two_triangles pipeline and bind groups.\n");
        let two_triangles_info = create_two_triangles_info(&TEXTURES.rock.name, sample_count); 
        let (two_triangles_bind_groups, two_triangles_render_pipeline) = create_render_pipeline_and_bind_groups(
                        &device,
                        &sc_desc,
                        &shaders,
                        &textures,
                        &buffers,
                        &two_triangles_info,
                        &wgpu::PrimitiveTopology::TriangleList,
                        sample_count);

        let two_triangles_vb_info = VertexBufferInfo {
            vertex_buffer_name: "two_triangles_buffer".to_string(),
            _index_buffer: None,
            start_index: 0,
            end_index: 6,
            instances: 1,
        };

        vertex_buffer_infos.insert("two_triangles_vb_info".to_string(), two_triangles_vb_info);

        let two_triangles = RenderPass {
            pipeline: two_triangles_render_pipeline,
            bind_groups: two_triangles_bind_groups,
        };

        render_passes.insert("two_triangles_render_pass".to_string(), two_triangles);

        /* CUBE */

        println!("\nCreating vtn_render pipeline and bind groups.\n");
        let vtn_info = vtn_renderer_info(sample_count);
        let (vtn_bind_groups, vtn_render_pipeline) = create_render_pipeline_and_bind_groups(
                        &device,
                        &sc_desc,
                        &shaders,
                        &textures,
                        &buffers,
                        &vtn_info,
                        &wgpu::PrimitiveTopology::TriangleList,
                        sample_count);

        let vtn_vb_info = VertexBufferInfo {
            vertex_buffer_name: "cube_buffer".to_string(),
            _index_buffer: None,
            start_index: 0,
            end_index: 36,
            instances: 1,
        };

        vertex_buffer_infos.insert("vtn_vb_info".to_string(), vtn_vb_info);

        let vtn_render_pass = RenderPass {
            pipeline: vtn_render_pipeline,
            bind_groups: vtn_bind_groups,
        };

        render_passes.insert("vtn_render_pass".to_string(), vtn_render_pass);

        println!("");

        /************** FAST MARCHING **************/

        // let mut aabb = BBox::create_from_triangle(&triangle.a, &triangle.b, &triangle.c);
        // let mut aabb_lines = aabb.to_lines();
        // aabb.expand_to_nearest_grids(0.1);

        // let mut aabb2 = BBox::create_from_triangle(&triangle_2.a, &triangle_2.b, &triangle_2.c);
        // let mut aabb2_lines = aabb2.to_lines();
        // aabb2.expand_to_nearest_grids(0.1);

        // let mut aabb_lines_expanded = aabb.to_lines();
        // aabb_lines_expanded.extend(&aabb2.to_lines());

        // let mut fmm_a_buffer: Vec<f32> = triangle.to_f32_vec(&VertexType::vvvvnnnn()); // Vec::new();
        // fmm_a_buffer.extend(&triangle_2.to_f32_vec(&VertexType::vvvvnnnn()));

        // /* AABB */

        // let (tr, bbox, plane) = domain.initialize_from_triangle_list(&vertex_list_f32, &VertexType::vvv());

        // let fast_maching_triangle_buffer = Buffer::create_buffer_from_data::<f32>(
        //     &device,
        //     &tr,
        //     wgpu::BufferUsage::VERTEX,
        //     None
        // );

        // let fast_maching_aabb = Buffer::create_buffer_from_data::<f32>(
        //     &device,
        //     &bbox,
        //     wgpu::BufferUsage::VERTEX,
        //     None
        // );

        // buffers.insert("fmm_aabb_buffer".to_string(), fast_maching_aabb);

        // buffers.insert("fmm_triangle_buffer".to_string(), fast_maching_triangle_buffer);

        //let fmm_boundary_data = domain.boundary_points_to_vec();
        //let fmm_cell_data = domain.cells_to_vec();
        // let boundary_lines = domain.boundary_grid_to_vec();


        // let aabb_vb_info = VertexBufferInfo {
        //     vertex_buffer_name: "fmm_aabb_buffer".to_string(),
        //     _index_buffer: None,
        //     start_index: 0,
        //     end_index: aabb_lines_expanded.len() as u32 / 4,
        //     instances: 1,
        // };

        // vertex_buffer_infos.insert("aabb_vb_info".to_string(), aabb_vb_info);

        // /* BOUNDARY POINTS */

        // let fmm_boundary_points = Buffer::create_buffer_from_data::<f32>(
        //     &device,
        //     &fmm_boundary_data,
        //     wgpu::BufferUsage::VERTEX,
        //     None
        // );
        // buffers.insert(BUFFERS.fmm_points.name.to_string(), fmm_boundary_points);


        // let boundary_point_size = domain.boundary_points.points.len() as u32;

        // let fmm_boundary_vb_info = VertexBufferInfo {
        //     vertex_buffer_name: BUFFERS.fmm_points.name.to_string(),
        //     _index_buffer: None,
        //     start_index: 0,
        //     end_index: boundary_point_size,
        //     instances: 1,
        // };

        // vertex_buffer_infos.insert("fmm_boundary_vb_info".to_string(), fmm_boundary_vb_info);

        /* RAY RENDERER */

        println!("\nCreating ray renderer pipeline and bind groups.\n");
        let ray_renderer_info = ray_renderer_info(sample_count); 
        let (ray_renderer_bind_groups, ray_renderer_pipeline) = create_render_pipeline_and_bind_groups(
                        &device,
                        &sc_desc,
                        &shaders,
                        &textures,
                        &buffers,
                        &ray_renderer_info,
                        &wgpu::PrimitiveTopology::TriangleList,
                        sample_count);

        let ray_renderer_vb_info = VertexBufferInfo {
            vertex_buffer_name: "two_triangles_buffer".to_string(),
            _index_buffer: None,
            start_index: 0,
            end_index: 6,
            instances: 1,
        };

        vertex_buffer_infos.insert("ray_renderer_vb_info".to_string(), ray_renderer_vb_info);

        let ray_renderer_pass = RenderPass {
            pipeline: ray_renderer_pipeline,
            bind_groups: ray_renderer_bind_groups,
        };

        render_passes.insert("ray_renderer_pass".to_string(), ray_renderer_pass);

        println!("");

        /* MARCHING CUBES RENDERER */

        println!("\nCreating mc_render pipeline and bind groups.\n");
        let mc_renderer_info = mc_renderer_info(sample_count); 
        let (mc_render_bind_groups, mc_render_pipeline) = create_render_pipeline_and_bind_groups(
                        &device,
                        &sc_desc,
                        &shaders,
                        &textures,
                        &buffers,
                        &mc_renderer_info,
                        &wgpu::PrimitiveTopology::TriangleList,
                        sample_count);

        let mc_renderer_pass = RenderPass {
            pipeline: mc_render_pipeline,
            bind_groups: mc_render_bind_groups,
        };

        render_passes.insert("mc_renderer_pass".to_string(), mc_renderer_pass);

        println!("");

        println!("\nCreating marching cubes pipeline and bind groups.\n");
        let mc_compute_info = marching_cubes_info();
        let (mc_compute_bind_groups, mc_compute_pipeline) = create_compute_pipeline_and_bind_groups(
                        &device,
                        &shaders,
                        &textures,
                        &buffers,
                        &mc_compute_info);

        let mc_compute_pass = ComputePass {
            pipeline: mc_compute_pipeline,
            bind_groups: mc_compute_bind_groups,
            dispatch_x: 8,
            dispatch_y: 8,
            dispatch_z: 8,
        };

        compute_passes.insert("mc_compute_pass".to_string(), mc_compute_pass);

        println!("");

        println!("\nCreating volumetric ray cast (noise) pipeline and bind groups.\n");
        let volume_noise_info = ray_march_info(sample_count); // TODO: rename info function.
        let (volume_noise_bind_groups, volume_noise_compute_pipeline) = create_compute_pipeline_and_bind_groups(
                        &device,
                        &shaders,
                        &textures,
                        &buffers,
                        &volume_noise_info);
        
        let volume_noise_pass = ComputePass {
            pipeline: volume_noise_compute_pipeline,
            bind_groups: volume_noise_bind_groups,
            dispatch_x: CAMERA_RESOLUTION.0 / 8,
            dispatch_y: CAMERA_RESOLUTION.1 / 8,
            dispatch_z: 1,
        };

        compute_passes.insert("volume_noise_pass".to_string(), volume_noise_pass);
        println!("\nLaunching marching cubes.\n");

        let mut mc_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        compute_passes.get("mc_compute_pass")
                      .unwrap()
                      .execute(&mut mc_encoder);

        queue.submit(Some(mc_encoder.finish()));

        let k = &buffers.get(BUFFERS.mc_counter_buffer.name).unwrap().to_vec::<u32>(&device, &queue).await;
        let mc_vertex_count = k[0];
        let mc_renderer_vb_info = VertexBufferInfo {
            vertex_buffer_name: BUFFERS.mc_output_buffer.name.to_string(),
            _index_buffer: None,
            start_index: 0,
            end_index: mc_vertex_count,
            instances: 1,
        };
        vertex_buffer_infos.insert("mc_renderer_vb_info".to_string(), mc_renderer_vb_info);

        println!("k[0] == {}", k[0]);

        heap_test();

        /////////////////////////////////////////////////////////////////////////
        ////
        ////         FMM NEW
        ////
        ////////////////////////////////////////////////////////////////////////
        println!("\nCreating boundarypoint pipeline and bind groups.\n");
        let point_info = vvvv_camera_info(BUFFERS.camera_uniform_buffer.name, LINE_SHADERS_4PX[0].name, LINE_SHADERS_4PX[1].name, sample_count);
        let (point_groups, point_pipeline) = create_render_pipeline_and_bind_groups(
                        &device,
                        &sc_desc,
                        &shaders,
                        &textures,
                        &buffers,
                        &point_info,
                        &wgpu::PrimitiveTopology::PointList,
                        sample_count);

        let point_render_pass = RenderPass {
            pipeline: point_pipeline,
            bind_groups: point_groups,
        };

        render_passes.insert("point_render_pass".to_string(), point_render_pass);

        ////// /* AABB */

        ////// let (tr, bbox, plane) = domain.initialize_from_triangle_list(&vertex_list_f32, &VertexType::vvv());


        ////vertex_buffer_infos.insert("fmm_triangles_vb_info".to_string(), fmm_triangle_vb_info);

        // let fast_maching_aabb = Buffer::create_buffer_from_data::<f32>(
        //     &device,
        //     &bbox,
        //     wgpu::BufferUsage::VERTEX,
        //     None
        // );

        // buffers.insert("fmm_aabb_buffer".to_string(), fast_maching_aabb);

        // CREATE DOMAIN 

        let (mut mc_triangle_data, aabb): (Vec<Triangle>, BBox) = load_triangles_from_obj("bunny.obj").unwrap();
        let length = aabb.min.distance(aabb.max);
        let mut scene_aabb = BBox { min: Vector3::<f32>::new(0.0, 0.0, 0.0), max: Vector3::<f32>::new(0.0, 0.0, 0.0), }; 
        scene_aabb.expand(&(aabb.min - Vector3::<f32>::new(aabb.min.x.abs(), aabb.min.y.abs(), aabb.min.z.abs())));
        scene_aabb.expand(&(aabb.max + Vector3::<f32>::new(aabb.max.x.abs(), aabb.max.y.abs(), aabb.max.z.abs())));
        let base_position = scene_aabb.min;
        let grid_length_x = (scene_aabb.min.x - scene_aabb.max.x).abs();
        let grid_length_y = (scene_aabb.min.y - scene_aabb.max.y).abs();
        let grid_length_z = (scene_aabb.min.z - scene_aabb.max.z).abs();
        let grid_length = grid_length_x.max(grid_length_y).max(grid_length_z) / 80.0;

        // BUNNY
        
        let mut bunny_data: Vec<Vertex_vvvv_nnnn> = Vec::new();
        for tr in mc_triangle_data.iter() {
            let normal = (tr.a-tr.b).cross(tr.a-tr.b).normalize(); //ac.cross(ab).normalize();
            let normal_array = [normal.x, normal.y, normal.z, 0.0];
            let result = Vertex_vvvv_nnnn {
                position: [tr.a.x, tr.a.y, tr.a.z, 1.0],
                normal: normal_array,
            };
            bunny_data.push(result);
            let result = Vertex_vvvv_nnnn {
                position: [tr.b.x, tr.b.y, tr.b.z, 1.0],
                normal: normal_array,
            };
            bunny_data.push(result);
            let result = Vertex_vvvv_nnnn {
                position: [tr.c.x, tr.c.y, tr.c.z, 1.0],
                normal: normal_array,
            };
            bunny_data.push(result);
        }
        let fast_maching_triangle_buffer = Buffer::create_buffer_from_data::<Vertex_vvvv_nnnn>(
            &device,
            &bunny_data,
            wgpu::BufferUsage::VERTEX,
            None
        );
        buffers.insert("fmm_triangle_buffer".to_string(), fast_maching_triangle_buffer);

        let fmm_triangle_vb_info = VertexBufferInfo {
            vertex_buffer_name: "fmm_triangle_buffer".to_string(),
            _index_buffer: None,
            start_index: 0,
            end_index: bunny_data.len() as u32,
            instances: 1,
        };

        vertex_buffer_infos.insert("fmm_triangles_vb_info".to_string(), fmm_triangle_vb_info);

        let mut fmm_domain = FMM_Domain::new((80, 80, 80), base_position, grid_length, false);
        println!("NEW TESTS");

        // Create vvvc_nnnn pipelines and render passes.
        println!("CREATING CELL CUBES GROUP");
        let vvvc_nnnn_info = vvvc_nnnn_camera_info(BUFFERS.camera_uniform_buffer.name, LINE_SHADERS_VVVC_NNNN[0].name, LINE_SHADERS_VVVC_NNNN[1].name, sample_count);
        let (cell_cubes_groups, cell_cubes_pipeline) = create_render_pipeline_and_bind_groups(
                        &device,
                        &sc_desc,
                        &shaders,
                        &textures,
                        &buffers,
                        &vvvc_nnnn_info,
                        &wgpu::PrimitiveTopology::TriangleList,
                        sample_count);

        let cell_cubes_render_pass = RenderPass {
            pipeline: cell_cubes_pipeline,
            bind_groups: cell_cubes_groups,
        };

        render_passes.insert("cell_cubes_render_pass".to_string(), cell_cubes_render_pass);


        //// vertex_buffer_infos.insert("fmm_cell_vb_info".to_string(), fmm_cell_vb_info);

        /* BOUNDARY LINES */

        ////let boundary_lines = fmm_domain.boundary_lines();

        ////let boundary_lines_buffer = Buffer::create_buffer_from_data::<Vertex_vvvc>(
        ////    &device,
        ////    &boundary_lines,
        ////    wgpu::BufferUsage::VERTEX,
        ////    None
        ////);
        ////buffers.insert(BUFFERS.fmm_boundary_lines.name.to_string(), boundary_lines_buffer);


        let boundary_line_info = vvvv_camera_info(BUFFERS.camera_uniform_buffer.name, LINE_SHADERS[0].name, LINE_SHADERS[1].name, sample_count);
        let (boundary_line_groups, boundary_line_pipeline) = create_render_pipeline_and_bind_groups(
                        &device,
                        &sc_desc,
                        &shaders,
                        &textures,
                        &buffers,
                        &boundary_line_info,
                        &wgpu::PrimitiveTopology::LineList,
                        sample_count);

        ////println!("boundary lines.len() == {}", boundary_lines.len());

        ////let boundary_line_vb_info = VertexBufferInfo {
        ////    vertex_buffer_name: BUFFERS.fmm_boundary_lines.name.to_string(),
        ////    _index_buffer: None,
        ////    start_index: 0,
        ////    end_index: boundary_lines.len() as u32,
        ////    instances: 1,
        ////};

        let boundary_line_render_pass = RenderPass {
            pipeline: boundary_line_pipeline,
            bind_groups: boundary_line_groups,
        };

        render_passes.insert("boundary_line_render_pass".to_string(), boundary_line_render_pass);

        ////vertex_buffer_infos.insert("boundary_line_vb_info".to_string(), boundary_line_vb_info);

        // The number of marching cubes vertices obtained from mc-algorithm.
        //let k = &buffers.get(BUFFERS.mc_counter_buffer.name).unwrap().to_vec::<u32>(&device, &queue).await;
        //let mc_vertex_count = k[0];

        // let marching_cubes_data = &buffers.get(BUFFERS.mc_output_buffer.name).unwrap().to_vec::<Vertex_vvvv_nnnn>(&device, &queue).await;
        let marching_cubes_data = &buffers.get(BUFFERS.mc_output_buffer.name).unwrap().to_vec::<Vertex_vvvv_nnnn>(&device, &queue).await;

        //for i in 0..mc_vertex_count*8 {
        //    println!("{}", marching_cubes_data[i as usize]);

        //    // println!("Vertex_vvvv_nnnn [({}, {}, {}, {}), {}, {}, {}, {})].",
        //    //     mcd.position[0], mcd.position[1], mcd.position[2], mcd.position[3], 
        //    //     mcd.normal[0], mcd.normal[1], mcd.normal[2], mcd.normal[3]); 
        //}

        //// let mut mc_triangle_data: Vec<Triangle> = Vec::new();
        //// for i in 0..(mc_vertex_count/3) {
        ////     let offset = (i*3) as usize;
        ////     let a = Vector3::<f32>::new(marching_cubes_data[offset].position[0]  , marching_cubes_data[offset].position[1],     marching_cubes_data[offset].position[2]); 
        ////     let b = Vector3::<f32>::new(marching_cubes_data[offset+1].position[0], marching_cubes_data[offset + 1].position[1], marching_cubes_data[offset + 1].position[2]); 
        ////     let c = Vector3::<f32>::new(marching_cubes_data[offset+2].position[0], marching_cubes_data[offset + 2].position[1], marching_cubes_data[offset + 2].position[2]); 
        ////     mc_triangle_data.push(Triangle {a: a, b: b, c: c, });
        //// }

        let (cubes, lines) = fmm_domain.add_triangles(&mc_triangle_data);

        // Triangles & lines
        let nearest_buffer = buffers.get("nearest_point_buffer"); 

        //////let two_triangles = 
        //////    Buffer::create_buffer_from_data::<f32>(
        //////    device,
        //////    // gl_Position     |    point_pos
        //////    &[-1.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
        //////       1.0, -1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0,
        //////       1.0,  1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0,
        //////       1.0,  1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0,
        //////      -1.0,  1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
        //////      -1.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
        //////    ],
        //////    wgpu::BufferUsage::VERTEX | wgpu::BufferUsage::COPY_SRC,
        //////    None
        //////    );

    //buffers.insert("two_triangles_buffer".to_string(), two_triangles);
    //    // Nearest point buffer.
        let nearest_point_buffer = Buffer::create_buffer_from_data::<Vertex_vvvc_nnnn>(
            &device,
            &cubes,
            wgpu::BufferUsage::VERTEX | wgpu::BufferUsage::COPY_DST,
            None
        );

        buffers.insert("nearest_point_buffer".to_string(), nearest_point_buffer);

        let fmm_nearest_point = VertexBufferInfo {
            vertex_buffer_name: "nearest_point_buffer".to_string(),
            _index_buffer: None,
            start_index: 0,
            end_index: cubes.len() as u32, // CHECK! // closest_data.len() as u32 / 4,
            instances: 1,
        };
        println!("cubes.len() == {}", cubes.len());
        vertex_buffer_infos.insert("fmm_nearest_vb_info".to_string(), fmm_nearest_point);


        // Line(s) between nearest cell point.
        let nearest_cell_line = Buffer::create_buffer_from_data::<Vertex_vvvc>(
            &device,
            &lines,
            wgpu::BufferUsage::VERTEX | wgpu::BufferUsage::COPY_DST,
            None
        );

        buffers.insert("nearest_cell_line_buffer".to_string(), nearest_cell_line);

        let fmm_cell_line_vb_info = VertexBufferInfo {
            vertex_buffer_name: "nearest_cell_line_buffer".to_string(),
            _index_buffer: None,
            start_index: 0,
            end_index: lines.len() as u32,
            instances: 1,
        };
        println!("lines.len() == {}", lines.len());
        vertex_buffer_infos.insert("fmm_cell_line_vb_info".to_string(), fmm_cell_line_vb_info);

        //let triangle = Triangle {
        //    a: Vector3::<f32>::new(1.46, 1.61, 1.59),
        //    b: Vector3::<f32>::new(1.66, 1.43, 1.3),
        //    c: Vector3::<f32>::new(1.2999, 1.0, 0.9999)
        //};
        //let cell_color = encode_rgba_u32(100, 180, 88, 255); 

        //let (mut closest_data2, mut closest_line2) = self.fmm_domain.add_triangle(&triangle);
        //// println!("closest_data2.len() == {}, closest_line2.len() == {}", closest_data2.len(), closest_line2.len());

        //let camera_cell = create_cube_triangles(self.camera.pos + self.camera.view * 0.20, 0.006, cell_color);
        //closest_data.extend(camera_cell);
        //closest_data.extend(closest_data2);
        //closest_line.extend(closest_line2);
        //// Nearest point buffer.
        //let nearest_point_buffer = Buffer::create_buffer(
        //    &self.device,
        //    (std::mem::size_of::<Vertex_vvvc_nnnn>() * mc_vertex_count * 8 * 4) as u64,
        //    wgpu::BufferUsage::VERTEX | wgpu::BufferUsage::COPY_DST,
        //    None
        //);

        //self.buffers.insert("nearest_point_buffer".to_string(), nearest_point_buffer);

        //let fmm_nearest_point = VertexBufferInfo {
        //    vertex_buffer_name: "nearest_point_buffer".to_string(),
        //    _index_buffer: None,
        //    start_index: 0,
        //    end_index: 0, // closest_data.len() as u32 / 4,
        //    instances: 1,
        //};
        //self.vertex_buffer_infos.insert("fmm_nearest_vb_info".to_string(), fmm_nearest_point);
        //self.queue.write_buffer(
        //    &buffer.buffer,
        //    0,
        //    bytemuck::cast_slice(&closest_data)
        //    //unsafe {std::slice::from_raw_parts(closest_data.as_ptr() as *const _, closest_data.len() * std::mem::size_of::<Vertex_vvvc_nnnn>()) },
        //);
        //self.queue.write_buffer(
        //    &self.buffers.get("nearest_cell_line_buffer").unwrap().buffer,
        //    0,
        //    bytemuck::cast_slice(&closest_line)
        //    //unsafe {std::slice::from_raw_parts(closest_data.as_ptr() as *const _, closest_data.len() * std::mem::size_of::<Vertex_vvvc_nnnn>()) },
        //);
        // println!("closest data len == {}", closest_data.len());
        
        // The vertex buffer info for camera cell and closest point cubes.
        //let fmm_nearest_point = VertexBufferInfo {
        //    vertex_buffer_name: "nearest_point_buffer".to_string(),
        //    _index_buffer: None,
        //    start_index: 0,
        //    end_index: closest_data.len() as u32,
        //    instances: 1,
        //};
        //self.vertex_buffer_infos.insert("fmm_nearest_vb_info".to_string(), fmm_nearest_point);

        //// The vertex buffer info for line between the camera cell and closest point.
        //let fmm_nearest_cell_line = VertexBufferInfo {
        //    vertex_buffer_name: "nearest_cell_line_buffer".to_string(),
        //    _index_buffer: None,
        //    start_index: 0,
        //    end_index: closest_line.len() as u32,
        //    instances: 1,
        //};
        //self.vertex_buffer_infos.insert("fmm_cell_line_vb_info".to_string(), fmm_nearest_cell_line);


        Self {
            surface,
            device,
            queue,
            sc_desc,
            swap_chain,
            size,
            buffers,
            camera,
            ray_camera, 
            camera_controller,
            camera_uniform,
            textures,
            example,
            ray_camera_uniform,
            time_counter,
            time_sum,
            time_sum_counter,
            multisampled_framebuffer, 
            sample_count,
            render_passes,
            compute_passes,
            vertex_buffer_infos,
            fmm_debug_state,
            fmm_domain,
        }
    } // new(...

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        self.size = new_size;
        self.sc_desc.width = new_size.width;
        self.sc_desc.height = new_size.height;
        self.swap_chain = self.device.create_swap_chain(&self.surface, &self.sc_desc);

        let depth_texture = Texture::create_depth_texture(&self.device, &self.sc_desc, Some(Borrowed("depth-texture")));
        self.textures.insert(TEXTURES.depth.name.to_string(), depth_texture);
    }

    pub fn input(&mut self, event: &WindowEvent) -> bool {
        let result = self.camera_controller.process_events(event);
        if result == true {
            self.camera_controller.update_camera(&mut self.camera);
            self.camera_controller.update_ray_camera(&mut self.ray_camera);
            self.update();
            //println!("({}, {}, {})",self.camera.pos.x, self.camera.pos.y, self.camera.pos.z);
        }
        result
    }

    pub fn update(&mut self) {

        self.camera_uniform.update_view_proj(&self.camera);
        self.ray_camera_uniform.update(&self.ray_camera);

        // TODO: Create a method for this in Buffer.
        self.queue.write_buffer(
            &self.buffers.get(BUFFERS.camera_uniform_buffer.name).unwrap().buffer,
            0,
            bytemuck::cast_slice(&[self.camera_uniform])
        );

        self.queue.write_buffer(
            &self.buffers.get(BUFFERS.ray_camera_uniform_buffer.name).unwrap().buffer,
            0,
            bytemuck::cast_slice(&[self.ray_camera_uniform])
        );

        ////let nearest_buffer = self.buffers.get("nearest_point_buffer"); 

        ////// Create neares_point_buffer is it's not created yet.
        ////match nearest_buffer {
        ////    None => { 

        ////        // Nearest point buffer.
        ////        let nearest_point_buffer = Buffer::create_buffer(
        ////            &self.device,
        ////            (std::mem::size_of::<Vertex_vvvc_nnnn>() * 3 * 1800) as u64,
        ////            wgpu::BufferUsage::VERTEX | wgpu::BufferUsage::COPY_DST,
        ////            None
        ////        );

        ////        self.buffers.insert("nearest_point_buffer".to_string(), nearest_point_buffer);

        ////        let fmm_nearest_point = VertexBufferInfo {
        ////            vertex_buffer_name: "nearest_point_buffer".to_string(),
        ////            _index_buffer: None,
        ////            start_index: 0,
        ////            end_index: 0, // closest_data.len() as u32 / 4,
        ////            instances: 1,
        ////        };
        ////        self.vertex_buffer_infos.insert("fmm_nearest_vb_info".to_string(), fmm_nearest_point);


        ////        // Line(s) between nearest cell point.
        ////        let nearest_cell_line = Buffer::create_buffer(
        ////            &self.device,
        ////            (std::mem::size_of::<Vertex_vvvc>() * 3 * 1800) as u64,
        ////            wgpu::BufferUsage::VERTEX | wgpu::BufferUsage::COPY_DST,
        ////            None
        ////        );

        ////        self.buffers.insert("nearest_cell_line_buffer".to_string(), nearest_cell_line);

        ////        let fmm_cell_line_vb_info = VertexBufferInfo {
        ////            vertex_buffer_name: "nearest_point_buffer".to_string(),
        ////            _index_buffer: None,
        ////            start_index: 0,
        ////            end_index: 0,
        ////            instances: 1,
        ////        };
        ////        self.vertex_buffer_infos.insert("fmm_cell_line_vb_info".to_string(), fmm_cell_line_vb_info);
        ////    }

        ////    // Update closest cell points and line between them.
        ////    Some(buffer) => { 

        ////        let triangle = Triangle {
        ////            a: Vector3::<f32>::new(1.46, 1.61, 1.59),
        ////            b: Vector3::<f32>::new(1.66, 1.43, 1.3),
        ////            c: Vector3::<f32>::new(1.2999, 1.0, 0.9999)
        ////        };
        ////        let (mut closest_data, mut closest_line) = self.fmm_domain.get_neighbor_grid_points(&(self.camera.pos + self.camera.view * 0.20));
        ////        // // println!("colusesttt data found!!!!");
        ////        let cell_color = encode_rgba_u32(100, 180, 88, 255); 

        ////        let (mut closest_data2, mut closest_line2) = self.fmm_domain.add_triangle(&triangle);
        ////        // println!("closest_data2.len() == {}, closest_line2.len() == {}", closest_data2.len(), closest_line2.len());

        ////        let camera_cell = create_cube_triangles(self.camera.pos + self.camera.view * 0.20, 0.006, cell_color);
        ////        closest_data.extend(camera_cell);
        ////        closest_data.extend(closest_data2);
        ////        closest_line.extend(closest_line2);
        ////        self.queue.write_buffer(
        ////            &buffer.buffer,
        ////            0,
        ////            bytemuck::cast_slice(&closest_data)
        ////            //unsafe {std::slice::from_raw_parts(closest_data.as_ptr() as *const _, closest_data.len() * std::mem::size_of::<Vertex_vvvc_nnnn>()) },
        ////        );
        ////        self.queue.write_buffer(
        ////            &self.buffers.get("nearest_cell_line_buffer").unwrap().buffer,
        ////            0,
        ////            bytemuck::cast_slice(&closest_line)
        ////            //unsafe {std::slice::from_raw_parts(closest_data.as_ptr() as *const _, closest_data.len() * std::mem::size_of::<Vertex_vvvc_nnnn>()) },
        ////        );
        ////        // println!("closest data len == {}", closest_data.len());
        ////        
        ////        // The vertex buffer info for camera cell and closest point cubes.
        ////        let fmm_nearest_point = VertexBufferInfo {
        ////            vertex_buffer_name: "nearest_point_buffer".to_string(),
        ////            _index_buffer: None,
        ////            start_index: 0,
        ////            end_index: closest_data.len() as u32,
        ////            instances: 1,
        ////        };
        ////        self.vertex_buffer_infos.insert("fmm_nearest_vb_info".to_string(), fmm_nearest_point);

        ////        // The vertex buffer info for line between the camera cell and closest point.
        ////        let fmm_nearest_cell_line = VertexBufferInfo {
        ////            vertex_buffer_name: "nearest_cell_line_buffer".to_string(),
        ////            _index_buffer: None,
        ////            start_index: 0,
        ////            end_index: closest_line.len() as u32,
        ////            instances: 1,
        ////        };
        ////        self.vertex_buffer_infos.insert("fmm_cell_line_vb_info".to_string(), fmm_nearest_cell_line);
        ////    }
        ////}
    }

    pub fn render(&mut self, window: &winit::window::Window) {
        let start = SystemTime::now();
        let time_now = start
            .duration_since(UNIX_EPOCH)
            .expect("Could't get the time.").as_nanos();
        let time_delta = time_now - self.time_counter;
        self.time_counter = time_now;
        
        self.time_sum = self.time_sum + time_delta;
        self.time_sum_counter = self.time_sum_counter + 1; 

        if self.time_sum_counter >= 10 {
            let fps = 1000000000 / (self.time_sum / self.time_sum_counter as u128);
            self.time_sum = 0;
            self.time_sum_counter = 0;
            let fps_text = fps.to_string(); 
            window.set_title(&fps_text);
        }

        //let frame = match self.swap_chain.get_current_frame() {
        let frame = match self.swap_chain.get_current_frame() {
            Ok(frame) => { frame.output },    
            Err(_) => {
                println!("FAILED");
                self.swap_chain = self.device.create_swap_chain(&self.surface, &self.sc_desc);
                self.swap_chain.get_current_frame().expect("Failed to acquire next swap chain texture").output
            },
        };

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(Borrowed("Render Encoder")),
        });

        match self.example {

                Example::VolumetricNoise => {

                    self.compute_passes.get("volume_noise_pass")
                    .unwrap()
                    .execute(&mut encoder);

                    encoder.copy_buffer_to_texture(
                        wgpu::BufferCopyView {
                            buffer: &self.buffers.get(BUFFERS.ray_march_output_buffer.name).unwrap().buffer,
                            layout: wgpu::TextureDataLayout {
                                offset: 0,
                                bytes_per_row: CAMERA_RESOLUTION.0 * 4,
                                rows_per_image: CAMERA_RESOLUTION.1,
                            },
                        },
                        wgpu::TextureCopyView{
                            texture: &self.textures.get(TEXTURES.ray_texture.name).unwrap().texture,
                            mip_level: 0,
                            origin: wgpu::Origin3d::ZERO,
                        },
                        wgpu::Extent3d {
                            width: CAMERA_RESOLUTION.0,
                            height: CAMERA_RESOLUTION.1,
                            depth: 1,
                    });
                },

                _ => {}
        }

        {
            match self.example {
                Example::TwoTriangles => {
                    let vb_info = self.vertex_buffer_infos.get("two_triangles_vb_info").expect("Could not find vertex buffer info");
                    self.render_passes.get("two_triangles_render_pass")
                    .unwrap()
                    .execute(&mut encoder, &frame, &self.multisampled_framebuffer, &self.textures, &self.buffers, &vb_info, self.sample_count, true);
                },
                Example::Cube => {
                    let vb_info = self.vertex_buffer_infos.get("vtn_vb_info").expect("Could not find vertex buffer info");
                    self.render_passes.get("vtn_render_pass")
                    .unwrap()
                    .execute(&mut encoder, &frame, &self.multisampled_framebuffer, &self.textures, &self.buffers, &vb_info, self.sample_count, true);
                },
                Example::Mc => {
                    let rp = self.vertex_buffer_infos.get("mc_renderer_vb_info").unwrap();
                    self.render_passes.get("mc_renderer_pass")
                    .unwrap()
                    .execute(&mut encoder, &frame, &self.multisampled_framebuffer, &self.textures, &self.buffers, &rp, self.sample_count, true);
                }
                Example::FmmGhostPoints => {

                    let mut clear = true;
                    
                    if self.fmm_debug_state.boundary_points == true {
                        let rp = self.vertex_buffer_infos.get("fmm_boundary_vb_info").unwrap();
                        self.render_passes.get("point_render_pass")
                        .unwrap()
                        .execute(&mut encoder, &frame, &self.multisampled_framebuffer, &self.textures, &self.buffers, &rp, self.sample_count, clear);
                        clear = false;
                    }

                    if self.fmm_debug_state.cells == true {
                        let rp_cell = self.vertex_buffer_infos.get("fmm_cell_vb_info").unwrap();
                        self.render_passes.get("cell_cubes_render_pass")
                        .unwrap()
                        .execute(&mut encoder, &frame, &self.multisampled_framebuffer, &self.textures, &self.buffers, &rp_cell, self.sample_count, clear);
                        clear = false;
                    }

                    if self.fmm_debug_state.grids == true {
                        let rp_boundary_lines = self.vertex_buffer_infos.get("boundary_line_vb_info").unwrap();
                        self.render_passes.get("boundary_line_render_pass")
                        .unwrap()
                        .execute(&mut encoder, &frame, &self.multisampled_framebuffer, &self.textures, &self.buffers, &rp_boundary_lines, self.sample_count, clear);
                        clear = false;
                    }
                                     
                    if self.fmm_debug_state.triangles == true {
                        // let rp = self.vertex_buffer_infos.get("mc_renderer_vb_info").unwrap();
                        // self.render_passes.get("mc_renderer_pass")
                        // .unwrap()
                        // .execute(&mut encoder, &frame, &self.multisampled_framebuffer, &self.textures, &self.buffers, &rp, self.sample_count, true);
                        // clear = false;
                        let rp_triangles = self.vertex_buffer_infos.get(&"fmm_triangles_vb_info".to_string()).unwrap();
                        self.render_passes.get("mc_renderer_pass")
                        .unwrap()
                        .execute(&mut encoder, &frame, &self.multisampled_framebuffer, &self.textures, &self.buffers, &rp_triangles, self.sample_count, clear);
                        clear = false;
                    }
                                     
                    if self.fmm_debug_state.aabb == true {
                        let rp_aabb = self.vertex_buffer_infos.get(&"aabb_vb_info".to_string()).unwrap();
                        self.render_passes.get("boundary_line_render_pass")
                        .unwrap()
                        .execute(&mut encoder, &frame, &self.multisampled_framebuffer, &self.textures, &self.buffers, &rp_aabb, self.sample_count, clear);
                        clear = false;
                    }

                    let rp_closest_point = self.vertex_buffer_infos.get(&"fmm_nearest_vb_info".to_string()).unwrap();
                    self.render_passes.get("cell_cubes_render_pass")
                    .unwrap()
                    .execute(&mut encoder, &frame, &self.multisampled_framebuffer, &self.textures, &self.buffers, &rp_closest_point, self.sample_count, clear);
                    clear = false;

                    let rp_closest_line = self.vertex_buffer_infos.get(&"fmm_cell_line_vb_info".to_string()).unwrap();
                    self.render_passes.get("boundary_line_render_pass") 
                    .unwrap()
                    .execute(&mut encoder, &frame, &self.multisampled_framebuffer, &self.textures, &self.buffers, &rp_closest_line, self.sample_count, clear);
                    clear = false;

                    // All above all disabled. We must draw something or application will crash.
                    // TODO: create some dummy draw prosedure. 
                    if clear == true {
                        let dummy = self.vertex_buffer_infos.get("dummy_vb_info").unwrap();
                        self.render_passes.get("boundary_line_render_pass")
                        .unwrap()
                        .execute(&mut encoder, &frame, &self.multisampled_framebuffer, &self.textures, &self.buffers, &dummy, self.sample_count, clear);
                    }
                },
                Example::CellPoints => {
                    let rp_cell = self.vertex_buffer_infos.get("fmm_cell_vb_info").unwrap();
                    self.render_passes.get("cell_point_render_pass")
                    .unwrap()
                    .execute(&mut encoder, &frame, &self.multisampled_framebuffer, &self.textures, &self.buffers, &rp_cell, self.sample_count, true);
                },
                Example::VolumetricNoise => {
                    let rp = self.vertex_buffer_infos.get("two_triangles_vb_info").unwrap();
                    self.render_passes.get("ray_renderer_pass")
                    .unwrap()
                    .execute(&mut encoder, &frame, &self.multisampled_framebuffer, &self.textures, &self.buffers, &rp, self.sample_count, true);
                },
            }
        }

        self.queue.submit(Some(encoder.finish()));
    }
}

fn create_multisampled_framebuffer(
    device: &wgpu::Device,
    sc_desc: &wgpu::SwapChainDescriptor,
    sample_count: u32,
) -> wgpu::TextureView {
    let multisampled_texture_extent = wgpu::Extent3d {
        width: sc_desc.width,
        height: sc_desc.height,
        depth: 1,
    };
    let multisampled_frame_descriptor = &wgpu::TextureDescriptor {
        size: multisampled_texture_extent,
        mip_level_count: 1,
        sample_count: sample_count,
        dimension: wgpu::TextureDimension::D2,
        format: sc_desc.format,
        usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT,
        label: None,
    };

    device
        .create_texture(multisampled_frame_descriptor)
        .create_default_view()
}

/// Load shaders.
fn create_shaders(device: &wgpu::Device) -> HashMap<String, wgpu::ShaderModule> {

    println!("\nCreating shaders.\n");
    let mut shaders = HashMap::new();

    print!("    * Creating 'two_triangles_vert' shader module from file 'two_triangles_vert.spv'");
    shaders.insert("two_triangles_vert".to_string(), device.create_shader_module(wgpu::include_spirv!("two_triangles.vert.spv")));
    println!(" ... OK'");

    print!("    * Creating 'two_triangles_frag' shader module from file 'two_triangles_frag.spv'");
    shaders.insert("two_triangles_frag".to_string(), device.create_shader_module(wgpu::include_spirv!("two_triangles.frag.spv")));
    println!(" ... OK'");

    print!("    * Creating '{}' shader module from file '{}'", VTN_SHADERS[0].name, VTN_SHADERS[0].source_file);
    shaders.insert(VTN_SHADERS[0].name.to_string(), device.create_shader_module(wgpu::include_spirv!("vtn_renderer.vert.spv")));
    println!(" ... OK'");

    print!("    * Creating '{}' shader module from file '{}'",VTN_SHADERS[1].name, VTN_SHADERS[1].source_file);
    shaders.insert(VTN_SHADERS[1].name.to_string(), device.create_shader_module(wgpu::include_spirv!("vtn_renderer.frag.spv")));
    println!(" ... OK'");

    print!("    * Creating '{}' shader module from file '{}'", MC_RENDER_SHADERS[0].name, MC_RENDER_SHADERS[0].source_file);
    shaders.insert(MC_RENDER_SHADERS[0].name.to_string(), device.create_shader_module(wgpu::include_spirv!("mc_render.vert.spv")));
    println!(" ... OK'");

    print!("    * Creating '{}' shader module from file '{}'",MC_RENDER_SHADERS[1].name, MC_RENDER_SHADERS[1].source_file);
    shaders.insert(MC_RENDER_SHADERS[1].name.to_string(), device.create_shader_module(wgpu::include_spirv!("mc_render.frag.spv")));
    println!(" ... OK'");

    print!("    * Creating '{}' shader module from file '{}'",MARCHING_CUBES_SHADER.name, MARCHING_CUBES_SHADER.source_file);
    shaders.insert(MARCHING_CUBES_SHADER.name.to_string(), device.create_shader_module(wgpu::include_spirv!("mc.comp.spv")));
    println!(" ... OK'");

    print!("    * Creating '{}' shader module from file '{}'",RAY_MARCH_SHADER.name, RAY_MARCH_SHADER.source_file);
    shaders.insert(RAY_MARCH_SHADER.name.to_string(), device.create_shader_module(wgpu::include_spirv!("ray.comp.spv")));
    println!(" ... OK'");

    print!("    * Creating '{}' shader module from file '{}'",SPHERE_TRACER_SHADER.name, SPHERE_TRACER_SHADER.source_file);
    shaders.insert(SPHERE_TRACER_SHADER.name.to_string(), device.create_shader_module(wgpu::include_spirv!("sphere_tracer.comp.spv")));
    println!(" ... OK'");

    print!("    * Creating '{}' shader module from file '{}'",GENERATE_3D_SHADER.name, GENERATE_3D_SHADER.source_file);
    shaders.insert(GENERATE_3D_SHADER.name.to_string(), device.create_shader_module(wgpu::include_spirv!("generate_noise3d.comp.spv")));
    println!(" ... OK'");

    print!("    * Creating '{}' shader module from file '{}'",BITONIC_SHADER.name, BITONIC_SHADER.source_file);
    shaders.insert(BITONIC_SHADER.name.to_string(), device.create_shader_module(wgpu::include_spirv!("local_sort.comp.spv")));
    println!(" ... OK'");

    print!("    * Creating '{}' shader module from file '{}'",LINE_SHADERS[0].name, LINE_SHADERS[0].source_file);
    shaders.insert(LINE_SHADERS[0].name.to_string(), device.create_shader_module(wgpu::include_spirv!("line.vert.spv")));
    println!(" ... OK'");

    print!("    * Creating '{}' shader module from file '{}'",LINE_SHADERS[1].name, LINE_SHADERS[1].source_file);
    shaders.insert(LINE_SHADERS[1].name.to_string(), device.create_shader_module(wgpu::include_spirv!("line.frag.spv")));
    println!(" ... OK'");

    print!("    * Creating '{}' shader module from file '{}'",LINE_SHADERS_4PX[0].name, LINE_SHADERS_4PX[0].source_file);
    shaders.insert(LINE_SHADERS_4PX[0].name.to_string(), device.create_shader_module(wgpu::include_spirv!("line_4px.vert.spv")));
    println!(" ... OK'");

    print!("    * Creating '{}' shader module from file '{}'",LINE_SHADERS_4PX[1].name, LINE_SHADERS_4PX[1].source_file);
    shaders.insert(LINE_SHADERS_4PX[1].name.to_string(), device.create_shader_module(wgpu::include_spirv!("line_4px.frag.spv")));
    println!(" ... OK'");

    print!("    * Creating '{}' shader module from file '{}'",LINE_SHADERS_VN[0].name, LINE_SHADERS_VN[0].source_file);
    shaders.insert(LINE_SHADERS_VN[0].name.to_string(), device.create_shader_module(wgpu::include_spirv!("line_vn.vert.spv")));
    println!(" ... OK'");

    print!("    * Creating '{}' shader module from file '{}'",LINE_SHADERS_VN[1].name, LINE_SHADERS_VN[1].source_file);
    shaders.insert(LINE_SHADERS_VN[1].name.to_string(), device.create_shader_module(wgpu::include_spirv!("line_vn.frag.spv")));
    println!(" ... OK'");

    print!("    * Creating '{}' shader module from file '{}'",RADIX_SHADER.name, RADIX_SHADER.source_file);
    shaders.insert(RADIX_SHADER.name.to_string(), device.create_shader_module(wgpu::include_spirv!("radix.comp.spv")));
    println!(" ... OK'");

    print!("    * Creating '{}' shader module from file '{}'",LINE_SHADERS_VVVC_NNNN[0].name, LINE_SHADERS_VVVC_NNNN[0].source_file);
    shaders.insert(LINE_SHADERS_VVVC_NNNN[0].name.to_string(), device.create_shader_module(wgpu::include_spirv!("cube_instanced.vert.spv")));
    println!(" ... OK'");

    print!("    * Creating '{}' shader module from file '{}'",LINE_SHADERS_VVVC_NNNN[1].name, LINE_SHADERS_VVVC_NNNN[1].source_file);
    shaders.insert(LINE_SHADERS_VVVC_NNNN[1].name.to_string(), device.create_shader_module(wgpu::include_spirv!("cube_instanced.frag.spv")));
    println!(" ... OK'");

    println!("\nShader created!\n");
    shaders
}

// TODO: separate each buffer creation.
fn create_vertex_buffers(device: &wgpu::Device, buffers: &mut HashMap::<String, Buffer>)  {

    println!("\nCreating buffers.\n");

    print!("    * Creating cube buffer as 'cube_buffer'");

    // The Cube.
    let vertex_data = create_cube();
    let cube = Buffer::create_buffer_from_data::<f32>(
        device,
        &vertex_data,
        wgpu::BufferUsage::VERTEX,
        None);

    buffers.insert("cube_buffer".to_string(), cube);

    println!(" ... OK'");

    print!("    * Creating two_triangles buffer as 'two_triangles_buffer'");

    // 2-triangles.

    let two_triangles = 
        Buffer::create_buffer_from_data::<f32>(
        device,
        // gl_Position     |    point_pos
        &[-1.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
           1.0, -1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0,
           1.0,  1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0,
           1.0,  1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0,
          -1.0,  1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
          -1.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
        ],
        wgpu::BufferUsage::VERTEX | wgpu::BufferUsage::COPY_SRC,
        None
        );

    buffers.insert("two_triangles_buffer".to_string(), two_triangles);

    println!(" ... OK'");

    print!("    * Creating marching cubes output buffer as '{}'", BUFFERS.mc_output_buffer.name);

    let marching_cubes_output = Buffer::create_buffer_from_data::<f32>(
        device,
        &vec![0 as f32 ; BUFFERS.mc_output_buffer.size.unwrap() as usize / 4],
        wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_DST |wgpu::BufferUsage::COPY_SRC | wgpu::BufferUsage::VERTEX,
        None
    );

    buffers.insert(BUFFERS.mc_output_buffer.name.to_string(), marching_cubes_output);

    println!(" ... OK");

    print!("    * Creating marching cubes counter buffer as '{}'", BUFFERS.mc_counter_buffer.name);

    let marching_cubes_counter = Buffer::create_buffer_from_data::<u32>(
        device,
        &[0 as u32],
        wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_DST |wgpu::BufferUsage::COPY_SRC,
        None
    );

    buffers.insert(BUFFERS.mc_counter_buffer.name.to_string(), marching_cubes_counter);

    println!(" ... OK");

    print!("    * Creating marching cubes uniform buffer as '{}'", BUFFERS.mc_uniform_buffer.name);

    let mc_u_data = Mc_uniform_data {
        base_position: cgmath::Vector4::new(1.0, 1.0, 1.0, 1.0),
        isovalue: 0.0,
        cube_length: 0.1,
        joopajoo: 0.0,
        joopajoo2: 0.0,
    };

    let marching_cubes_uniform_buffer = Buffer::create_buffer_from_data::<Mc_uniform_data>(
        device,
        &[mc_u_data],
        wgpu::BufferUsage::COPY_DST |wgpu::BufferUsage::UNIFORM,
        None
    );

    buffers.insert(BUFFERS.mc_uniform_buffer.name.to_string(), marching_cubes_uniform_buffer);

    println!(" ... OK'");

    print!("    * Creating ray march output buffer as '{}'", BUFFERS.ray_march_output_buffer.name);

    let ray_march_output = Buffer::create_buffer_from_data::<u32>(
        device,
        &vec![0 as u32 ; (BUFFERS.ray_march_output_buffer.size.unwrap() / 4) as usize],
        wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_DST | wgpu::BufferUsage::COPY_SRC,
        None
    );

    buffers.insert(BUFFERS.ray_march_output_buffer.name.to_string(), ray_march_output);
    println!(" ... OK'");
    
    print!("    * Creating ray march output buffer as '{}'", BUFFERS.ray_debug_buffer.name);

    let ray_march_debug = Buffer::create_buffer_from_data::<u32>(
        device,
        &vec![0 as u32 ; BUFFERS.ray_debug_buffer.size.unwrap() as usize / 4],
        wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_DST | wgpu::BufferUsage::COPY_SRC,
        None
    );

    buffers.insert(BUFFERS.ray_debug_buffer.name.to_string(), ray_march_debug);

    println!(" ... OK'");

    println!("");
}


async fn create_sdqs(window: &winit::window::Window) -> (wgpu::Surface, wgpu::Device, wgpu::Queue, winit::dpi::PhysicalSize<u32>) {

        // Get the size of the window.
        let size = window.inner_size();

        let instance = wgpu::Instance::new(wgpu::BackendBit::PRIMARY);

        // Create the surface.
        let surface = unsafe { instance.create_surface(window) };

        let needed_features = wgpu::Features::empty();

        // Create the adapter.
        let adapter = instance.request_adapter(
            &wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance, // Default
                compatible_surface: Some(&surface),
            },
        )
        .await
        .unwrap();

        let adapter_features = adapter.features();

        // TODO: check what this mean.
        let trace_dir = std::env::var("WGPU_TRACE");
        let (device, queue) = adapter.request_device(
             &wgpu::DeviceDescriptor {
                features: adapter_features & needed_features,
                limits: wgpu::Limits::default(), 
                shader_validation: true,
             },
             trace_dir.ok().as_ref().map(std::path::Path::new),
        )
        .await
        .unwrap();

        (surface, device,queue,size)
}

fn create_swap_chain(size: winit::dpi::PhysicalSize<u32>, surface: &wgpu::Surface, device: &wgpu::Device) -> (wgpu::SwapChainDescriptor, wgpu::SwapChain) {
                                            
        let sc_desc = wgpu::SwapChainDescriptor {
            usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT,
            //format: wgpu::TextureFormat::Bgra8Unorm,
            format: if cfg!(target_arch = "wasm32") { wgpu::TextureFormat::Bgra8Unorm } else { wgpu::TextureFormat::Bgra8UnormSrgb },
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Mailbox,
        };

        let swap_chain = device.create_swap_chain(&surface, &sc_desc);
        (sc_desc, swap_chain)
}

fn run(window: Window, event_loop: EventLoop<()>, mut state: App) {

    #[cfg(all(not(target_arch = "wasm32"), feature = "subscriber"))]
    {
        let chrome_tracing_dir = std::env::var("WGPU_CHROME_TRACING");
        wgpu::util::initialize_default_subscriber(chrome_tracing_dir.as_ref().map(std::path::Path::new).ok());
    };

    #[cfg(not(target_arch = "wasm32"))]
    let (mut pool, _spawner) = {

        let local_pool = futures::executor::LocalPool::new();
        let spawner = local_pool.spawner();
        (local_pool, spawner)
    };

    #[cfg(target_arch = "wasm32")]
    let spawner = {
        use futures::{future::LocalFutureObj, task::SpawnError};
        use winit::platform::web::WindowExtWebSys;

        struct WebSpawner {}
        impl LocalSpawn for WebSpawner {
            fn spawn_local_obj(
                &self,
                future: LocalFutureObj<'static, ()>,
            ) -> Result<(), SpawnError> {
                Ok(wasm_bindgen_futures::spawn_local(future))
            }
        }

        //std::panic::set_hook(Box::new(console_error_panic_hook::hook));

        // On wasm, append the canvas to the document body
        web_sys::window()
            .and_then(|win| win.document())
            .and_then(|doc| doc.body())
            .and_then(|body| {
                body.append_child(&web_sys::Element::from(window.canvas()))
                    .ok()
            })
            .expect("couldn't append canvas to document body");

        WebSpawner {}
    };

    event_loop.run(move |event, _, control_flow| {
        let _ = (&state,&window);
        *control_flow = ControlFlow::Poll;
        pool.run_until_stalled();

        match event {
            Event::MainEventsCleared => {
                #[cfg(not(target_arch = "wasm32"))]
                {
                    window.request_redraw();
                    //pool.run_until_stalled();
                }

            }
            Event::WindowEvent { event: WindowEvent::Resized(size), .. } => {
                state.resize(size);
            }
            Event::WindowEvent {event, .. } => {
                if state.input(&event) { /* state.update() */ }
                match event { 
                    WindowEvent::CloseRequested => {
                        *control_flow = ControlFlow::Exit 
                    }
                    WindowEvent::KeyboardInput { input, ..  } => {
                        match input {
                            KeyboardInput {
                                state: ElementState::Pressed,
                                virtual_keycode: Some(VirtualKeyCode::Escape),
                                ..
                            } => *control_flow = ControlFlow::Exit,
                            KeyboardInput {
                                state: ElementState::Pressed,
                                virtual_keycode: Some(VirtualKeyCode::Key1),
                                ..
                            } => state.example = Example::TwoTriangles,
                            KeyboardInput {
                                state: ElementState::Pressed,
                                virtual_keycode: Some(VirtualKeyCode::Key2),
                                ..
                            } => state.example = Example::Cube,
                            KeyboardInput {
                                state: ElementState::Pressed,
                                virtual_keycode: Some(VirtualKeyCode::Key3),
                                ..
                            } => state.example = Example::Mc,
                            KeyboardInput {
                                state: ElementState::Pressed,
                                virtual_keycode: Some(VirtualKeyCode::Key4),
                                ..
                            } => state.example = Example::FmmGhostPoints,
                            KeyboardInput {
                                state: ElementState::Pressed,
                                virtual_keycode: Some(VirtualKeyCode::Key5),
                                ..
                            } => state.example = Example::CellPoints,
                            KeyboardInput {
                                state: ElementState::Pressed,
                                virtual_keycode: Some(VirtualKeyCode::Key6),
                                ..
                            } => state.example = Example::VolumetricNoise,
                            // KeyboardInput {
                            //     state: ElementState::Pressed,
                            //     virtual_keycode: Some(VirtualKeyCode::G),
                            //     ..
                            //} => state.fmm_debug_state.boundary_points = !state.fmm_debug_state.boundary_points,
                            KeyboardInput {
                                state: ElementState::Pressed,
                                virtual_keycode: Some(VirtualKeyCode::L),
                                ..
                            } => state.fmm_debug_state.grids = !state.fmm_debug_state.grids,
                            KeyboardInput {
                                state: ElementState::Pressed,
                                virtual_keycode: Some(VirtualKeyCode::N),
                                ..
                            } => state.fmm_debug_state.cells = !state.fmm_debug_state.cells,
                            KeyboardInput {
                                state: ElementState::Pressed,
                                virtual_keycode: Some(VirtualKeyCode::B),
                                ..
                            } => state.fmm_debug_state.triangles = !state.fmm_debug_state.triangles,
                            KeyboardInput {
                                state: ElementState::Pressed,
                                virtual_keycode: Some(VirtualKeyCode::T),
                                ..
                            } => state.fmm_debug_state.triangles = !state.fmm_debug_state.triangles,
                            _ => {}
                        } // match input 
                    } // KeyboardInput
                    _ => { /*state.update()*/ } // Other WindowEvents
                } // match event (WindowEvent)
            } // Event::WindowEvent
            Event::RedrawRequested(_) => {
                state.render(&window);
            }
            _ => { } // Any other events
        } // match event
    }); // run
}

fn main() {
      
    let event_loop = EventLoop::new();
    let mut builder = winit::window::WindowBuilder::new();
    builder = builder.with_title("Joo");
    let window = builder.build(&event_loop).unwrap();
    //let window = winit::window::Window::new(&event_loop).unwrap();
      
    #[cfg(not(target_arch = "wasm32"))]
    {
        let state = futures::executor::block_on(App::new(&window));
        run(window, event_loop, state);
    }

    #[cfg(target_arch = "wasm32")]
    {
        std::panic::set_hook(Box::new(console_error_panic_hook::hook));
        console_log::init().expect("could not initialize logger");
        use winit::platform::web::WindowExtWebSys;
        // On wasm, append the canvas to the document body
        web_sys::window()
            .and_then(|win| win.document())
            .and_then(|doc| doc.body())
            .and_then(|body| {
                body.append_child(&web_sys::Element::from(window.canvas()))
                    .ok()
            })
            .expect("couldn't append canvas to document body");
        wasm_bindgen_futures::spawn_local(async move {let mut state = App::new(&window).await; run(window, event_loop, state);});
    }
}
