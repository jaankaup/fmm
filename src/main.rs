//use futures::task::LocalSpawn;
use std::borrow::Cow::Borrowed;
//use futures::executor::LocalPool;
//use futures::executor::LocalSpawner;
use std::collections::HashMap;
use rand::prelude::*;
use std::time::{SystemTime, UNIX_EPOCH};
use jaankaup_hilbert::hilbert::hilbert_index_reverse;
use cgmath::{prelude::*, Vector3};
use fmm::misc::*;
use fmm::buffer::*;
use fmm::texture::*;
use fmm::camera::*;
//use fmm::radix_sort::*;
use fmm::marching_cubes::*;
//use fmm::radix_sort::*;
use fmm::app_resources::*;
use fmm::fast_marching::*;

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
    Volumetric3dTexture,
    Hilbert2d,
    FmmGhostPoints,
}

struct Buffers {
    camera_uniform_buffer: BufferInfo,
    ray_camera_uniform_buffer: BufferInfo,
    mc_uniform_buffer: BufferInfo,
    mc_counter_buffer: BufferInfo,
    mc_output_buffer: BufferInfo,
    ray_march_output_buffer: BufferInfo,
    ray_debug_buffer: BufferInfo,
    sphere_tracer_output_buffer: BufferInfo,
    noise_3d_output_buffer: BufferInfo,
    hilbert_2d: BufferInfo,
    fmm_points: BufferInfo,
}

// Noise 3d resolution.
static N_3D_RES: (u32, u32, u32) = (128,128,128);

// TODO: is noise_3d_output_buffer necessery? Why not save noise directly to noise3d texture
// (storage texture). 

// TODO: create a simple way to disable unwanted textures and buffers. Maybe each resource could be
// released and recreated based on the program state. 

// Size in bytes.
static BUFFERS:  Buffers = Buffers {
    camera_uniform_buffer:       BufferInfo { name: "camera_uniform_buffer",     size: None,},
    ray_camera_uniform_buffer:   BufferInfo { name: "ray_camera_uniform_buffer", size: None,},
    mc_uniform_buffer:           BufferInfo { name: "mc_uniform_buffer",         size: None,},
    mc_counter_buffer:           BufferInfo { name: "mc_counter_buffer",         size: None,},
    mc_output_buffer:            BufferInfo { name: "mc_output_buffer",          size: Some(64*64*64*24), },
    ray_march_output_buffer:     BufferInfo { name: "ray_march_output",          size: Some(CAMERA_RESOLUTION.0 as u32 * CAMERA_RESOLUTION.1 as u32 * 4),},
    ray_debug_buffer:            BufferInfo { name: "ray_debug_buffer",          size: Some(CAMERA_RESOLUTION.0 as u32 * CAMERA_RESOLUTION.1 as u32 * 4 * 4),},
    sphere_tracer_output_buffer: BufferInfo { name: "sphere_tracer_output",      size: Some(CAMERA_RESOLUTION.0 as u32 * CAMERA_RESOLUTION.1 as u32 * 12 * 4),},
    noise_3d_output_buffer:      BufferInfo { name: "noise_3d_output_buffer",    size: Some(N_3D_RES.0 * N_3D_RES.1 * N_3D_RES.2 * 4),},
    hilbert_2d:                  BufferInfo { name: "hilbert_2d",                size: None,},
    fmm_points:                  BufferInfo { name: "fmm_points",                size: None,}, // TODO: to radix_sort.rs
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
               sample_count: u32) {

            let multi_sampled = multisampled(sample_count);

            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: Borrowed(&[
                    wgpu::RenderPassColorAttachmentDescriptor {
                            attachment: match multi_sampled { false => &frame.view, true => &multisampled_framebuffer, },
                            resolve_target: match multi_sampled { false => None, true => Some(&frame.view), },
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color { 
                                    r: 0.0,
                                    g: 0.0,
                                    b: 0.0,
                                    a: 1.0,
                                }),
                                store: true,
                            },
                    }
                ]),
                //depth_stencil_attachment: None,
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachmentDescriptor {
                    attachment: &textures.get(TEXTURES.depth.name).unwrap().view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0), 
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

// static TWO_TRIANGLES_SHADERS: [ShaderModuleInfo; 2]  = [
//     ShaderModuleInfo {name: "two_triangles_vert", source_file: "two_triangles_vert.spv", stage: "vertex"},
//     ShaderModuleInfo {name: "two_triangles_frag", source_file: "two_triangles_frag.spv", stage: "frag"},
// ];

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

fn create_two_triangles_info(sample_count: u32) -> RenderPipelineInfo { 
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
            (wgpu::VertexFormat::Float4, 4 * std::mem::size_of::<f32>() as u64),
            (wgpu::VertexFormat::Float4, 4 * std::mem::size_of::<f32>() as u64)
        ],
    };

    two_triangles_info
}

fn line_info(_sample_count: u32) -> RenderPipelineInfo { 
    let line_info: RenderPipelineInfo = RenderPipelineInfo {
        vertex_shader: ShaderModuleInfo {
            name: LINE_SHADERS[0].name,
            source_file: LINE_SHADERS[0].source_file,
            _stage: "vertex"
        }, 
        fragment_shader: Some(ShaderModuleInfo {
            name: LINE_SHADERS[1].name,
            source_file: LINE_SHADERS[1].source_file,
            _stage: "frag"
        }), 
        bind_groups: vec![
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
        ],
        input_formats: vec![
            (wgpu::VertexFormat::Float4, 4 * std::mem::size_of::<f32>() as u64),
        ],
    };

    line_info
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
                           //component_type: wgpu::TextureComponentType::Uint,
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

fn sphere_tracer_info(sample_count: u32) -> ComputePipelineInfo {
   let sphere_tracer_info: ComputePipelineInfo = ComputePipelineInfo {
       compute_shader: ShaderModuleInfo {
           name: SPHERE_TRACER_SHADER.name,
           source_file: SPHERE_TRACER_SHADER.source_file,
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
                           min_binding_size: None,
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
                       resource: Resource::TextureSampler(TEXTURES.grass.name),
                       binding_type: wgpu::BindingType::Sampler {
                          comparison: false,
                       },
                   },
                   BindGroupInfo {
                        binding: 2,
                        visibility: wgpu::ShaderStage::COMPUTE,
                        resource: Resource::TextureView(TEXTURES.rock.name),
                        binding_type: wgpu::BindingType::SampledTexture {
                           multisampled: multisampled(sample_count),
                           component_type: wgpu::TextureComponentType::Float,
                           dimension: wgpu::TextureViewDimension::D2,
                        },
                   },
                   BindGroupInfo {
                       binding: 3,
                       visibility: wgpu::ShaderStage::COMPUTE,
                       resource: Resource::TextureSampler(TEXTURES.rock.name),
                       binding_type: wgpu::BindingType::Sampler {
                          comparison: false,
                       },
                   },
                   BindGroupInfo {
                        binding: 4,
                        visibility: wgpu::ShaderStage::COMPUTE,
                        resource: Resource::TextureView(TEXTURES.noise3d.name), // TODO: create texture.
                        binding_type: wgpu::BindingType::SampledTexture {
                           multisampled: false,
                           component_type: wgpu::TextureComponentType::Float,
                           dimension: wgpu::TextureViewDimension::D3,
                        },
                   },
                   BindGroupInfo {
                       binding: 5,
                       visibility: wgpu::ShaderStage::COMPUTE,
                       resource: Resource::TextureSampler(TEXTURES.noise3d.name),
                       binding_type: wgpu::BindingType::Sampler {
                          comparison: false,
                       },
                   },
               ],
               vec![
                   BindGroupInfo {
                        binding: 0,
                        visibility: wgpu::ShaderStage::COMPUTE,
                        resource: Resource::Buffer(BUFFERS.sphere_tracer_output_buffer.name),
                        binding_type: wgpu::BindingType::StorageBuffer {
                           dynamic: false,
                           readonly: false,
                           min_binding_size: wgpu::BufferSize::new(BUFFERS.sphere_tracer_output_buffer.size.unwrap().into()),
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
                           min_binding_size: wgpu::BufferSize::new(BUFFERS.ray_march_output_buffer.size.unwrap() as u64 / 4),
                        },
                   },
               ],
           ],
    };

    sphere_tracer_info
}

fn generate_noise3d_info() -> ComputePipelineInfo {
   let generate_noise3d_info: ComputePipelineInfo = ComputePipelineInfo {
       compute_shader: ShaderModuleInfo {
           name: GENERATE_3D_SHADER.name,
           source_file: GENERATE_3D_SHADER.source_file,
           _stage: "compute"
       }, 
       bind_groups:
           vec![ 
               vec![
                   BindGroupInfo {
                        binding: 0,
                        visibility: wgpu::ShaderStage::COMPUTE,
                        resource: Resource::Buffer(BUFFERS.noise_3d_output_buffer.name),
                        binding_type: wgpu::BindingType::StorageBuffer {
                           dynamic: false,
                           readonly: false,
                           min_binding_size:
                                wgpu::BufferSize::new(
                                    (TEXTURES.noise3d.width.unwrap() * TEXTURES.noise3d.height.unwrap() * TEXTURES.noise3d.depth.unwrap() * 4).into()
                                ),
                        },
                   },
               ],
           ],
    };

    generate_noise3d_info
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
      for i in 0..rpi.input_formats.len() {
          vertex_attributes.push(
              wgpu::VertexAttributeDescriptor {
                  format: rpi.input_formats[0].0,
                  offset: stride,
                  shader_location: i as u32,
              }
          );
          stride = stride + rpi.input_formats[i].1;  
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
            znear: 0.1,
            zfar: 1000.0,
        };

        // The camera controller.
        let camera_controller = CameraController::new(0.1,0.1);

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

        /* TWO TRIANGLES */

        println!("Creating two_triangles pipeline and bind groups.\n");
        let two_triangles_info = create_two_triangles_info(sample_count); 
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

        /* POINT () */

        println!("\nCreating ghostpoint pipeline and bind groups.\n");
        let point_info = line_info(sample_count);
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

        let mut domain = DomainE::new(20,20,20);
        //domain.initialize_boundary(|x, y, z| x.powf(0.3) + y.powf(0.3) + z.powf(0.3) - 5.0); 
        domain.initialize_boundary(|x, y, z| y - 3.5); 
        let fmm_ghost_data = domain.ghost_points_to_vec();

//        let initial_values: Vec<cgmath::Vector3<f32>> = Vec::new();
//        fmm::initialize_interface(&mut domain, &initial_values);

        let fmm_ghost_points = Buffer::create_buffer_from_data::<f32>(
            &device,
            &fmm_ghost_data,
            wgpu::BufferUsage::VERTEX,
            None
        );
        buffers.insert(BUFFERS.fmm_points.name.to_string(), fmm_ghost_points);

        let ghost_point_size = (domain.dim_x + 1) * (domain.dim_x + 1) * (domain.dim_x + 1);

        let fmm_ghost_vb_info = VertexBufferInfo {
            vertex_buffer_name: BUFFERS.fmm_points.name.to_string(),
            _index_buffer: None,
            start_index: 0,
            end_index: ghost_point_size,
            instances: 1,
        };

        vertex_buffer_infos.insert("fmm_ghost_vb_info".to_string(), fmm_ghost_vb_info);

        println!("");

        /* LINE (hilbert2d) */

        println!("\nCreating line pipeline and bind groups.\n");
        let line_info = line_info(sample_count);
        let (line_groups, line_pipeline) = create_render_pipeline_and_bind_groups(
                        &device,
                        &sc_desc,
                        &shaders,
                        &textures,
                        &buffers,
                        &line_info,
                        &wgpu::PrimitiveTopology::LineStrip,
                        sample_count);

        let line_vb_info = VertexBufferInfo {
            vertex_buffer_name: BUFFERS.hilbert_2d.name.to_string(),
            _index_buffer: None,
            start_index: 0,
            end_index: 8*8*8,
            instances: 1,
        };

        vertex_buffer_infos.insert("line_vb_info".to_string(), line_vb_info);

        let line_render_pass = RenderPass {
            pipeline: line_pipeline,
            bind_groups: line_groups,
        };

        render_passes.insert("line_render_pass".to_string(), line_render_pass);

        println!("");

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
            dispatch_x: 18,
            dispatch_y: 18,
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
        println!("");

        println!("\nCreating volumetric ray caster (3d texture) pipeline and bind groups.\n");
        let volume_3d_info = sphere_tracer_info(sample_count); // TODO: rename info function.
        let (volume_3d_bind_groups, volume_3d_compute_pipeline) = create_compute_pipeline_and_bind_groups(
                        &device,
                        &shaders,
                        &textures,
                        &buffers,
                        &volume_3d_info);

        let volume_3d_pass = ComputePass {
            pipeline: volume_3d_compute_pipeline,
            bind_groups: volume_3d_bind_groups,
            dispatch_x: CAMERA_RESOLUTION.0 / 8,
            dispatch_y: CAMERA_RESOLUTION.1 / 8,
            dispatch_z: 1,
        };

        compute_passes.insert("volume_3d_pass".to_string(), volume_3d_pass);
        println!("");

        println!("\nCreating generate 3d noise pipeline and bind groups.\n");
        let noise3d_info = generate_noise3d_info();
        let (noise3d_bind_groups, noise3d_compute_pipeline) = create_compute_pipeline_and_bind_groups(
                        &device,
                        &shaders,
                        &textures,
                        &buffers,
                        &noise3d_info);

        let noise_3d_pass = ComputePass {
            pipeline: noise3d_compute_pipeline,
            bind_groups: noise3d_bind_groups,
            dispatch_x: TEXTURES.noise3d.width.unwrap() as u32 / 4,
            dispatch_y: TEXTURES.noise3d.height.unwrap() as u32 / 4,
            dispatch_z: TEXTURES.noise3d.depth.unwrap() as u32 / 4,
        };

        compute_passes.insert("noise_3d_pass".to_string(), noise_3d_pass);

        println!("");

        println!("\nLaunching generate 3d noise.\n");

        let mut noise_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        compute_passes.get("noise_3d_pass")
                      .unwrap()
                      .execute(&mut noise_encoder);

        let noise_texture_dimension_x = TEXTURES.noise3d.width.expect("Consider giving TEXTURES.noise3d a width.") as u32;
        let noise_texture_dimension_y = TEXTURES.noise3d.height.expect("Consider giving TEXTURES.noise3d a height.") as u32;
        let noise_texture_dimension_z = TEXTURES.noise3d.depth.expect("Consider giving TEXTURES.noise3d a depth.") as u32;

        noise_encoder.copy_buffer_to_texture(
            wgpu::BufferCopyView {
                buffer: &buffers.get(BUFFERS.noise_3d_output_buffer.name).unwrap().buffer,
                layout: wgpu::TextureDataLayout {
                    offset: 0,
                    bytes_per_row: noise_texture_dimension_x * 4, // 16 
                    rows_per_image: noise_texture_dimension_z,
                },
            },
            wgpu::TextureCopyView{
                texture: &textures.get(TEXTURES.noise3d.name).unwrap().texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
            },
            wgpu::Extent3d {
                width: noise_texture_dimension_x,
                height: noise_texture_dimension_y,
                depth: noise_texture_dimension_z,
            });

        queue.submit(Some(noise_encoder.finish()));

//        let noise_output = &textures.get(TEXTURES.noise3d.name).unwrap().to_vec::<u8>(&device, &queue).await;
//        for i in 0..1024 {
//            println!("{}", noise_output[i]);
//        }
//        let mut noise_counter = 0;
//        for i in 0..131072*4 {
//            if noise_counter == 0 {
//                print!("{} :: (", i/4);
//            }
//            print!(" {} ",noise_output[i]);
//            if noise_counter == 3 {
//                println!(")");
//                noise_counter = 0;
//                continue;
//            }
//            noise_counter = noise_counter + 1;
            //println!("{} :: origin: ({}, {}, {}, {})", i, sphere_output[offset], sphere_output[offset+1], sphere_output[offset+2], sphere_output[offset+3]);
            //println!("      intersection_point: ({}, {}, {}, {})", sphere_output[offset+4], sphere_output[offset+5], sphere_output[offset+6], sphere_output[offset+7]);
            //println!("      normal: ({}, {}, {}, {})", sphere_output[offset+8], sphere_output[offset+9], sphere_output[offset+10], sphere_output[offset+11]);
//        }

        println!("");

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

                // Launch sprhere tracer.
                Example::Volumetric3dTexture => {

                    self.compute_passes.get("volume_3d_pass")
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
                    .execute(&mut encoder, &frame, &self.multisampled_framebuffer, &self.textures, &self.buffers, &vb_info, self.sample_count);
                },
                Example::Cube => {
                    let vb_info = self.vertex_buffer_infos.get("vtn_vb_info").expect("Could not find vertex buffer info");
                    self.render_passes.get("vtn_render_pass")
                    .unwrap()
                    .execute(&mut encoder, &frame, &self.multisampled_framebuffer, &self.textures, &self.buffers, &vb_info, self.sample_count);
                },
                Example::Mc => {
                    let rp = self.vertex_buffer_infos.get("mc_renderer_vb_info") .unwrap();
                    self.render_passes.get("mc_renderer_pass")
                    .unwrap()
                    .execute(&mut encoder, &frame, &self.multisampled_framebuffer, &self.textures, &self.buffers, &rp, self.sample_count);
                }
                Example::FmmGhostPoints => {
                    let rp = self.vertex_buffer_infos.get("fmm_ghost_vb_info") .unwrap();
                    self.render_passes.get("point_render_pass")
                    .unwrap()
                    .execute(&mut encoder, &frame, &self.multisampled_framebuffer, &self.textures, &self.buffers, &rp, self.sample_count);
                },
                Example::VolumetricNoise => {
                    let rp = self.vertex_buffer_infos.get("two_triangles_vb_info") .unwrap();
                    self.render_passes.get("ray_renderer_pass")
                    .unwrap()
                    .execute(&mut encoder, &frame, &self.multisampled_framebuffer, &self.textures, &self.buffers, &rp, self.sample_count);

                },
                Example::Volumetric3dTexture => {
                    let rp = self.vertex_buffer_infos.get("two_triangles_vb_info") .unwrap();
                    self.render_passes.get("ray_renderer_pass")
                    .unwrap()
                    .execute(&mut encoder, &frame, &self.multisampled_framebuffer, &self.textures, &self.buffers, &rp, self.sample_count);
                },
                Example::Hilbert2d => {
                    let rp = self.vertex_buffer_infos.get("line_vb_info") .unwrap();
                    self.render_passes.get("line_render_pass")
                    .unwrap()
                    .execute(&mut encoder, &frame, &self.multisampled_framebuffer, &self.textures, &self.buffers, &rp, self.sample_count);
                },

                //_ => {},
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

    print!("    * Creating '{}' shader module from file '{}'",RADIX_SHADER.name, RADIX_SHADER.source_file);
    shaders.insert(RADIX_SHADER.name.to_string(), device.create_shader_module(wgpu::include_spirv!("radix.comp.spv")));
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
        base_position: cgmath::Vector4::new(0.0, 0.0, 0.0, 1.0),
        isovalue: 0.0,
        cube_length: 0.15,
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

    print!("    * Creating sphere_output_buffer buffer as '{}'", BUFFERS.sphere_tracer_output_buffer.name);

    let sphere_output_buffer = Buffer::create_buffer_from_data::<f32>(
        device,
        &vec![0 as f32 ; (BUFFERS.sphere_tracer_output_buffer.size.unwrap() / 4) as usize],
        wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_DST | wgpu::BufferUsage::COPY_SRC,
        None
    );

    buffers.insert(BUFFERS.sphere_tracer_output_buffer.name.to_string(), sphere_output_buffer);
    println!(" ... OK'");

    print!("    * Creating noise output buffer as '{}'", BUFFERS.noise_3d_output_buffer.name);
    let noise_output_buffer = Buffer::create_buffer_from_data::<f32>(
        device,
        &vec![0 as f32 ; BUFFERS.noise_3d_output_buffer.size.unwrap() as usize / 4],
        wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_DST | wgpu::BufferUsage::COPY_SRC,
        None
    );
    buffers.insert(BUFFERS.noise_3d_output_buffer.name.to_string(), noise_output_buffer);
    println!(" ... OK'");

    let mut hilbert: Vec<f32> = Vec::new(); //vec![0 as u32 ; 64*2];
    //let mut previous: [u32 ; 2] = [0,0];
    for i in 0..(8*8*8) {
        //println!("i == {}", i);
        let inverse = hilbert_index_reverse(3, 3, i);
        hilbert.push(inverse[0] as f32 * 0.5);
        hilbert.push(inverse[1] as f32 * 0.5);
        hilbert.push(inverse[2] as f32 * 0.5);
        hilbert.push(((8.0*8.0*8.0) - i as f32) / (8.0*8.0*8.0));
    }

    let hilbert_buffer = Buffer::create_buffer_from_data::<f32>(
        device,
        &hilbert,
        wgpu::BufferUsage::VERTEX,
        None
    );

    buffers.insert(BUFFERS.hilbert_2d.name.to_string(), hilbert_buffer);

    println!(" ... OK'");

    //let radix_data_size = 18000;
    //let mut radix_example_data = vec![0 as u32 ; radix_data_size];
    //for i in 0..radix_data_size {
    //    let random_number: u8 = rng.gen(); 
    //    radix_example_data[i] = (random_number as u32) << 24;
    //}

    //print!("    * Creating radix_input buffer as '{}'", BUFFERS.radix_input.name);
    //let radix_initial_data = Buffer::create_buffer_from_data::<u32>(
    //    device,
    //    &radix_example_data,
    //    wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_DST | wgpu::BufferUsage::COPY_SRC,
    //    None
    //);
    //buffers.insert(BUFFERS.radix_input.name.to_string(), radix_initial_data);
    //println!(" ... OK'");

    //// TODO: create not from data
    //print!("    * Creating radix_auxiliary buffer as '{}'", BUFFERS.radix_auxiliary.name);
    //let radix_auxiliary = Buffer::create_buffer_from_data::<u32>(
    //    device,
    //    &radix_example_data,
    //    wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_DST | wgpu::BufferUsage::COPY_SRC,
    //    None
    //);
    //buffers.insert(BUFFERS.radix_auxiliary.name.to_string(), radix_auxiliary);
    //println!(" ... OK'");

    //// TODO: create not from data
    //print!("    * Creating radix_histogram buffer as '{}'", BUFFERS.radix_histogram.name);
    //let radix_histogram = Buffer::create_buffer_from_data::<u32>(
    //    device,
    //    &radix_example_data,
    //    wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_DST | wgpu::BufferUsage::COPY_SRC,
    //    None
    //);
    //buffers.insert(BUFFERS.radix_histogram.name.to_string(), radix_histogram);
    //println!(" ... OK'");

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
                            } => state.example = Example::VolumetricNoise,
                            KeyboardInput {
                                state: ElementState::Pressed,
                                virtual_keycode: Some(VirtualKeyCode::Key6),
                                ..
                            } => state.example = Example::Volumetric3dTexture,
                            KeyboardInput {
                                state: ElementState::Pressed,
                                virtual_keycode: Some(VirtualKeyCode::Key7),
                                ..
                            } => state.example = Example::Hilbert2d,
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
