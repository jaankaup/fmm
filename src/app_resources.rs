use crate::texture::Texture;
use std::borrow::Cow::Borrowed;
use std::collections::HashMap;

pub struct Textures {
    pub grass: TextureInfo,
    pub rock: TextureInfo,
    pub noise3d: TextureInfo,
    pub depth: TextureInfo,
    pub ray_texture: TextureInfo,
}

pub struct TextureInfo {
    pub name: &'static str,
    pub source: Option<&'static str>,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub depth: Option<u32>,
}

// Ray camera resolution.
pub static CAMERA_RESOLUTION: (u32, u32) = (512,512);

// Noise 3d resolution.
static N_3D_RES: (u32, u32, u32) = (128,128,128);

pub static TEXTURES: Textures = Textures {
    grass:       TextureInfo { name: "GrassTexture",     source: Some("grass2.png"), width: None,                      height: None,                      depth: None, },
    rock:        TextureInfo { name: "rock_texture",     source: Some("rock.png"),   width: None,                      height: None,                      depth: None, },
    noise3d:     TextureInfo { name: "noise_3d_texture", source: None,               width: Some(N_3D_RES.0),          height: Some(N_3D_RES.1),          depth: Some(N_3D_RES.2), },
    depth:       TextureInfo { name: "depth_texture",    source: None,               width: None,                      height: None,                      depth: None, },
    ray_texture: TextureInfo { name: "ray_texture",      source: None,               width: Some(CAMERA_RESOLUTION.0), height: Some(CAMERA_RESOLUTION.1), depth: Some(1), },
};

pub fn create_textures(device: &wgpu::Device, queue: &wgpu::Queue, sc_desc: &wgpu::SwapChainDescriptor, textures: &mut HashMap<String, Texture>, sample_count: u32) {

    println!("\nCreating textures.\n");

    print!("    * Creating texture from {}.", TEXTURES.grass.source.expect("Missing texture source."));
    let grass_texture = Texture::create_from_bytes(&queue, &device, &sc_desc, sample_count, &include_bytes!("grass2.png")[..], None);
    textures.insert(TEXTURES.grass.name.to_string(), grass_texture);
    println!(" ... OK'");

    print!("    * Creating texture from '{}'", TEXTURES.rock.source.expect("Missing texture source."));
    let rock_texture = Texture::create_from_bytes(&queue, &device, &sc_desc, sample_count, &include_bytes!("rock.png")[..], None);
    textures.insert(TEXTURES.rock.name.to_string(), rock_texture);
    println!(" ... OK'");

    print!("    * Creating depth texture.");
    let depth_texture = Texture::create_depth_texture(&device, &sc_desc, Some(Borrowed("depth-texture")));
    textures.insert(TEXTURES.depth.name.to_string(), depth_texture);
    println!(" ... OK'");
      
    print!("    * Creating ray texture.");
    let ray_texture = Texture::create_texture2d(&device, &sc_desc, sample_count, CAMERA_RESOLUTION.0, CAMERA_RESOLUTION.1);
    textures.insert(TEXTURES.ray_texture.name.to_string(), ray_texture);
    println!(" ... OK'");

    print!("    * Creating {} texture.", TEXTURES.noise3d.name);
    let noise3dtexture = Texture::create_texture3d(
        &device,
        &sc_desc.format,
        TEXTURES.noise3d.width.unwrap() as u32,
        TEXTURES.noise3d.height.unwrap() as u32,
        TEXTURES.noise3d.depth.unwrap() as u32,
    );
    textures.insert(TEXTURES.noise3d.name.to_string(), noise3dtexture);
    println!(" ... OK'");
}
