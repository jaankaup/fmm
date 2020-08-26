#version 450

// layout(location = 0) in vec4 pos;
layout(location = 0) flat in uint color_out;

layout(location = 0) out vec4 final_color;

//pub fn encode_vector_f32(r: u8, g: u8, b: u8, a: u8) -> f32 {
//
//
//  let mut result: f32 = 0.0;
//  let mut color: u32 = 0;
//  color = ((r as u32) << 24) | ((g as u32) << 16) | ((b as u32) << 8) | a as u32; 
//
//  // Copy color bits to f32. This value is then decoded in shader.
//  unsafe { result = std::mem::transmute::<u32, f32>(color); }
//
//  result

vec4 decode_color(uint c) {
  float a = (c & 0xff) / 255.0;     
  float b = ((c & 0xff00) >> 8) / 255.0;
  float g = ((c & 0xff0000) >> 16) / 255.0;    
  float r = ((c & 0xff000000) >> 24) / 255.0;    
  return vec4(r,g,b,a);
}

void main() {
    //float color_a = pos.w;
    final_color = decode_color(color_out);
    //final_color = vec4(color_a, color_a, color_a, 1.0);
    //final_color = vec4(color_r / 255.0, color_g / 255.0, color_b / 255.0, color_a / 255.0);
}
