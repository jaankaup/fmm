#version 450

layout(location = 0) in vec3 pos;
layout(location = 1) in uint color;
layout(location = 2) in vec4 normal;

layout(location = 0) out vec3 position_out;
layout(location = 1) out vec3 normal_out;
layout(location = 2) out vec3 color_out;

layout(set=0, binding=0) uniform camerauniform {
    mat4 u_view_proj;
    vec4 camera_pos;
};

// layout(set=0, binding=1) readonly buffer instanced_buffer {
//     vec4 cube_positions;
// };

vec4 decode_color(uint c) {
  float a = (c & 0xff) / 255.0;     
  float b = ((c & 0xff00) >> 8) / 255.0;
  float g = ((c & 0xff0000) >> 16) / 255.0;    
  float r = ((c & 0xff000000) >> 24) / 255.0;    
  return vec4(r,g,b,a);
}

void main() {
    gl_Position = u_view_proj * vec4(pos, 1.0); 
    position_out = pos;
    normal_out = normal.xyz;
    color_out = decode_color(color).xyz;
}
