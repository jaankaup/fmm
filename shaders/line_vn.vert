#version 450

layout(location = 0) in vec4 pos;
layout(location = 1) in vec4 nor;

layout(location = 0) out vec4 nor_out;

layout(set=0, binding=0) uniform camerauniform {
    mat4 u_view_proj;
    vec4 camera_pos;
};

void main() {
    gl_Position = u_view_proj * vec4(pos.xyz, 1.0); 
    gl_PointSize = 2.0;
    nor_out = nor;
}
