#version 450

layout(location = 0) in vec4 nor_f;

layout(location = 0) out vec4 final_color;

void main() {
    final_color = nor_f;
}
