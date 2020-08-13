#version 450

layout(location = 0) in vec4 pos;

layout(location = 0) out vec4 final_color;

void main() {
    //float color_a = pos.w;
    //float color_b = 1.0 - pos.w;
    //final_color = vec4(color_a, 0.0, color_b, 1.0);
    float color_a = pos.w;
    //if (pos.w < 0.45) color_a = 0.0;
    //if (pos.w > 0.55) color_a = 0.0;
    //float color_b = 1.0 - pos.w;
    final_color = vec4(color_a, color_a, color_a, 1.0);
    //final_color = vec4(1.0, 0.0, 0.0, 1.0);
}
