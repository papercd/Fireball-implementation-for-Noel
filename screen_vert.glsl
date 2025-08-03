#version 330
in vec2 in_position;
out vec2 vUV;

void main() {
    gl_Position = vec4(in_position, 0.0, 1.0);
    vUV = (in_position + 1.0) * 0.5;  // now in [0, 1] range
}
