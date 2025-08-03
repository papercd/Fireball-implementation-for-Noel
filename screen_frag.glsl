#version 330
out vec4 fragColor;
in vec2 vUV;
uniform sampler2D u_texture;

void main() {
    fragColor = texture(u_texture, vUV);
}
