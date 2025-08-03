#version 330
uniform sampler3D tex3D;
uniform float sliceIndex; // [0.0, 1.0]

in vec2 uv;
out vec4 fragColor;

void main() {
    vec3 texCoord = vec3(uv, sliceIndex);
    fragColor = texture(tex3D, texCoord);
}
