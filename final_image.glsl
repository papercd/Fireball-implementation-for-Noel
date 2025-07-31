#version 330
out vec4 fragColor;
in vec2 fragCoord;

uniform sampler2D iChannel0; // bufferB (fluid)
uniform sampler2D iChannel1; // bufferC (force)
uniform vec2 iResolution;

vec2 DecodeForce(vec2 force) {
    return force * 2.0 - 1.0;
}

vec3 gradient(float value) {
    vec3 color1 = vec3(0.0, 0.05, 0.2);
    vec3 color2 = vec3(0.1, 0.0, 0.1);
    vec3 color3 = vec3(0.5, 0.15, 0.25);
    vec3 color4 = vec3(2.0, 1.25, 0.7);
    vec3 color5 = vec3(2.0, 2.0, 2.0);

    float a = 0.125, b = 0.35, c = 0.5;
    vec4 start = vec4(0.0, a, b, c);
    vec4 end = vec4(a, b, c, 1.0);
    vec4 mixValue = smoothstep(start, end, vec4(value));

    vec3 color = mix(color1, color2, mixValue.x);
    color = mix(color, color3, mixValue.y);
    color = mix(color, color4, mixValue.z);
    color = mix(color, color5, mixValue.w);

    return color;
}

void main() {
    
    vec2 uv = fragCoord.xy / iResolution;
    vec4 forceColor = texture(iChannel1, uv);  // bufferC output
    fragColor = forceColor;

}
