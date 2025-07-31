#version 330

uniform vec2 iResolution;
uniform float iTime;

uniform sampler2D iChannel1; // bufferC (previous force)
uniform sampler2D iChannel2; // noise

out vec4 fragColor;
in vec2 fragCoord;

vec2 DecodeForce(vec2 force) {
    return force * 2.0 - 1.0;
}

vec2 EncodeForce(vec2 force) {
    return clamp(force, -1.0, 1.0) * 0.5 + 0.5;
}

mat2 rot(float a) {
    vec2 s = sin(vec2(a, a + 3.14159265359 / 2.0));
    return mat2(s.y, s.x, -s.x, s.y);
}

vec4 GetNoise(vec2 uv, float ratio)
{
    vec3 noiseCoord1 = vec3(uv.x * ratio, uv.y, 0.0) + iTime * vec3(0.0, 0.1, 0.2);
    noiseCoord1 *= 2.7;

    vec3 noiseCoord2 = vec3(uv.x * ratio, uv.y, 0.0) + iTime * vec3(0.0, -0.1, -0.2);
    noiseCoord2 *= 0.8;


    vec2 offset = vec2(iTime * 0.15);
    vec4 noise1 = texture(iChannel2, noiseCoord1.xy + offset);
    vec4 noise2 = texture(iChannel2, noiseCoord2.xy - offset);


    return (noise1 + noise2) / 2.0;
}

const int Xiterations = 2;
const int Yiterations = 2;

const float sampleDistance1 = 0.006;
const float sampleDistance2 = 0.0001;

const float forceDamping = 0.01;
const float turbulenceAmount = 2.0;

void main()
{
    vec2 uv = fragCoord.xy / iResolution;
    vec2 fakeForce = vec2(sin(uv.x * 40.0 + iTime), cos(uv.y * 40.0 + iTime));
    fragColor = vec4(EncodeForce(fakeForce), 0.0, 1.0);
}
