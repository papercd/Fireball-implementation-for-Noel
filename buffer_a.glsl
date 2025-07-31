#version 330

uniform vec2 iResolution;
uniform float iTime;
uniform vec4 iMouse;
uniform sampler2D iChannel2; // Noise texture

out vec4 fragColor;
in vec2 fragCoord;

// === Common functions ===
vec2 EncodeForce(vec2 force)
{
    force = clamp(force, -1.0, 1.0);
    return force * 0.5 + 0.5;
}

vec2 DecodeForce(vec2 force)
{
    force = force * 2.0 - 1.0;
    return force;
}

const float pi = 3.14159265359;
const float tau = 6.28318530718;

mat2 rot(float a)
{
    vec2 s = sin(vec2(a, a + pi / 2.0));
    return mat2(s.y, s.x, -s.x, s.y);
}

float linearStep(float a, float b, float x)
{
    return clamp((x - a) / (b - a), 0.0, 1.0);
}

// === Buffer A shader ===
vec4 GetNoise(vec2 uv, float ratio)
{
    vec3 noiseCoord1;
    noiseCoord1.xy = uv;
    noiseCoord1.x *= ratio;
    noiseCoord1 += iTime * vec3(-0.05, 0.0, 0.2);
    noiseCoord1 *= 3.3;

    vec3 noiseCoord2;
    noiseCoord2.xy = uv;
    noiseCoord2.x *= ratio;
    noiseCoord2 += iTime * vec3(0.05, 0.0, -0.2);
    noiseCoord2 *= 0.8;

    vec2 offset = vec2(iTime * 0.15);
    vec4 noise1 = texture(iChannel2, noiseCoord1.xy + offset);
    vec4 noise2 = texture(iChannel2, noiseCoord2.xy - offset);

    return (noise1 + noise2) / 2.0;
}

void main()
{
    float ratio = iResolution.x / iResolution.y;
    vec2 uv = fragCoord.xy / iResolution.xy;

    vec2 circleCoord = uv;
    vec2 mousePos = vec2(0.0);
    vec2 circleVelocity = vec2(0.0);

    if (iMouse.z > 0.5) {
        circleCoord -= iMouse.xy / iResolution.xy;
    } else {
        circleCoord -= 0.5;
        circleCoord.xy += sin(iTime * vec2(1.0, 2.0)) * vec2(0.35, 0.25);
    }

    circleCoord.x *= ratio;

    float circle = length(circleCoord);
    float bottom = uv.y;

    vec4 masksIN = vec4(0.08, 0.35, 0.05, 0.2);
    vec4 masksOUT = vec4(0.06, 0.0, 0.0, 0.0);
    vec4 masksValue = vec4(circle, circle, bottom, bottom);
    vec4 masks = smoothstep(masksIN, masksOUT, masksValue);

    vec2 mask = masks.xy + masks.zw;

    vec4 noise = GetNoise(uv, ratio);

    vec2 force = circleCoord * noise.xy * 45.0 * masks.x;
    force += (noise.xy - 0.5) * (masks.x * 0.5 + masks.z * 0.75);
    force.y += (0.25 + 0.75 * noise.z) * (masks.x * 0.0 + masks.z * 0.8);
    force = EncodeForce(force);

    fragColor = vec4(force.x, force.y, mask.x, mask.y);
}
