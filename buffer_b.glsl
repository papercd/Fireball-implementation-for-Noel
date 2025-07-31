#version 330

uniform vec2 iResolution;
uniform sampler2D iChannel0; // bufferA output
uniform sampler2D iChannel1; // bufferC output

out vec4 fragColor;
in vec2 fragCoord;

vec2 DecodeForce(vec2 force)
{
    force = force * 2.0 - 1.0;
    return force;
}

const float flow1 = 0.5;
const float flow2 = 0.75;
const float speed = 0.02;
const float gravity = -0.15;

void main()
{
    float ratio = iResolution.x / iResolution.y;
    vec2 uv = fragCoord.xy / iResolution.xy;

    vec4 source = texture(iChannel0, uv);
    vec2 force = texture(iChannel1, uv).xy;
    force = DecodeForce(force);
    force.y -= gravity;

    vec2 s = vec2(speed);
    s.x /= ratio;
    force *= s;

    source.z = smoothstep(flow1, flow2, source.z);  // flow alpha blend

    vec2 movedForce = texture(iChannel1, uv - force).xy;
    movedForce = mix(movedForce, source.xy, source.z);

    fragColor = vec4(movedForce.x, movedForce.y, 0.0, 1.0);
}
