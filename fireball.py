import moderngl
import numpy as np 
import pygame 
import time 
from scipy.ndimage import gaussian_filter
from double_buffer import DoubleBuffer

pygame.init()
size = (960,540)
screen = pygame.display.set_mode(size,pygame.OPENGL | pygame.DOUBLEBUF)
ctx = moderngl.create_context()

start_time  =time.time()
mouse_pos = (0.0,0.0)
mouse_pressed = False

# setup shaders 
common = """
vec2 EncodeForce(vec2 force) {
    force = clamp(force, -1.0, 1.0);
    return force * 0.5 + 0.5;
}

vec2 DecodeForce(vec2 force) {
    force = force * 2.0 - 1.0;
    return force;
}

const float pi = 3.14159265359;
const float tau = 6.28318530718;

mat2 rot(float a) {
    vec2 s = sin(vec2(a, a + pi/2.0));
    return mat2(s.y, s.x, -s.x, s.y);
}

float linearStep(float a, float b, float x) {
    return clamp((x - a)/(b - a), 0.0, 1.0);
}
"""

vertex_shader = """
#version 330 
in vec2 in_position;
out vec2 fragCoord; 

void main(){
    gl_Position = vec4(in_position,0.0,1.0);
    fragCoord = (in_position + 1.0) * 0.5 * vec2(800.0, 600.0);
}
"""
buffer_a_frag = f"""
#version 330
    {common}
    uniform float iTime;
    uniform vec3 iResolution;
    uniform vec4 iMouse;
    uniform sampler3D iChannel2;
    in vec2 fragCoord;
    out vec4 fragColor;
    
    const vec3 noiseSpeed1 = vec3(-0.05, 0.0, 0.2);
    const float noiseSize1 = 3.3;
    const vec3 noiseSpeed2 = vec3(0.05, 0.0, -0.2);
    const float noiseSize2 = 0.8;
    const float circleForceAmount = 15.0;
    const vec2 randomForceAmount = vec2(0.5, 0.75);
    const vec2 upForce = vec2(0.0, 0.8);
    const vec2 moveSpeed = vec2(1.0, 2.0);
    
    vec4 GetNoise(vec2 uv, float ratio) {{
        vec3 noiseCoord1;
        noiseCoord1.xy = uv;
        noiseCoord1.x *= ratio;
        noiseCoord1 += iTime * noiseSpeed1;
        noiseCoord1 *= noiseSize1;
        
        vec3 noiseCoord2;
        noiseCoord2.xy = uv;
        noiseCoord2.x *= ratio;
        noiseCoord2 += iTime * noiseSpeed2;
        noiseCoord2 *= noiseSize2;
        
        vec4 noise1 = texture(iChannel2, noiseCoord1);
        vec4 noise2 = texture(iChannel2, noiseCoord2);
        
        vec4 noise = (noise1 + noise2) / 2.0;
        
        return noise;
    }}
    
    void main() {{
        float ratio = iResolution.x / iResolution.y;
        vec2 uv = fragCoord / iResolution.xy;
        
        vec2 circleCoord = uv;   
        vec2 mousePos = vec2(0.0);
        vec2 circleVelocity = vec2(0.0);
        
        if(iMouse.z > 0.5) {{     
            circleCoord -= iMouse.xy/iResolution.xy;
        }} else {{
            circleCoord -= 0.5;
            circleCoord.xy += sin(iTime * moveSpeed) * vec2(0.35, 0.25);
        }}
        
        circleCoord.x *= ratio;
        
        float circle = length(circleCoord);
        float bottom = uv.y;
        
        vec4 masksIN = vec4(0.08, 0.35, 0.05, 0.2);
        vec4 masksOUT = vec4(0.06, 0.0, 0.0, 0.0);
        vec4 masksValue = vec4(circle, circle, bottom, bottom);
        vec4 masks = smoothstep(masksIN, masksOUT, masksValue);
        vec2 mask = masks.xy + masks.zw;
        
        vec4 noise = GetNoise(uv, ratio);
            
        vec2 force = circleCoord * noise.xy * circleForceAmount * masks.x;
        force += (noise.xy - 0.5) * (masks.x * randomForceAmount.x + masks.z * randomForceAmount.y);
        force.y += (0.25 + 0.75 * noise.z) * (masks.x * upForce.x + masks.z * upForce.y);
        force = EncodeForce(force);
        
        fragColor = vec4(force.x, force.y, mask.x, mask.y);
    }}
"""
buffer_b_frag = f"""
        #version 330
        {common}
        uniform float iTime;
        uniform vec3 iResolution;
        uniform sampler2D iChannel0;
        uniform sampler2D iChannel1;
        in vec2 fragCoord;
        out vec4 fragColor;
        
        const float flow1 = 0.5;
        const float flow2 = 0.75;
        const float speed = 0.02;
        const float gravity = -0.15;
        
        void main() {{
            float ratio = iResolution.x / iResolution.y;
            vec2 uv = fragCoord / iResolution.xy;
            vec4 source = texture(iChannel0, uv);
         
            vec2 force = texture(iChannel1, uv).xy;
            force = DecodeForce(force);
            force.y -= gravity;
            
            vec2 s = vec2(speed);
            s.x /= ratio;
            force *= s;
            
            source.z = smoothstep(flow1, flow2, source.z);
            
            vec2 movedForce = texture(iChannel1, uv - force).xy;
            movedForce = mix(movedForce, source.xy, source.z);
            
            fragColor = vec4(movedForce.x, movedForce.y, 0.0, 1.0);
        }}
        """
buffer_c_frag = f"""
        #version 330
        {common}
        uniform float iTime;
        uniform vec3 iResolution;
        uniform sampler2D iChannel1;
        uniform sampler3D iChannel2;
        in vec2 fragCoord;
        out vec4 fragColor;
        
        const int Xiterations = 2;
        const int Yiterations = 2;
        const float sampleDistance1 = 0.006;
        const float sampleDistance2 = 0.0001;
        const float forceDamping = 0.01;
        const vec3 noiseSpeed1 = vec3(0.0, 0.1, 0.2);
        const float noiseSize1 = 2.7;
        const vec3 noiseSpeed2 = vec3(0.0, -0.1, -0.2);
        const float noiseSize2 = 0.8;
        const float turbulenceAmount = 2.0;
        
        vec4 GetNoise(vec2 uv, float ratio) {{
            vec3 noiseCoord1;
            noiseCoord1.xy = uv;
            noiseCoord1.x *= ratio;
            noiseCoord1 += iTime * noiseSpeed1;
            noiseCoord1 *= noiseSize1;
            
            vec3 noiseCoord2;
            noiseCoord2.xy = uv;
            noiseCoord2.x *= ratio;
            noiseCoord2 += iTime * noiseSpeed2;
            noiseCoord2 *= noiseSize2;
            
            vec4 noise1 = texture(iChannel2, noiseCoord1);
            vec4 noise2 = texture(iChannel2, noiseCoord2);
            
            vec4 noise = (noise1 + noise2) / 2.0;
            
            return noise;
        }}
        
        void main() {{
            float ratio = iResolution.x / iResolution.y;
            vec2 uv = fragCoord / iResolution.xy;
            
            vec2 currentForce = DecodeForce(texture(iChannel1, uv).xy);
            float currentForceMagnitude = length(currentForce);
               
            vec3 sampleDistance; 
            sampleDistance.xy = vec2(mix(sampleDistance1, sampleDistance2, smoothstep(-0.25, 0.65, currentForceMagnitude)));
            sampleDistance.z = 0.0;
            
            vec2 totalForce = vec2(0.0);
            float iterations = 0.0;
            
            for(int x = -Xiterations; x <= Xiterations; x++) {{
                for(int y = -Yiterations; y <= Yiterations; y++) {{
                    vec3 dir = vec3(float(x), float(y), 0.0);
                    vec4 sampledValue = texture(iChannel1, uv + dir.xy * sampleDistance.xy);
                    
                    vec2 force = DecodeForce(sampledValue.xy); 
                    float forceValue = length(force);
                    totalForce += force * forceValue;
                    iterations += forceValue;
                }}
            }}
            
            totalForce /= iterations;  
            totalForce -= totalForce * forceDamping;
            
            float turbulence = GetNoise(uv, ratio).z - 0.5;
            turbulence *= mix(0.0, turbulenceAmount, smoothstep(0.0, 1.0, currentForceMagnitude));
            
            totalForce *= rot(turbulence);
            totalForce = EncodeForce(totalForce);
            
            fragColor = vec4(totalForce.x, totalForce.y, 0.0, 1.0);
        }}
        """

image_frag = f"""
        #version 330
        {common}
        uniform float iTime;
        uniform vec3 iResolution;
        uniform sampler2D iChannel0;
        uniform sampler2D iChannel1;
        in vec2 fragCoord;
        out vec4 fragColor;
        
        const vec3 color1 = vec3(0.0, 0.05, 0.2);
        const vec3 color2 = vec3(0.1, 0.0, 0.1);
        const vec3 color3 = vec3(0.5, 0.15, 0.25);
        const vec3 color4 = vec3(2.0, 1.25, 0.7);
        const vec3 color5 = vec3(2.0, 2.0, 2.0);
        const vec3 glowColor1 = vec3(1.5, 0.5, 0.0);
        const vec3 glowColor2 = vec3(1.5, 1.5, 0.5);
        const vec3 lightColor = vec3(1.0, 1.5, 0.75);
        const vec3 lightDirection = normalize(vec3(0.0, -1.0, 0.0));
        const float a = 0.125;
        const float b = 0.35;
        const float c = 0.5;
        
        vec3 gradient(float value) {{
            vec4 start = vec4(0.0, a, b, c);
            vec4 end = vec4(a, b, c, 1.0);
            vec4 mixValue = smoothstep(start, end, vec4(value));
            
            vec3 color = mix(color1, color2, mixValue.x);
            color = mix(color, color3, mixValue.y);
            color = mix(color, color4, mixValue.z);
            color = mix(color, color5, mixValue.w);
            
            return color;
        }}
        
        void main() {{
            vec2 uv = fragCoord / iResolution.xy;
            
            vec4 source = texture(iChannel0, uv);
            
            vec2 force = texture(iChannel1, uv).xy;
            force = DecodeForce(force);
            
            float value = length(force);
            
            float glow = source.w + source.z * 0.75;
            glow /= 2.0;
            
            vec3 color = gradient(value);
            color += mix(glowColor1, glowColor2, glow) * glow;
            
            vec3 normal = vec3(force.x, force.y, 1.0) * 0.5;
            normal = normalize(normal);
            
            float NdotL = smoothstep(-0.5, 0.5, dot(normal, lightDirection));
            color += color * NdotL * lightColor;
            
            fragColor = vec4(color, 1.0);
        }}
        """

screen_frag = open("screen_frag.glsl").read()


# create shader programs 

buffer_a_prog = ctx.program(vertex_shader= vertex_shader,fragment_shader= buffer_a_frag)
buffer_b_prog = ctx.program(vertex_shader= vertex_shader,fragment_shader= buffer_b_frag)
buffer_c_prog = ctx.program(vertex_shader= vertex_shader,fragment_shader= buffer_c_frag)
image_program = ctx.program(vertex_shader=vertex_shader,fragment_shader=image_frag)
to_screen_prog = ctx.program(vertex_shader= vertex_shader,fragment_shader= screen_frag)


# double buffers
bufferA = DoubleBuffer(ctx,size,4)
bufferB = DoubleBuffer(ctx,size,4)
bufferC = DoubleBuffer(ctx,size,4)


# noise texture 
z, y, x = np.mgrid[0:32, 0:32, 0:32]

noise = np.random.rand(32,32,32,4).astype(np.float32)
for c in range(4):
    noise[...,c] = gaussian_filter(noise[..., c], sigma=1)

noise -= noise.min()
noise /= noise.max()
noise *= 255

noise_3d = noise.astype(np.uint8)

noise_texture = ctx.texture3d((32,32,32),4,noise_3d.tobytes(),dtype='u1')
noise_texture.filter = (moderngl.LINEAR,moderngl.LINEAR)

noise_texture.repeat_x = True 
noise_texture.repeat_y = True 
noise_texture.repeat_z = True 


# setup quad vertices 
vertices = np.array([
    -1.0,-1.0,
    1.0, -1.0,
    1.0,1.0,
    -1.0 ,1.0,
],dtype = np.float32)

indices = np.array([0,1,2,0,2,3],dtype = np.uint32)

vbo = ctx.buffer(vertices.tobytes())
ibo = ctx.buffer(indices.tobytes())


# create vaos
vao_a = ctx.vertex_array(buffer_a_prog,[(vbo,'2f','in_position')],ibo)
vao_b = ctx.vertex_array(buffer_b_prog,[(vbo,'2f','in_position')],ibo)
vao_c = ctx.vertex_array(buffer_c_prog,[(vbo,'2f','in_position')],ibo)
vao_image = ctx.vertex_array(image_program,[(vbo,'2f','in_position')],ibo)

# test to screen draw vao 
vao_screen = ctx.vertex_array(to_screen_prog,[(vbo,'2f','in_position')],ibo)

clock = pygame.time.Clock()
# run the render loop


buffer_a_prog['iResolution'] = (float(size[0]),float(size[1]),1.0)

running = True
while running: 
    current_time = time.time() - start_time
    for event in pygame.event.get():
        if event.type == pygame.QUIT: 
            running = False 
        elif event.type == pygame.MOUSEMOTION:
            mouse_pos = pygame.mouse.get_pos()
        elif event.type == pygame.MOUSEBUTTONDOWN: 
            mouse_pressed = True
        elif event.type == pygame.MOUSEBUTTONUP: 
            mouse_pressed = False 
        elif event.type == pygame.KEYDOWN: 
            if event.key == pygame.K_ESCAPE: 
                running = False
        
    bufferA.fbo.use()
    ctx.clear(0.0,0.0,0.0,1.0)

    # update bufferA prog's uniforms 
    buffer_a_prog['iTime'] = current_time
    mouse_z = 1.0 if mouse_pressed else 0.0
    buffer_a_prog['iMouse'] = (mouse_pos[0],size[1] - mouse_pos[1],mouse_z,0.0)
    noise_texture.use(2)
    vao_a.render()
    bufferA.flip()

    bufferB.fbo.use()
    ctx.clear(0.0,0.0,0.0,1.0)    
    
    bufferA.tex.use(0)
    bufferC.previous_tex.use(1)
    
    vao_b.render()
    bufferB.flip()

    bufferC.fbo.use()
    buffer_c_prog['iTime'] = current_time

    bufferB.tex.use(1)
    noise_texture.use(2)   
    vao_c.render()

    bufferC.flip()


    # final render 
    ctx.screen.use()
    ctx.clear(0.0,0.0,0.0,1.0)
    bufferA.tex.use(0)
    bufferC.tex.use(1)

    vao_image.render()




    pygame.display.flip()
    clock.tick(60)
    # render the frame 

pygame.quit()
            
