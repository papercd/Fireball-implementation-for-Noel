import moderngl
import numpy as np
import pygame
import math
import time
from PIL import Image
import os

class FireballShader:
    def __init__(self, width=800, height=600):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height), pygame.OPENGL | pygame.DOUBLEBUF)
        pygame.display.set_caption("Fireball Shader")
        
        self.ctx = moderngl.create_context()
        #self.ctx.enable(moderngl.BLEND)
        #self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        
        self.start_time = time.time()
        self.mouse_pos = (0.0, 0.0)
        self.mouse_pressed = False
        
        self.setup_shaders()
        self.setup_buffers()
        self.setup_noise_texture()
        self.setup_quad()
        
    def setup_shaders(self):
        # Common functions
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
        
        void main() {
            gl_Position = vec4(in_position, 0.0, 1.0);
            fragCoord = (in_position + 1.0) * 0.5 * vec2(800.0, 600.0);
        }
        """
        
        # Buffer A - Init Fluid
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
        
        # Buffer B - Move Fluid
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
        
        # Buffer C - Update Fluid
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
        
        # Final Image shader
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
        
        # Create shader programs
        self.buffer_a_program = self.ctx.program(vertex_shader=vertex_shader, fragment_shader=buffer_a_frag)
        self.buffer_b_program = self.ctx.program(vertex_shader=vertex_shader, fragment_shader=buffer_b_frag)
        self.buffer_c_program = self.ctx.program(vertex_shader=vertex_shader, fragment_shader=buffer_c_frag)
        self.image_program = self.ctx.program(vertex_shader=vertex_shader, fragment_shader=image_frag)
        
    def setup_buffers(self):
        # Create textures for each buffer (double buffering for feedback)
        self.buffer_a_tex_0 = self.ctx.texture((self.width, self.height), 4)
        self.buffer_a_tex_1 = self.ctx.texture((self.width, self.height), 4)
        self.buffer_b_tex_0 = self.ctx.texture((self.width, self.height), 4)
        self.buffer_b_tex_1 = self.ctx.texture((self.width, self.height), 4)
        self.buffer_c_tex_0 = self.ctx.texture((self.width, self.height), 4)
        self.buffer_c_tex_1 = self.ctx.texture((self.width, self.height), 4)
        
        # Create framebuffers
        self.buffer_a_fbo_0 = self.ctx.framebuffer(self.buffer_a_tex_0)
        self.buffer_a_fbo_1 = self.ctx.framebuffer(self.buffer_a_tex_1)
        self.buffer_b_fbo_0 = self.ctx.framebuffer(self.buffer_b_tex_0)
        self.buffer_b_fbo_1 = self.ctx.framebuffer(self.buffer_b_tex_1)
        self.buffer_c_fbo_0 = self.ctx.framebuffer(self.buffer_c_tex_0)
        self.buffer_c_fbo_1 = self.ctx.framebuffer(self.buffer_c_tex_1)
        
        # Set texture filtering
        for tex in [self.buffer_a_tex_0, self.buffer_a_tex_1, self.buffer_b_tex_0, 
                   self.buffer_b_tex_1, self.buffer_c_tex_0, self.buffer_c_tex_1]:
            tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            tex.repeat_x = True
            tex.repeat_y = True
        
        # Initialize ping-pong state
        self.frame_count = 0
    
    def setup_noise_texture(self):
        import numpy as np
        size = 32

        # Generate simple coherent 3D RGBA noise (per-channel randomized)
        def generate_rgba_noise_3d(size):
            # Random per-channel 3D noise
            noise = np.random.rand(size, size, size, 4).astype(np.float32)

            

            # Normalize each channel to 0–255 uint8
            noise -= noise.min()
            noise /= noise.max()
            noise *= 255
           
            return noise.astype(np.uint8)

        # Generate the noise
        noise_3d = generate_rgba_noise_3d(size)

        # Upload to moderngl
        self.noise_texture = self.ctx.texture3d((size, size, size), 4, noise_3d.tobytes(), dtype='u1')
        self.noise_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.noise_texture.repeat_x = True
        self.noise_texture.repeat_y = True
        self.noise_texture.repeat_z = True

    
    def setup_quad(self):
        # Create a full-screen quad
        vertices = np.array([
            -1.0, -1.0,
             1.0, -1.0,
             1.0,  1.0,
            -1.0,  1.0,
        ], dtype=np.float32)
        
        indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)
        
        self.vbo = self.ctx.buffer(vertices.tobytes())
        self.ibo = self.ctx.buffer(indices.tobytes())
        
        # Create VAOs for each shader program
        self.vao_a = self.ctx.vertex_array(self.buffer_a_program, [(self.vbo, '2f', 'in_position')], self.ibo)
        self.vao_b = self.ctx.vertex_array(self.buffer_b_program, [(self.vbo, '2f', 'in_position')], self.ibo)
        self.vao_c = self.ctx.vertex_array(self.buffer_c_program, [(self.vbo, '2f', 'in_position')], self.ibo)
        self.vao_image = self.ctx.vertex_array(self.image_program, [(self.vbo, '2f', 'in_position')], self.ibo)
    
    def update_uniforms(self, program):
        current_time = time.time() - self.start_time
        
        if program != self.buffer_b_program and program != self.image_program:
            program['iTime'] = current_time
        program['iResolution'] = (float(self.width), float(self.height), 1.0)
        
        if program != self.buffer_b_program and program != self.buffer_c_program and program != self.image_program:
            mouse_z = 1.0 if self.mouse_pressed else 0.0
            program['iMouse'] = (self.mouse_pos[0], self.height - self.mouse_pos[1], mouse_z, 0.0)
    
    def render_frame(self):
        # Get current and next buffer indices for ping-pong
        curr = self.frame_count % 2
        next_idx = (self.frame_count + 1) % 2
        
        # Get current buffer textures and fbos
        buffer_a_tex_curr = self.buffer_a_tex_0 if curr == 0 else self.buffer_a_tex_1
        buffer_a_tex_next = self.buffer_a_tex_1 if curr == 0 else self.buffer_a_tex_0
        buffer_a_fbo_next = self.buffer_a_fbo_1 if curr == 0 else self.buffer_a_fbo_0
        
        buffer_b_tex_curr = self.buffer_b_tex_0 if curr == 0 else self.buffer_b_tex_1
        buffer_b_tex_next = self.buffer_b_tex_1 if curr == 0 else self.buffer_b_tex_0
        buffer_b_fbo_next = self.buffer_b_fbo_1 if curr == 0 else self.buffer_b_fbo_0
        
        buffer_c_tex_curr = self.buffer_c_tex_0 if curr == 0 else self.buffer_c_tex_1
        buffer_c_tex_next = self.buffer_c_tex_1 if curr == 0 else self.buffer_c_tex_0
        buffer_c_fbo_next = self.buffer_c_fbo_1 if curr == 0 else self.buffer_c_fbo_0
        
        # Render Buffer A (always generates new forces)
        buffer_a_fbo_next.use()
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        self.update_uniforms(self.buffer_a_program)
        self.noise_texture.use(2)  # iChannel2
        self.vao_a.render()

       

        
        # Render Buffer B (moves fluid based on Buffer A and previous Buffer C)
        buffer_b_fbo_next.use()
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        self.update_uniforms(self.buffer_b_program)
        buffer_a_tex_next.use(0)  # iChannel0 - current Buffer A
        buffer_c_tex_curr.use(1)  # iChannel1 - previous Buffer C
        self.vao_b.render()
        
       

         # Render Buffer C (updates fluid based on current Buffer B)
        buffer_c_fbo_next.use()
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        self.update_uniforms(self.buffer_c_program)
        buffer_b_tex_next.use(1)  # iChannel1 - current Buffer B
        self.noise_texture.use(2)  # iChannel2
        self.vao_c.render()
       
        # Render final image to screen
        self.ctx.screen.use()
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        self.update_uniforms(self.image_program)
        buffer_a_tex_next.use(0)  # iChannel0 - current Buffer A
        buffer_c_tex_next.use(1)  # iChannel1 - current Buffer C
        self.vao_image.render()

        
        # Update frame counter
        self.frame_count += 1
    
    def run(self):
        clock = pygame.time.Clock()
        running = True
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEMOTION:
                    self.mouse_pos = pygame.mouse.get_pos()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.mouse_pressed = True
                elif event.type == pygame.MOUSEBUTTONUP:
                    self.mouse_pressed = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
            
            self.render_frame()
            pygame.display.flip()
            clock.tick(60)
        
        pygame.quit()

if __name__ == "__main__":
    shader = FireballShader()
    shader.run()