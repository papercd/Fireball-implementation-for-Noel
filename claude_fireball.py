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
        pygame.display.set_caption("Fireball Projectile")
        
        self.ctx = moderngl.create_context()
        
        self.start_time = time.time()
        self.mouse_pos = (0.0, 0.0)
        
        # Game-specific variables
        self.projectiles = []  # List of active projectiles
        self.max_projectiles = 3  # Maximum simultaneous projectiles
        
        self.setup_background()
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
        
        # Modified Buffer A - Projectile System
        buffer_a_frag = f"""
        #version 330
        {common}
        uniform float iTime;
        uniform vec3 iResolution;
        uniform vec4 iProjectile1; // x,y = position, z = launch_time, w = active (1.0/0.0)
        uniform vec4 iProjectile2;
        uniform vec4 iProjectile3;
        uniform vec2 iProjectileDir1; // direction vector
        uniform vec2 iProjectileDir2;
        uniform vec2 iProjectileDir3;
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
        const float projectileSpeed = 10.0;
        const float projectileLifetime = 3.0;
        const float dissipationStart = 1.5;
        
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
        
        vec4 ProcessProjectile(vec2 uv, float ratio, vec4 projectile, vec2 direction) {{
            if (projectile.w < 0.5) return vec4(0.0); // Inactive projectile
            
            float timeSinceLaunch = iTime - projectile.z;
            if (timeSinceLaunch > projectileLifetime) return vec4(0.0); // Expired
            
            // Calculate current projectile position
            vec2 startPos = projectile.xy / iResolution.xy;
            vec2 currentPos = startPos + direction * projectileSpeed * timeSinceLaunch * 0.1;
            
            // Calculate distance from current fragment to projectile
            vec2 circleCoord = uv - currentPos;
            circleCoord.x *= ratio;
            float circle = length(circleCoord);
            
            // Create masks with dissipation over time
            float dissipationFactor = 1.0;
            if (timeSinceLaunch > dissipationStart) {{
                dissipationFactor = 1.0 - smoothstep(dissipationStart, projectileLifetime, timeSinceLaunch);
            }}
            
            vec4 masksIN = vec4(0.08, 0.35, 0.05, 0.2) * dissipationFactor;
            vec4 masksOUT = vec4(0.06, 0.0, 0.0, 0.0) * dissipationFactor;
            vec4 masksValue = vec4(circle, circle, circle, circle);
            vec4 masks = smoothstep(masksIN, masksOUT, masksValue);
            
            vec2 mask = masks.xy;
            
            vec4 noise = GetNoise(uv, ratio);
            
            // Create forces pointing opposite to projectile direction (trail effect)
            vec2 force = -direction * noise.xy * circleForceAmount * masks.x * dissipationFactor;
            force += (noise.xy - 0.5) * masks.x * randomForceAmount.x * dissipationFactor;
            force.y += (0.25 + 0.75 * noise.z) * masks.x * upForce.x * dissipationFactor * 0.5;
            
            force = EncodeForce(force);
            return vec4(force.x, force.y, mask.x * dissipationFactor, mask.y * dissipationFactor);
        }}
        
        void main() {{
            float ratio = iResolution.x / iResolution.y;
            vec2 uv = fragCoord / iResolution.xy;
            
            // Process all active projectiles and combine their effects
            vec4 result1 = ProcessProjectile(uv, ratio, iProjectile1, iProjectileDir1);
            vec4 result2 = ProcessProjectile(uv, ratio, iProjectile2, iProjectileDir2);
            vec4 result3 = ProcessProjectile(uv, ratio, iProjectile3, iProjectileDir3);
            
            // Combine results (you might want to blend them differently)
            vec4 finalResult = result1 + result2 + result3;
            
            // Clamp to prevent overflow
            finalResult.xy = clamp(finalResult.xy, 0.0, 1.0);
            finalResult.zw = clamp(finalResult.zw, 0.0, 1.0);
            
            fragColor = finalResult;
        }}
        """
        
        # Buffer B - Move Fluid (unchanged)
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
        
        # Buffer C - Update Fluid (unchanged)
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
        
        # Final Image shader (unchanged)
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
            
            const float redThreshold = 0.35;
           
            float alpha = value + glow + source.z + source.w;
            
            if (color.r < redThreshold){{
                discard;
            }}
            
            fragColor = vec4(color, 1.0);
        }}
        """

        screen_frag = """
        #version 330
        uniform sampler2D tex;
        in vec2 fragCoord;
        out vec4 fragColor;
        
        void main() {
            vec2 uv = fragCoord / vec2(800.0, 600.0);
            fragColor = texture(tex, uv);
        }
        """
        
        screen_vert = """
        #version 330
        in vec2 in_position;
        out vec2 fragCoord;
        
        void main() {
            gl_Position = vec4(in_position, 0.0, 1.0);
            fragCoord = (in_position + 1.0) * 0.5 * vec2(800.0, 600.0);
        }
        """
        
        # Create shader programs
        self.buffer_a_program = self.ctx.program(vertex_shader=vertex_shader, fragment_shader=buffer_a_frag)
        self.buffer_b_program = self.ctx.program(vertex_shader=vertex_shader, fragment_shader=buffer_b_frag)
        self.buffer_c_program = self.ctx.program(vertex_shader=vertex_shader, fragment_shader=buffer_c_frag)
        self.image_program = self.ctx.program(vertex_shader=vertex_shader, fragment_shader=image_frag)
        self.to_screen_program = self.ctx.program(vertex_shader=screen_vert, fragment_shader=screen_frag)

        
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

        def generate_rgba_noise_3d(size):
            noise = np.random.randint(0,256,(size,size,size,4),dtype = np.uint8)
            return noise 

        noise_3d = generate_rgba_noise_3d(size)
        noise_data = noise_3d.flatten()
    
        self.noise_texture = self.ctx.texture3d((size, size, size), 4, data = noise_data)
        self.noise_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.noise_texture.repeat_x = True
        self.noise_texture.repeat_y = True
        self.noise_texture.repeat_z = True

    
    def setup_background(self):
        # Create a simple gradient background if image not found
        try:
            from PIL import Image 
            img = Image.open("background_test.png")
            img_data = np.array(img)
            self.background_tex = self.ctx.texture((img.width,img.height),4,img_data.tobytes())
            self.background_tex.filter = (moderngl.LINEAR,moderngl.LINEAR)
        except:
            # Create simple gradient texture
            gradient_data = np.zeros((self.height, self.width, 4), dtype=np.uint8)
            for y in range(self.height):
                gradient_data[y, :, :3] = int(y / self.height * 50)  # Dark gradient
                gradient_data[y, :, 3] = 255
            self.background_tex = self.ctx.texture((self.width, self.height), 4, gradient_data.tobytes())
            self.background_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)

    
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
        self.screen_vao = self.ctx.vertex_array(self.to_screen_program, [(self.vbo, '2f', 'in_position')], self.ibo)
    
    def add_projectile(self, start_pos, target_pos):
        current_time = time.time() - self.start_time
        
        # Calculate direction vector
        dx = target_pos[0] - start_pos[0]
        dy = target_pos[1] - start_pos[1]
        length = math.sqrt(dx*dx + dy*dy)
        
        if length > 0:
            direction = (dx / length, dy / length)
        else:
            direction = (1.0, 0.0)  # Default direction
        
        # Create projectile data: (x, y, launch_time, active)
        projectile = {
            'position': start_pos,
            'direction': direction,
            'launch_time': current_time,
            'active': True
        }
        
        # Add to list (remove oldest if at capacity)
        if len(self.projectiles) >= self.max_projectiles:
            self.projectiles.pop(0)
        
        self.projectiles.append(projectile)
    
    def update_projectiles(self):
        current_time = time.time() - self.start_time
        lifetime = 3.0  # Should match shader constant
        
        # Remove expired projectiles
        self.projectiles = [p for p in self.projectiles 
                          if current_time - p['launch_time'] < lifetime]
    
    def update_uniforms(self, program):
        current_time = time.time() - self.start_time

        if program == self.buffer_a_program:
            # Update projectile uniforms
            self.update_projectiles()
            
            # Set up to 3 projectiles
            for i in range(3):
                proj_uniform = f'iProjectile{i+1}'
                dir_uniform = f'iProjectileDir{i+1}'
                
                if i < len(self.projectiles):
                    proj = self.projectiles[i]
                    program[proj_uniform] = (proj['position'][0], proj['position'][1], 
                                           proj['launch_time'], 1.0)
                    program[dir_uniform] = proj['direction']
                else:
                    program[proj_uniform] = (0.0, 0.0, 0.0, 0.0)  # Inactive
                    program[dir_uniform] = (0.0, 0.0)
            
            program['iChannel2'].value = 2
            self.noise_texture.use(location=2)

        if program == self.buffer_b_program: 
            program['iChannel0'].value = 0
            program['iChannel1'].value = 1

        if program == self.buffer_c_program: 
            program['iChannel1'].value = 1
            program['iChannel2'].value = 2

        if program == self.image_program:
            program['iChannel0'].value = 0
            program['iChannel1'].value = 1

        if program != self.buffer_b_program and program != self.image_program:
            program['iTime'] = current_time
        program['iResolution'] = (float(self.width), float(self.height), 1.0)
    
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
        
        # Render Buffer A (projectile forces)
        buffer_a_fbo_next.use()
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        self.update_uniforms(self.buffer_a_program)
        self.vao_a.render()

        # Render Buffer B (moves fluid)
        buffer_b_fbo_next.use()
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        self.update_uniforms(self.buffer_b_program)
        buffer_a_tex_next.use(0)
        buffer_c_tex_curr.use(1)
        self.vao_b.render()

        # Render Buffer C (updates fluid)
        buffer_c_fbo_next.use()
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        self.update_uniforms(self.buffer_c_program)
        buffer_b_tex_next.use(1)
        self.noise_texture.use(2)
        self.vao_c.render()
       
        # Render final image to screen
        self.ctx.screen.use()
        self.ctx.clear(0.0, 0.0, 0.0, 0.0)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self.update_uniforms(self.image_program)
        buffer_a_tex_next.use(0)
        buffer_c_tex_next.use(1)
        self.vao_image.render()
        self.ctx.disable(moderngl.BLEND)
        
        # Update frame counter
        self.frame_count += 1
    
    def run(self):
        clock = pygame.time.Clock()
        running = True
        
        print("Click to shoot fireballs!")
        print("Press 'C' to shoot towards center")
        print("Press 'R' to shoot random directions")
        print("ESC to quit")
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEMOTION:
                    self.mouse_pos = pygame.mouse.get_pos()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left click
                        # Shoot towards center
                        center = (self.width // 2, self.height // 2)
                        corrected_mouse_pos = (self.mouse_pos[0], self.height - self.mouse_pos[1])
                        self.add_projectile(corrected_mouse_pos, center)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_c:
                        # Shoot from mouse position towards center
                        center = (self.width // 2, self.height // 2)
                        self.add_projectile(self.mouse_pos, center)
                    elif event.key == pygame.K_r:
                        # Shoot in random direction from mouse position
                        import random
                        angle = random.random() * 2 * math.pi
                        target_distance = 200
                        target = (
                            self.mouse_pos[0] + math.cos(angle) * target_distance,
                            self.mouse_pos[1] + math.sin(angle) * target_distance
                        )
                        self.add_projectile(self.mouse_pos, target)

            self.render_frame()
            pygame.display.flip()
            clock.tick(60)
        
        pygame.quit()

if __name__ == "__main__":
    shader = FireballShader()
    shader.run()