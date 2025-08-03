import moderngl
import pygame
import numpy as np
from pygame.locals import *
from PIL import Image

# === CONFIG ===
width, height = 800, 600
depth = 32  # depth of 3D texture
slice_index = 0  # initial slice

# === Setup Pygame + ModernGL Context ===
pygame.init()
pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL)
ctx = moderngl.create_context()

# === Load Shaders ===
prog = ctx.program(
    vertex_shader=open("slice_view.vert").read(),
    fragment_shader=open("slice_view.frag").read(),
)

# === Setup Fullscreen Quad ===
quad = np.array([
    -1.0, -1.0,
     1.0, -1.0,
    -1.0,  1.0,
    -1.0,  1.0,
     1.0, -1.0,
     1.0,  1.0,
], dtype='f4')

vbo = ctx.buffer(quad.tobytes())
vao = ctx.simple_vertex_array(prog, vbo, 'in_position')

# === Generate a test 3D texture ===
def make_3d_noise_texture(size=32):
    data = (np.random.rand(size, size, size, 4) * 255).astype('u1')
    tex = ctx.texture3d((size, size, size), 4, data.tobytes())
    tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
    tex.repeat_x = tex.repeat_y = tex.repeat_z = True
    return tex

tex3D = make_3d_noise_texture()
prog['tex3D'] = 0  # bound to texture unit 0

# === Main Loop ===
running = True
clock = pygame.time.Clock()
while running:
    for event in pygame.event.get():
        if event.type == QUIT:
            running = False
        if event.type == KEYDOWN:
            if event.key == K_UP:
                slice_index = min(slice_index + 1, depth - 1)
            elif event.key == K_DOWN:
                slice_index = max(slice_index - 1, 0)

    ctx.clear(0.0, 0.0, 0.0)
    tex3D.use(location=0)

    prog['sliceIndex'].value = slice_index / (depth - 1)
    vao.render()
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
