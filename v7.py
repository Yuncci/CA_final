import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

# 模擬參數
N = 64
dt = 0.1
diff = 0.0001
visc = 0.0001

# 初始化場域
dens_r = np.zeros((N, N))
dens_g = np.zeros((N, N))
dens_b = np.zeros((N, N))
dens_r_prev = np.zeros_like(dens_r)
dens_g_prev = np.zeros_like(dens_g)
dens_b_prev = np.zeros_like(dens_b)
u = np.zeros((N, N))
v = np.zeros((N, N))
u_prev = np.zeros_like(u)
v_prev = np.zeros_like(v)

# 障礙物：中央橫牆
obstacle = np.zeros((N, N), dtype=bool)
obstacle[N//2:N//2+4, N//2-6:N//2+6] = True

def add_source(x, s): x += dt * s; x[obstacle] = 0

def diffuse(b, x, x0, diff):
    a = dt * diff * N * N
    for _ in range(20):
        x[1:-1,1:-1] = (x0[1:-1,1:-1] + a * (
            x[2:,1:-1] + x[:-2,1:-1] + x[1:-1,2:] + x[1:-1,:-2])) / (1 + 4*a)
        x[obstacle] = 0

def advect(b, d, d0, u, v):
    for i in range(1,N-1):
        for j in range(1,N-1):
            x = i - dt * u[i,j] * N
            y = j - dt * v[i,j] * N
            x = min(max(x, 0.5), N - 1.5)
            y = min(max(y, 0.5), N - 1.5)
            i0, j0 = int(x), int(y)
            i1, j1 = i0+1, j0+1
            s1, t1 = x - i0, y - j0
            s0, t0 = 1 - s1, 1 - t1
            d[i,j] = (s0 * (t0 * d0[i0,j0] + t1 * d0[i0,j1]) +
                      s1 * (t0 * d0[i1,j0] + t1 * d0[i1,j1]))
    d[obstacle] = 0

def project(u, v, p, div):
    div[1:-1,1:-1] = -0.5 * (u[2:,1:-1] - u[:-2,1:-1] + v[1:-1,2:] - v[1:-1,:-2]) / N
    p.fill(0)
    for _ in range(20):
        p[1:-1,1:-1] = (div[1:-1,1:-1] + p[2:,1:-1] + p[:-2,1:-1] +
                        p[1:-1,2:] + p[1:-1,:-2]) / 4
    u[1:-1,1:-1] -= 0.5 * N * (p[2:,1:-1] - p[:-2,1:-1])
    v[1:-1,1:-1] -= 0.5 * N * (p[1:-1,2:] - p[1:-1,:-2])
    u[obstacle] = 0
    v[obstacle] = 0

def step():
    for d, dp in zip((dens_r, dens_g, dens_b), (dens_r_prev, dens_g_prev, dens_b_prev)):
        add_source(d, dp)
        dp[:,:] = d
        diffuse(0, d, dp, diff)
        dp[:,:] = d
        advect(0, d, dp, u, v)

    add_source(u, u_prev)
    add_source(v, v_prev)
    u_prev[:,:] = u
    diffuse(1, u, u_prev, visc)
    v_prev[:,:] = v
    diffuse(2, v, v_prev, visc)
    project(u, v, np.zeros_like(u), np.zeros_like(v))
    u_prev[:,:] = u
    v_prev[:,:] = v
    advect(1, u, u_prev, u_prev, v_prev)
    advect(2, v, v_prev, u_prev, v_prev)
    project(u, v, np.zeros_like(u), np.zeros_like(v))

# 多色煙霧風源
def add_initial_conditions(frame):
    cx, cy = N // 2, 6
    size = 4
    strength = 10
    theta = np.pi / 2 + (np.pi / 3) * np.sin(0.1 * frame)
    ux = strength * np.cos(theta)
    uy = strength * np.sin(theta)
    dens_r_prev[cx - size:cx + size, cy - size:cy + size] += 400  # 紅
    dens_g_prev[cx - size:cx + size, cy + 4:cy + 8] += 400        # 綠（偏右）
    dens_b_prev[cx - size:cx + size, cy - 8:cy - 4] += 400        # 藍（偏左）
    u_prev[cx - size:cx + size, cy - size:cy + size] += ux
    v_prev[cx - size:cx + size, cy - size:cy + size] += uy

fig, ax = plt.subplots()
rgb_img = np.zeros((N, N, 3))
im = ax.imshow(rgb_img, origin='lower', vmin=0, vmax=500)
Y, X = np.mgrid[0:N, 0:N]
quiv = ax.quiver(X, Y, v, u, color='white', scale=20)
obstacle_mask = ax.imshow(obstacle, cmap='gray', alpha=0.3, origin='lower')

def update(frame):
    u_prev.fill(0)
    v_prev.fill(0)
    add_initial_conditions(frame)
    step()
    rgb_img[..., 0] = np.clip(dens_r, 0, 500)
    rgb_img[..., 1] = np.clip(dens_g, 0, 500)
    rgb_img[..., 2] = np.clip(dens_b, 0, 500)
    im.set_array(rgb_img / 500.0)  # normalize
    quiv.set_UVC(v, u)
    return [im, quiv, obstacle_mask]

ani = animation.FuncAnimation(fig, update, frames=300, interval=50)

# 儲存動畫為 GIF（不需 ffmpeg）
from matplotlib.animation import PillowWriter

writer = PillowWriter(fps=30)
ani.save("fluid_sim.gif", writer=writer)
print("動畫已儲存為 fluid_sim.gif")

