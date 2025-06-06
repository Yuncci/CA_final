import numpy as np
import pygame

# 模擬參數
N = 64
dt = 0.1
diff = 0.0001
visc = 0.0001
scale = 8
frame = 0

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

# 中央障礙物
obstacle = np.zeros((N, N), dtype=bool)
obstacle[N//2:N//2+4, N//2-6:N//2+6] = True

wind_dir = [0, 15]  # 初始為向上吹（v方向負）

hole_positions = []  # 吸力洞列表
source_on = True  # 預設打開風源

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
    global dens_r, dens_g, dens_b
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
    u[0, :] = u[-1, :] = u[:, 0] = u[:, -1] = 0
    v[0, :] = v[-1, :] = v[:, 0] = v[:, -1] = 0
    decay_rate = 0.99  # 可微調（建議範圍 0.99 ~ 0.999）
    dens_r *= decay_rate
    dens_g *= decay_rate
    dens_b *= decay_rate


def add_initial_conditions():
    cx, cy = N // 2, 6
    size = 4
    ux, uy = wind_dir

    if source_on:
        dens_r_prev[cx - size:cx + size, cy - size:cy + size] += 400
        dens_g_prev[cx - size:cx + size, cy + 4:cy + 8] += 400
        dens_b_prev[cx - size:cx + size, cy - 8:cy - 4] += 400

    u_prev[cx - size:cx + size, cy - size:cy + size] += ux
    v_prev[cx - size:cx + size, cy - size:cy + size] += uy


# ---------- pygame 初始化 ----------
pygame.init()
win = pygame.display.set_mode((N * scale, N * scale))
pygame.display.set_caption("Stable Fluids - RGB + Quiver + 點擊注入藍煙霧")
clock = pygame.time.Clock()


running = True
while running:
    frame += 1
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif pygame.mouse.get_pressed()[0]:  # 左鍵按住
            x, y = pygame.mouse.get_pos()
            i = y // scale
            j = x // scale
            if 3 <= i < N-3 and 3 <= j < N-3:
                dens_b_prev[i-3:i+4, j-3:j+4] += 2000
                u_prev[i-3:i+4, j-3:j+4] += 8

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                wind_dir[0] += -10
                wind_dir[1] += 0
            elif event.key == pygame.K_DOWN:
                wind_dir[0] += 10
                wind_dir[1] += 0
            elif event.key == pygame.K_s:
                source_on = not source_on  # 切換風源狀態
                print(f"煙霧 {'開啟' if source_on else '關閉'}")
        
        elif event.type == pygame.MOUSEBUTTONDOWN:
            x, y = pygame.mouse.get_pos()
            i = y // scale
            j = x // scale
            if event.button == 3:
                hole_positions.append((i, j))

    u_prev.fill(0)
    v_prev.fill(0)
    if source_on:
        add_initial_conditions()
    for hi, hj in hole_positions:
        for i in range(hi-10, hi+11):
            for j in range(hj-10, hj+11):
                if 0 <= i < N and 0 <= j < N:
                    dx = hj - j
                    dy = hi - i
                    dist = np.sqrt(dx*dx + dy*dy) + 1e-5
                    strength = 50 / (dist**2 + 1)

                    u_prev[i, j] += strength * dx / dist
                    v_prev[i, j] += strength * dy / dist

                    if dist < 10:
                        dens_r[i, j] = 0
                        dens_g[i, j] = 0
                        dens_b[i, j] = 0
                    else:
                        dens_r[i, j] *= 0.5
                        dens_g[i, j] *= 0.5
                        dens_b[i, j] *= 0.5



    step()

    rgb_img = np.stack([
        np.clip(dens_r, 0, 500),
        np.clip(dens_g, 0, 500),
        np.clip(dens_b, 0, 500)
    ], axis=-1) / 500.0
    rgb_img = (rgb_img * 255).astype(np.uint8)

    surf = pygame.surfarray.make_surface(np.transpose(rgb_img, (1, 0, 2)))
    surf = pygame.transform.scale(surf, (N * scale, N * scale))
    win.blit(surf, (0, 0))

    # 顯示吸力洞位置
    for hi, hj in hole_positions:
        pygame.draw.circle(win, (255, 100, 100), (hj*scale, hi*scale), 50, 1)

    # quiver 顯示速度箭頭
    step_size = 1
    arrow_color = (255, 255, 255)
    arrow_scale = 1.0 * scale
    for i in range(0, N, step_size):
        for j in range(0, N, step_size):
            if obstacle[i, j]:
                continue
            px = j * scale
            py = i * scale
            dx = u[i, j] * arrow_scale
            dy = v[i, j] * arrow_scale
            pygame.draw.line(win, arrow_color, (px, py), (px + dx, py + dy), 1)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
