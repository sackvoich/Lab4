import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from IPython.display import clear_output
import time

def solve_bvp(h):
    x = np.arange(0.3, 0.65 + h, h)
    n = len(x)
    main_diag = -2 - h**2 * np.ones(n)
    side_diag = 1 + h * np.ones(n-1)
    A = diags([side_diag, main_diag, side_diag], [-1, 0, 1], shape=(n, n)).tocsr()
    b = -0.4 * h**2 * np.ones(n)
    b[0] = 1 - (1 + h) * 0.3
    b[-1] = 2 * h
    A[0, 0] = 1
    A[0, 1] = 0
    A[-1, -1] = 1
    A[-1, -2] = 0
    y = spsolve(A, b)
    return x, y

def interactive_solve():
    epsilon = 0.1
    h = 0.2
    h_prev = 0.4
    min_h = 1e-6  # Минимально допустимый шаг
    
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    while True:
        x_1, y_1 = solve_bvp(h)
        x_2, y_2 = solve_bvp(h_prev)
        
        y_2_interp = np.interp(x_1, x_2, y_2)
        diff = np.abs(y_1 - y_2_interp)
        
        max_diff = np.max(diff)
        
        clear_output(wait=True)
        
        ax1.clear()
        ax1.plot(x_1, y_1, 'b-', label=f'h = {h:.6f}')
        ax1.plot(x_2, y_2, 'r--', label=f'h = {h_prev:.6f}')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_title('Решение краевой задачи')
        ax1.legend()
        ax1.grid(True)
        
        ax2.clear()
        ax2.plot(x_1, diff, 'g-')
        ax2.set_xlabel('x')
        ax2.set_ylabel('Разность')
        ax2.set_title(f'Разность между решениями (max = {max_diff:.6f})')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.draw()
        plt.pause(0.1)
        
        print(f"Текущий шаг h = {h:.6f}")
        print(f"Максимальная разность: {max_diff:.6f}")
        
        if max_diff < epsilon:
            print("Решение сходится с заданной точностью")
            break
        
        if h < min_h:
            print("Достигнут минимально допустимый шаг")
            break
        
        user_input = input("Нажмите Enter для продолжения или 'q' для выхода: ")
        if user_input.lower() == 'q':
            print("Вычисления прерваны пользователем")
            break
        
        h_prev = h
        h /= 2

    plt.ioff()
    plt.show()

interactive_solve()
