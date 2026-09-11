import cma
import numpy as np
import time

# --- 1. 定义黑盒函数 (Ackley Function) ---
# 这是一个经典的测试函数，在 x = [0, 0, ..., 0] 处有全局最小值 0。
# 它有许多局部最优，对优化算法构成了挑战。
def ackley_function(x):
    """
    Ackley-Funktion für n Dimensionen.
    x: Ein numpy-Array.
    """
    n = len(x)
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(2 * np.pi * x))
    term1 = -20 * np.exp(-0.2 * np.sqrt(sum1 / n))
    term2 = -np.exp(sum2 / n)
    return term1 + term2 + 20 + np.e

# --- 2. 创建多种群CMA-ES管理器 ---
class MultiPopulationCMAES:
    """
    一个使用多种群思想来运行CMA-ES的管理器。
    
    参数:
    - objective_function: 需要被最小化的黑盒目标函数。
    - dimension: 目标函数输入的维度。
    - n_populations: 要创建的种群（岛屿）数量。
    - population_size: 每个CMA-ES种群的大小 (lambda)。
    - sigma0: 初始标准差（探索步长）。
    - search_bounds: [min_val, max_val]，用于随机初始化种群的起始点。
    - migration_interval: 整数，每隔多少代进行一次“迁移”。
    """
    def __init__(self, objective_function, dimension, n_populations=5, population_size=10, sigma0=0.5, search_bounds=[-15, 15], migration_interval=25):
        self.objective_function = objective_function
        self.dimension = dimension
        self.n_populations = n_populations
        self.population_size = population_size
        self.sigma0 = sigma0
        self.search_bounds = search_bounds
        self.migration_interval = migration_interval
        
        self.populations = []
        self.global_best_solution = None
        self.global_best_fitness = float('inf')
        
        # 初始化所有种群
        self._initialize_populations()

    def _initialize_populations(self):
        """为每个岛屿创建并初始化一个CMA-ES实例。"""
        print(f"初始化 {self.n_populations} 个种群...")
        for i in range(self.n_populations):
            # 在搜索空间内随机生成一个起始点，以增加多样性
            x0 = np.random.uniform(self.search_bounds[0], self.search_bounds[1], self.dimension)
            
            # CMA-ES的选项
            opts = cma.CMAOptions()
            opts['popsize'] = self.population_size
            opts['bounds'] = self.search_bounds
            opts['verbose'] = -9 # 关闭每个实例的独立输出
            
            # 创建CMA-ES实例并添加到列表中
            es = cma.CMAEvolutionStrategy(x0, self.sigma0, opts)
            self.populations.append(es)
            print(f"  种群 {i+1}/{self.n_populations} 已创建，起始点: {np.round(x0, 2)}")

    def run(self, max_generations=200):
        """
        运行整个优化过程。
        """
        start_time = time.time()
        print(f"\n开始优化过程，最大代数: {max_generations}\n" + "="*40)
        
        for g in range(max_generations):
            all_current_bests = []
            
            # --- 并行进化每个种群 ---
            for i, es in enumerate(self.populations):
                # 1. 'ask' 获取新一代的候选解
                solutions = es.ask()
                
                # 2. 评估每个候选解的适应度（调用黑盒函数）
                fitnesses = [self.objective_function(s) for s in solutions]
                print(solutions)
                print(fitnesses)
                # 3. 'tell' 将解和对应的适应度返回给CMA-ES，以更新其内部状态
                es.tell(solutions, fitnesses)
                
                # 跟踪每个种群当前的最优解
                all_current_bests.append((es.result.fbest, es.result.xbest))

            # --- 更新全局最优解 ---
            current_best_fitness, current_best_solution = min(all_current_bests, key=lambda item: item[0])
            if current_best_fitness < self.global_best_fitness:
                self.global_best_fitness = current_best_fitness
                self.global_best_solution = current_best_solution
            
            # --- 定期进行迁移 ---
            if (g + 1) % self.migration_interval == 0:
                self._perform_migration()

            # --- 打印进度 ---
            if (g + 1) % 10 == 0:
                print(f"代: {g+1:4d} | 全局最优适应度: {self.global_best_fitness:.6f}")

        end_time = time.time()
        print("="*40 + "\n优化完成！")
        print(f"总耗时: {end_time - start_time:.2f} 秒")
        print(f"找到的最优解 (x): {self.global_best_solution}")
        print(f"对应的函数值 (f(x)): {self.global_best_fitness}")
        
        return self.global_best_solution, self.global_best_fitness

    def _perform_migration(self):
        """
        执行迁移操作：
        将全局最优解注入到一个随机选择的、表现较差的种群中，以帮助其跳出局部最优。
        """
        print(f"\n--- 第 {self.populations[0].countiter} 代，执行迁移 ---")
        
        # 找到表现最差的种群（不是必须的，但可以作为一种策略）
        fitnesses = [es.result.fbest for es in self.populations]
        worst_pop_index = np.argmax(fitnesses)
        
        # 将该种群的平均值（下一代采样的中心）重置为已知的全局最优解
        # 这是一种“软”注入，引导该种群向更有希望的区域探索
        print(f"将全局最优解 (适应度: {self.global_best_fitness:.4f}) 注入到种群 {worst_pop_index+1}")
        self.populations[worst_pop_index].mean = self.global_best_solution.copy()
        # 也可以考虑重置其步长，以鼓励在新区域进行更精细的搜索
        # self.populations[worst_pop_index].sigma = self.sigma0 
        print("--- 迁移完成 ---\n")


# --- 3. 主执行脚本 ---
if __name__ == '__main__':
    # --- 参数设置 ---
    PROBLEM_DIMENSION = 10      # 黑盒函数的参数维度
    N_POPULATIONS = 4           # 种群数量
    POPULATION_SIZE = 15        # 每个种群的大小
    SIGMA0 = 5.0                # 初始探索步长（对于大范围搜索，可以设大一些）
    MAX_GENERATIONS = 250       # 最大进化代数
    MIGRATION_INTERVAL = 40     # 每40代进行一次种群间信息迁移
    SEARCH_BOUNDS = [-30, 30]   # 参数的搜索范围

    # --- 初始化并运行优化器 ---
    optimizer = MultiPopulationCMAES(
        objective_function=ackley_function,
        dimension=PROBLEM_DIMENSION,
        n_populations=N_POPULATIONS,
        population_size=POPULATION_SIZE,
        sigma0=SIGMA0,
        search_bounds=SEARCH_BOUNDS,
        migration_interval=MIGRATION_INTERVAL
    )
    
    best_params, best_value = optimizer.run(max_generations=MAX_GENERATIONS)
    
    # --- 验证结果 ---
    # Ackley函数的理论最优解是全零向量，值为0
    print("\n--- 结果分析 ---")
    print(f"理论最优值: 0.0")
    print(f"算法找到的最优值: {best_value}")
    print(f"理论最优参数: [0., 0., ...]")
    # 打印部分找到的参数
    print(f"算法找到的参数 (前5维): {np.round(best_params[:5], 5)}")