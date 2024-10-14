import numpy as np
import matplotlib.pyplot as plt
import mplcursors

class Kappa:
    def __init__(self):
        self.k1 = 0
        self.k2 = 0
        self.k3 = 0
        self.k4 = 0


def sysFun(x,num):
    if num == 0:
        return ( -10 * x[0] + x[1])
    elif num == 1:
        return (-2.02*x[0] - 2 * x[1])
    else:
        print("Error In Num")
        return


class Plotter():
    def __init__(self, t, y):
        self.t = t
        self.y = y

    def plot(self, eqNumber = '', grid = True):
        s = []
        legendTitle = []
        t = self.createAxis()

        if eqNumber == '':
            s = self.allSolutions   
        else:
            for k in self.allSolutions:
                s.append(k[eqNumber])

        plt.figure(figsize=(8,6))
        plt.style.use('seaborn-v0_8-deep')
            
        plt.plot(t, s, linewidth = 3)
        plt.xlabel("Time", fontsize=14, fontweight='bold', color='#555')
        plt.ylabel("x[{}]".format(eqNumber), fontsize=14, fontweight='bold', color='#555')
        plt.title("Solution", fontsize=18, fontweight='bold', color='#333')
        
        plt.xticks(fontsize=12, color='#444')
        plt.yticks(fontsize=12, color='#444')

        if grid: 
            plt.grid(which='both', color='gray', linestyle='--', linewidth=0.7)
            plt.minorticks_on()  # Enable minor ticks
            plt.grid(which='minor', color='gray', linestyle=':', linewidth=0.5, alpha=0.7)  # Customize minor gridlines

        for i in range(len(s)):
            legendTitle.append("x[{}]".format(i))

        plt.legend(legendTitle,loc='upper right', fancybox=True, shadow=True, ncol=2)  # Multi-column legends

        mplcursors.cursor()  # Enable hovering for values

        plt.tight_layout()

        plt.show()

    def phasePortrait(self, eqNumber, grid = True):
        x = []
        x1 = []
        x2 = []
        xDot = []

        for i in range(self.MATRIX_SIZE):
            for k in self.allSolutions:
                # x1.append(k[eqNumber])
                # x2.append(k[eqNumber + 1])
                # x.append()
                pass

        for k in range(len(x1)):
            xDot.append(sysFun([x1[k], x2[k]],eqNumber))

        x1d = [d * (-1) for d in x1]
        xDotd = [d * (-1) for d in xDot]


        plt.figure(figsize=(8,6))
        plt.style.use('seaborn-v0_8-deep')
            
        plt.plot(x1, xDot, linewidth = 3)
        plt.plot(x1d, xDotd, linewidth = 3)
        plt.xlabel("Time", fontsize=14, fontweight='bold', color='#555')
        plt.ylabel("x[{}]".format(eqNumber), fontsize=14, fontweight='bold', color='#555')
        plt.title("Solution", fontsize=18, fontweight='bold', color='#333')
        
        plt.xticks(fontsize=12, color='#444')
        plt.yticks(fontsize=12, color='#444')

        if grid: 
            plt.grid(which='both', color='gray', linestyle='--', linewidth=0.7)
            plt.minorticks_on()  # Enable minor ticks
            plt.grid(which='minor', color='gray', linestyle=':', linewidth=0.5, alpha=0.7)  # Customize minor gridlines

        # for i in range(len(s)):
            # legendTitle.append("x[{}]".format(i))

        # plt.legend(legendTitle,loc='upper right', fancybox=True, shadow=True, ncol=2)  # Multi-column legends

        mplcursors.cursor()  # Enable hovering for values

        plt.tight_layout()

        plt.show()


    def createAxis(self):
        return  np.linspace(0,self.MAXSTOPTIME, self.ITERATIONS)

class Solver(Plotter):
    def __init__(self, sysFun, x, STEP, MAXSTOPTIME):

        self.sysFun = sysFun
        self.x = x
        self.allSolutions = []
        self.STEP = STEP
        self.MAXSTOPTIME = MAXSTOPTIME
        self.ITERATIONS =  int(self.MAXSTOPTIME / self.STEP)
        self.MATRIX_SIZE = len(x)

    def calcuateKappa(self, num):
        xForK = np.zeros(self.MATRIX_SIZE)
        K = Kappa()
        
        K.k1 = self.sysFun(self.x, num)

        for i in range(self.MATRIX_SIZE):
            if i != num:
                xForK[i] = self.x[i] + 0.5 * self.STEP
            else:
                xForK[i] = self.x[i] + 0.5 * K.k1 * self.STEP

        K.k2 = self.sysFun(xForK, num)
        
        xForK[num] = self.x[num] + 0.5 * K.k2 * self.STEP
        
        K.k3 = self.sysFun(xForK, num)

        for i in range(self.MATRIX_SIZE):
            if i != num:
                xForK[i] = self.x[i] + self.STEP
            else:
                xForK[i] = self.x[i] + K.k3 * self.STEP
        
        K.k4 = self.sysFun(xForK, num)
        return K

    def solve(self):
        k = 0
        for k in range(self.ITERATIONS):
            # print("Iteration No", k + 1)
            
            for i in range(self.MATRIX_SIZE):
                K = self.calcuateKappa(i)
                # print("K1 = {:.10f}, K2 = {:.10f}, K3 = {:.10f}, K4 = {:.10f}".format(K.k1, K.k2, K.k3, K.k4))
                self.x[i] = self.x[i] + (K.k1 + 2*K.k2 + 2*K.k3 + K.k4)*(self.STEP/6)
                # print("x[{}] = {:.10f}".format(i, self.x[i]))
            # print()
            self.allSolutions.append(np.copy(self.x))

        # print("Solution Found In {} self.ITERATIONS".format(k + 1))
        # for i in range(self.MATRIX_SIZE):
            # print("x[{}] = {:.5f}".format(i, self.x[i]))

        return (self.x)
    
    def getAllSolutions(self):
        return self.allSolutions
    
    def printSolution(self):
        print("Solution for step size {}, total iterations {}".format(self.STEP, self.ITERATIONS))
        for k in range(self.MATRIX_SIZE):
            print("x[{}] = {:.5f}".format(k, self.x[k]))



def main():
    x = np.array([0.0,30.0])

    solver = Solver(sysFun, x,0.0001, 2)
    x = solver.solve()

    # solver.plot()
    solver.phasePortrait(0)
if __name__ == "__main__":
    main()

