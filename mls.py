import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import t as student_t
from scipy.stats import f as fstat


class mls():
    def __init__(self, data, y=None, x=None, nocon=False) -> None:

        
        self.Y = data[[y]].to_numpy()
        self.N = len(self.Y)
        self.ylabel = y

        if nocon:
            self.X = data[x].to_numpy()
            self.beta_names = x
        else:
            self.X = np.hstack( [np.ones((self.N,1)), data[x].to_numpy()])
            self.beta_names = ['Constant'] + x
        
        Xtrans = self.X.T
        XTX_inv = np.linalg.inv(Xtrans.dot(self.X))

        self.beta_hat = XTX_inv.dot(Xtrans).dot(self.Y)
        
        self.nparams = self.X.shape[1]
        self.dof_res = self.N - self.nparams
        self.dof_tot = self.N - 1

        self.y_hat = self.X.dot(self.beta_hat)
        self.residuals = self.Y-self.y_hat
        self.SSE = (self.residuals**2).sum()
        self.SST = ((self.Y - self.Y.mean())**2).sum()
        self.SSR = ((self.y_hat - self.Y.mean())**2).sum()
        self.SXX = ((self.X - self.X.mean(axis=0))**2).sum(axis=0)

        self.r_squared = self.SSR/self.SST
        self.rbar_squared = 1.0 - (self.SSE/self.dof_res)/(self.SST/self.dof_tot)
        self.sigma = np.sqrt(self.SSE/self.dof_res)
        self.beta_hat_sigma = self.sigma*np.sqrt(np.diag(XTX_inv))
        self.beta_hat_t = self.beta_hat[:,0]/self.beta_hat_sigma
        self.beta_hat_p  = (1-student_t.cdf(np.abs(self.beta_hat_t),self.dof_res))*2
        self.dof_F = ((self.dof_tot - self.dof_res),self.dof_res )
        self.F = ((self.SST-self.SSE)/self.dof_F[0])/ \
                    (self.SSE/self.dof_F[1])
        self.F_p = 1-fstat.cdf(self.F, self.dof_F[0], self.dof_F[1])

    def __repr__(self) -> str:
        s = 'MLS from __repr__'
        return s
    
    def __str__(self) -> str:
        p_df = pd.DataFrame({'Coef':self.beta_hat[:,0], 'SE_Coef':self.beta_hat_sigma,
                             'T-Value':self.beta_hat_t, 'p-Value':self.beta_hat_p},
                             index = pd.Index(self.beta_names, name='Variable'))
        s = str(p_df)

        rsq = f'\n\nR-squared: {self.r_squared:.4f}, R-bar-squared: {self.rbar_squared:.4f}' +\
            f'\nF-statistic {self.dof_F}: {self.F:.4f}, p-value: {self.F_p:.4f}' +\
            f'\nNumber of observations: {self.N}'

        s += rsq
        
        return s
    
    def plotfit(self, ax=None, ls='-'):

        legend=False
        if ax is None:
            legend = True
            fig, ax = plt.subplots()

        ax.plot(np.arange(self.N), self.y_hat, ls=ls, marker='x', label='Fitted')
        ax.plot(np.arange(self.N), self.Y, ls='None', marker='o', label='Actual')

        if legend:
            ax.legend()
        
        ax.set_xlabel('Case Number')
        ax.set_ylabel(self.ylabel)

    def plotresid(self, ax=None, ls='-'):

        legend=False
        if ax is None:
            legend = True
            fig, ax = plt.subplots()

        ax.plot(np.arange(self.N), self.residuals, ls='None', marker='o', label='Residual')
        ax.axhline(0, color='black', ls=':')

        if legend:
            ax.legend()
        ax.set_xlabel('Case Number')
        ax.set_ylabel('Residual')

    def to_latex(self):

        ltx = r'\begin{tabular}{lrrrr}' +'\n' \
            r'\multicolumn{5}{c}{Ordinary Least Squares}\\'+'\n' \
            r'\hline \hline'
        ltx += '\n' +' & '.join(['Variable', 'Coef', 'SE-Coef', 'T-stat', 'P-stat']) +r'\\'
        ltx += '\n' + r'\hline'
        for i in range(self.nparams):
            ltx += '\n' + ' & '.join([self.beta_names[i], f'{self.beta_hat[i,0]:.2f}',
                            f'{self.beta_hat_sigma[i]:.2f}', f'{self.beta_hat_t[i]:.2f}',
                            f'{self.beta_hat_p[i]:.4f}']) + r' \\'
        
        ltx += '\n' + r'\hline'
        #ltx += r'\\'
        ltx += '\n'+ r'\multicolumn{5}{l}{' + f'R-squared: {self.r_squared:.4f}, R-bar-squared: {self.rbar_squared:.4f}' +r'} \\'
        ltx += '\n'+ r'\multicolumn{5}{l}{' + f'F-statistic {self.dof_F}: {self.F:.4f}, p-value: {self.F_p:.4f}' + r'} \\'
        ltx += '\n'+ r'\multicolumn{5}{l}{' + f'Number of observations: {self.N}' +'}'

        ltx += '\n' + r'\end{tabular}'
        
        return ltx